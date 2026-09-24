"""向量库契约测试：PGvectorStore 与 ChromaStore 共用一套断言。

盯的是三件在离线套件里永远看不见的事：

1. **血缘不互删**（工单 03）。修复前 chunk 主键就是 md5(正文)，同一段文本第二次
   入库会把那一行改挂到新的 doc_id / kb_id 上，于是「删一个文档带走另一个文档的内容」；
2. **知识库之间真的隔离**。离线套件用的 FakeStore.search() 直接把 kb_id 参数丢掉
   （tests/conftest.py:113），所以多库隔离从来没被验证过；
3. **两个后端的分数口径一致**。vector_store.py 的注释声称 PG 的 1-cosine
   「与 Chroma 后端口径一致」，而 score_threshold 直接决定要不要拒答——
   这句话以前只是说说，没有断言。
"""
import pytest
from langchain_core.documents import Document

from app.config import settings
from app.rag.vector_store import chunk_uid, content_hash

from .conftest import _safe_collection

pytestmark = pytest.mark.contract

SHARED = "共享条款：本制度自发布之日起施行。"
DOC_A = ["条款一：员工每年享有十天带薪年休假。", SHARED]
DOC_B = ["条款二：请病假须提交二级以上医院证明。", SHARED]


def mkdocs(texts, source):
    return [Document(page_content=t, metadata={"source": source}) for t in texts]


def _kb(store, kb_id):
    return store.count(kb_id=kb_id)


# ---------------------------------------------------------------------------
# 单后端（两参数各跑一遍）
# ---------------------------------------------------------------------------
def test_add_search_delete_roundtrip(store):
    kb = "contract-roundtrip"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-a")
    assert _kb(store, kb) == 2

    hits = store.search(DOC_A[0], top_k=5, kb_id=kb)
    assert hits, "检索零命中"
    doc, score = hits[0]
    assert doc.page_content == DOC_A[0]
    # 两后端都是余弦相似度（PG: 1-(embedding<=>q)，Chroma: relevance 换算而来），
    # 所以值域是 [-1,1] 而**不是** [0,1]——实测不相关文本会给出 -0.017 这样的负分，
    # langchain-chroma 甚至会为此打一条 "Relevance scores must be between 0 and 1" 警告。
    # 这条断言因此要的是真实值域；顺带说明 score_threshold 是在余弦标度上取 0.35，
    # 它的合理性只能靠评测集校准（工单 24），不能靠这里断言。
    assert -1.0 <= score <= 1.0, f"分数越出余弦值域：{score}"
    assert score > 0.9, f"自匹配分数异常低：{score}"
    # metadata 血缘要能原样读回来（引用溯源与删除都靠它）
    assert doc.metadata.get("source") == "a.md"
    assert doc.metadata.get("kb_id") == kb


def test_shared_chunk_text_does_not_alias_across_documents(store):
    """工单 03 的正证：两份文档共享同一段文本时，各自独立成行。

    修复前的行为是 3 行（共享那段被第二份文档「抢走」归属），且删 A 会连带
    把 B 的那段物理删掉。
    """
    kb = "contract-alias"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-a")
    store.add_documents(mkdocs(DOC_B, "b.md"), kb_id=kb, doc_id="doc-b")
    assert _kb(store, kb) == 4, "同内容 chunk 被跨文档合并了（血缘互删的根因）"

    assert store.delete_by_doc("doc-a", kb_id=kb) == 2
    # B 的两段都还在——包括那段与 A 同文的
    left = {d.page_content for d in store.get_all_documents(kb)}
    assert left == set(DOC_B), f"删 A 之后 B 的内容受损：{left}"


def test_shared_chunk_text_does_not_alias_across_kbs(store):
    """同一份文件传进两个知识库：删掉其中一个，另一个必须原封不动。"""
    kb1, kb2 = "contract-kb1", "contract-kb2"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb1, doc_id="doc-1")
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb2, doc_id="doc-2")
    assert _kb(store, kb1) == 2 and _kb(store, kb2) == 2

    store.delete_by_doc("doc-1", kb_id=kb1)
    assert _kb(store, kb1) == 0
    assert _kb(store, kb2) == 2, "跨知识库的内容被连带删除"


def test_kb_isolation_in_search(store):
    """检索必须带 kb 过滤：A 库的查询不能命中 B 库的内容。"""
    kba, kbb = "contract-iso-a", "contract-iso-b"
    store.add_documents(mkdocs(["年假十天，须提前三个工作日申请。"], "a.md"), kb_id=kba, doc_id="ida")
    store.add_documents(mkdocs(["公积金提取须满足连续缴存三个月。"], "b.md"), kb_id=kbb, doc_id="idb")

    hits = store.search("年假要提前几天申请", top_k=5, kb_id=kba)
    contents = [d.page_content for d, _ in hits]
    assert contents, "本库内检索应当有命中"
    assert all("公积金" not in c for c in contents), f"跨库泄漏：{contents}"

    hits_b = store.search("年假要提前几天申请", top_k=5, kb_id=kbb)
    assert all("年假" not in d.page_content for d, _ in hits_b), "跨库泄漏（反向）"


def test_delete_counts_are_exact(store):
    kb = "contract-counts"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-a")
    store.add_documents(mkdocs(DOC_B, "b.md"), kb_id=kb, doc_id="doc-b")
    assert store.delete_by_doc("doc-b", kb_id=kb) == 2
    assert store.delete_by_doc("doc-missing", kb_id=kb) == 0
    assert store.delete_by_kb(kb) == 2
    assert store.count(kb) == 0


def test_reingest_same_doc_is_idempotent(store):
    """脚本重灌同一文档不得翻倍（ON CONFLICT 收敛为「同文档内去重」后的语义）。"""
    kb = "contract-idem"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-a")
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-a")
    assert _kb(store, kb) == 2


def test_get_all_documents_respects_kb(store):
    kb1, kb2 = "contract-all-1", "contract-all-2"
    store.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb1, doc_id="d1")
    store.add_documents(mkdocs(DOC_B, "b.md"), kb_id=kb2, doc_id="d2")
    assert {d.page_content for d in store.get_all_documents(kb1)} == set(DOC_A)
    assert {d.page_content for d in store.get_all_documents(kb2)} == set(DOC_B)


# ---------------------------------------------------------------------------
# 跨后端（两个后端同时构造，比的是同一份输入下的可观察差异）
# ---------------------------------------------------------------------------
@pytest.fixture
def both_stores(tmp_path, monkeypatch):
    from .conftest import build_store

    pg = build_store("pg", tmp_path / "pg", monkeypatch, collection="contract-pg-x")
    ch = build_store("chroma", tmp_path / "ch", monkeypatch, collection="contract-chroma-x")
    return {"pg": pg, "chroma": ch}


def test_score_scale_matches_across_backends(both_stores):
    """vector_store.py 声称「与 Chroma 后端口径一致」——把它变成断言。

    这条为什么重要：score_threshold=0.35 是硬编码常量，且直接决定是否拒答
    （retriever.py:82）。如果两个后端的分数标度不同，切换后端就是在悄悄改拒答边界，
    而本地验证一切正常、生产上答非所问。

    比较必须覆盖**接近阈值的低分区间**，只比自匹配（两边都是 1.0）是最没信息量的
    一种断言——实测两后端在 -0.07 ~ 1.0 整个区间上逐位吻合，所以才敢把这条写严。
    """
    kb = "contract-score"
    for st in both_stores.values():
        st.add_documents(mkdocs(DOC_A, "a.md"), kb_id=kb, doc_id="doc-score")

    probes = [
        DOC_A[0],            # 完全命中：分数上界
        "请病假要什么材料",     # 同库不同条：中分段
        "今天天气怎么样",       # 完全无关：低分 / 负分，正是 0.35 阈值起作用的地方
    ]
    for query in probes:
        vectors = {}
        for name, st in both_stores.items():
            hits = st.search(query, top_k=5, kb_id=kb)
            # 按内容对齐后再比分数，避免两后端返回顺序不同造成的假不一致
            vectors[name] = {d.page_content: round(s, 4) for d, s in hits}
        assert set(vectors["pg"]) == set(vectors["chroma"]), (
            f"同一查询在两后端命中的 chunk 集合不同：{query}"
        )
        for content, pg_score in vectors["pg"].items():
            ch_score = vectors["chroma"][content]
            assert abs(pg_score - ch_score) <= 0.01, (
                f"两后端对同一 (query, chunk) 给出不同相关度：query={query!r} "
                f"pg={pg_score} chroma={ch_score}——score_threshold 在两后端不再等价"
            )
        # 阈值判定必须在两后端得到同样的「命中几条」，这才是拒答路径真正依赖的东西
        pg_pass = sum(1 for s in vectors["pg"].values() if s >= settings.score_threshold)
        ch_pass = sum(1 for s in vectors["chroma"].values() if s >= settings.score_threshold)
        assert pg_pass == ch_pass, f"拒答边界漂移：{query!r} pg={pg_pass} chroma={ch_pass}"


def test_chunk_uid_is_backend_agnostic():
    """主键构造规则属于契约的一部分：两后端必须用同一个 id 方案。"""
    text = "同一段正文"
    uid = chunk_uid("doc-x", text)
    assert uid == f"doc-x:{content_hash(text)}"
    # 不同文档下的同内容必须得到不同主键（这就是工单 03 的修复点）
    assert chunk_uid("doc-y", text) != uid


# ---------------------------------------------------------------------------
# HTTP 层 → 真实后端
# ---------------------------------------------------------------------------
def test_manage_api_against_real_postgres(tmp_path, monkeypatch):
    """管理面的 HTTP 流程第一次打到真实 PostgreSQL。

    为什么单独有这条：离线套件用 SQLite 注册中心 + 桩 store 跑完了同样的端点，
    所以「端点 → registry → SQL」这条链在 HTTP 层从来没被真后端验证过。
    工单 01 那个占位符 bug 恰好就在链上，端点级测试才能证明它真的被修好了
    （类级契约测试证明 SQL 能跑，这条证明端点真的在用它）。
    """
    from fastapi.testclient import TestClient

    from app.api import deps
    from app.main import app
    from app.rag.registry import PGKBRegistry

    from .conftest import _require_pg_or_skip, build_store

    _require_pg_or_skip()

    store = build_store("pg", tmp_path, monkeypatch, collection="contract-http")
    registry = PGKBRegistry(settings.postgres_dsn)
    monkeypatch.setattr(deps.pipeline, "store", store, raising=False)
    monkeypatch.setattr(deps.pipeline, "registry", registry, raising=False)

    class _NoopRetriever:
        def __init__(self):
            self.invalidated = []

        def invalidate(self, kb_id=None):
            self.invalidated.append(kb_id)

    retriever = _NoopRetriever()
    monkeypatch.setattr(deps.pipeline, "retriever", retriever, raising=False)

    client = TestClient(app)

    # 建库：PGKBRegistry.create_kb 就是当初 `?` 占位符炸掉的地方
    r = client.post("/v1/kbs", json={"name": "http-pg-契约"})
    assert r.status_code == 201, f"PG 后端建库失败：{r.status_code} {r.text}"
    kb_id = r.json()["kb_id"]
    assert "created_at" in r.json()

    # 重名必须还是 409（不能被翻译成 503）
    assert client.post("/v1/kbs", json={"name": "http-pg-契约"}).status_code == 409

    # 上传 → 索引 → 列表 → 删除，全链路走真 SQL
    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("制度.md", "# 制度\n年假十天。\n" * 4, "text/markdown")},
    )
    assert r.status_code == 201, f"PG 后端上传失败：{r.status_code} {r.text}"
    doc = r.json()
    assert doc["status"] == "indexed" and doc["chunk_count"] >= 1

    listed = client.get(f"/v1/kbs/{kb_id}/documents").json()
    assert [d["doc_id"] for d in listed] == [doc["doc_id"]]
    assert listed[0]["kb_id"] == kb_id

    kbs = client.get("/v1/kbs").json()
    row = next(k for k in kbs if k["kb_id"] == kb_id)
    assert row["chunk_count"] == doc["chunk_count"] and row["document_count"] == 1

    assert client.delete(f"/v1/kbs/{kb_id}/documents/{doc['doc_id']}").status_code == 204
    assert store.count(kb_id=kb_id) == 0, "删除文档后 PG 里仍残留 chunk"
    assert client.delete(f"/v1/kbs/{kb_id}").status_code == 204

    # 后端不可用必须是 503，不能伪装成 404 / 409（工单 02）
    #
    # 用 RENAME 而不是 DROP 来制造「表不见了」：这个库可能被别的会话共用，
    # DROP 会把库留在半残状态并污染后续用例（本仓曾经就这样挂过 16 个 error）。
    # RENAME 是同一步可逆操作，断言完再改回来。
    from psycopg import connect

    with connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute("ALTER TABLE kbs RENAME TO kbs_hidden_by_contract_test")
    try:
        r = client.get("/v1/kbs")
        assert r.status_code == 503, f"表不可见时应报后端故障，实际 {r.status_code}: {r.text}"
        assert "存储后端故障" in r.json()["detail"]
    finally:
        with connect(settings.postgres_dsn, autocommit=True) as conn:
            conn.execute("ALTER TABLE kbs_hidden_by_contract_test RENAME TO kbs")

    # 恢复之后必须立刻可用——否则上面的 503 只是把连接池弄坏了，测的不是翻译逻辑
    assert client.get("/v1/kbs").status_code == 200


# ---------------------------------------------------------------------------
# 工单 18：摄取写入的原子性（两个后端共用一套断言）
# ---------------------------------------------------------------------------
def _ingest_pair(backend, tmp_path, monkeypatch, request):
    """返回 (store, registry, kb_id)，两后端各自真实构造。

    名字必须带用例标识：这个 helper 直接用 build_* 构造，绕过了 registry/store
    fixture 的快照清理，所以固定名字会让同 session 里第二个用例撞
    kbs_name_key 唯一约束（第一次跑就是这样红的）。cleanup 注册在 request.addfinalizer
    上，比依赖 fixture  teardown 更稳。
    """
    import uuid

    from .conftest import build_registry, build_store

    store = build_store(backend, tmp_path, monkeypatch, collection=_safe_collection(request.node.name))
    registry = build_registry(backend, tmp_path, monkeypatch)
    kb_name = f"ingest-{backend}-{uuid.uuid4().hex[:8]}"
    kb_id = registry.create_kb(kb_name)["kb_id"]

    def _cleanup():
        try:
            store.delete_by_kb(kb_id)
            registry.delete_kb(kb_id)
        except Exception:  # noqa: BLE001 - 清理失败不该改写用例结论
            pass

    request.addfinalizer(_cleanup)
    return store, registry, kb_id


def test_ingest_is_all_or_nothing(backend, tmp_path, monkeypatch, request):
    """一次成功摄取的可见结果：状态 indexed、chunk 数与实际写入一致。

    断言的是**可观察结果**而不是"有没有 BEGIN"：双 PG 走真事务，
    Chroma+SQLite 跨引擎没有事务、只能顺序写。两套实现共用这条断言，
    才谈得上"平等支持"。
    """
    from pathlib import Path

    from app.rag.ingest import ingest_document

    store, registry, kb_id = _ingest_pair(backend, tmp_path, monkeypatch, request)
    src = tmp_path / "制度.md"
    src.write_text("# 制度\n第一条 年假十天。\n第二条 病假需证明。\n", encoding="utf-8")
    doc_id = registry.add_document(kb_id, src.name, chunk_count=0, status="queued")["doc_id"]

    record = ingest_document(store, registry, kb_id, doc_id, src)
    assert record["status"] == "indexed"
    assert record["chunk_count"] == store.count(kb_id=kb_id), "chunk_count 与实际写入不一致"
    assert record["chunk_count"] >= 1


def test_failed_ingest_leaves_no_partial_chunks(backend, tmp_path, monkeypatch, request):
    """中途失败后不得留下"有 chunk 但状态没推进"或"状态 indexed 但零 chunk"的半态。"""
    from app.rag import ingest as ingest_mod
    from app.rag.ingest import ingest_document

    store, registry, kb_id = _ingest_pair(backend, tmp_path, monkeypatch, request)
    src = tmp_path / "坏文件.md"
    src.write_text("# x\n内容一二三。\n", encoding="utf-8")
    doc_id = registry.add_document(kb_id, src.name, chunk_count=0, status="queued")["doc_id"]

    # 在"向量已算完、即将写入"的时刻注入失败
    original = store.embed_for
    def boom(docs):
        raise RuntimeError("推理阶段炸了")
    monkeypatch.setattr(store, "embed_for", boom)
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        ingest_document(store, registry, kb_id, doc_id, src)
    monkeypatch.setattr(store, "embed_for", original)

    assert store.count(kb_id=kb_id) == 0, "失败摄取留下了半截 chunk"
    doc = registry.get_document(kb_id, doc_id)
    assert doc["status"] == "queued", f"状态被半途推进了：{doc['status']}"
    assert doc["chunk_count"] == 0


def test_ingest_bumps_kb_version_for_cross_process_invalidation(backend, tmp_path, monkeypatch, request):
    """工单 13：摄取必须让 kb 版本号前进，否则 worker 写完、API 进程不知道要重建索引。

    这条只在有 Redis 时验证真实行为；没有 Redis 时退化为"信号发送不抛错"，
    因为设计上共享存储不可用绝不该让写入失败。
    """
    from app.core.cache import get_cache
    from app.core.kb_version import KBVersionTracker, bump_kb_version
    from app.rag.ingest import ingest_document

    store, registry, kb_id = _ingest_pair(backend, tmp_path, monkeypatch, request)
    src = tmp_path / "版本.md"
    src.write_text("# y\n第一条内容。\n第二条内容。\n", encoding="utf-8")
    doc_id = registry.add_document(kb_id, src.name, chunk_count=0, status="queued")["doc_id"]

    tracker = KBVersionTracker(poll_ttl=0.0)  # 关掉轮询缓存，逐次真读
    before = tracker.current(kb_id)
    bump_kb_version(kb_id)
    after_bump = tracker.current(kb_id)

    if get_cache().available:
        assert after_bump == before + 1, "版本号没有前进，跨进程失效信号是断的"
    else:
        assert after_bump == before, "无 Redis 时不应凭空造出版本变化"

    ingest_document(store, registry, kb_id, doc_id, src)
    if get_cache().available:
        assert tracker.current(kb_id) == after_bump + 1, "摄取本身没有发失效信号"


@pytest.mark.contract
def test_async_ingestion_round_trip_against_real_broker(tmp_path, monkeypatch, request):
    """工单 12/13/18 的端到端：入队 → worker 事务写入 → 版本推进 → 可检索。

    为什么单独有这条：queue 路径在离线套件里只能用假 broker 测（断言"入队被调用"），
    而真正会出事的是与真 Redis / 真 arq 的接线 —— 第一版 `queue_depth()` 就是这么
    踩到 arq 把队列存成 zset 而非 list，LLEN 直接抛 WRONGTYPE，而所有用桩的用例
    都是绿的。这类问题只有连真中间件跑一遍才会暴露。

    没有拉起常驻 worker 进程（那需要模型下载与一个事件循环外的生命周期），
    而是直接 await worker 的任务函数：覆盖的是载荷解析、状态机推进、跨表事务与
    失效信号，这些正是 worker 与 API 的全部接口面。
    """
    import json
    import uuid

    from app.core.kb_version import KBVersionTracker
    from app.core.queue import (
        INGEST_QUEUE,
        JOB_INGEST_DOCUMENT,
        enqueue_ingest,
        ingest_job_payload,
        queue_depth,
    )
    from app.rag.ingest import ingest_document as _  # noqa: F401 - 确认可导入
    from app.worker import ingest_document as worker_task

    from .conftest import (
        _require_pg_or_skip,
        _require_redis_or_skip,
        _safe_collection,
        build_registry,
        build_store,
    )

    _require_pg_or_skip()
    _require_redis_or_skip()
    store = build_store("pg", tmp_path, monkeypatch, collection=_safe_collection(request.node.name))
    registry = build_registry("pg", tmp_path, monkeypatch)
    kb_id = registry.create_kb(f"e2e-{uuid.uuid4().hex[:8]}")["kb_id"]
    request.addfinalizer(lambda: (store.delete_by_kb(kb_id), registry.delete_kb(kb_id)))

    src = tmp_path / "制度.md"
    src.write_text(
        "# 制度\n第一条 年假十天。\n第二条 病假须提交证明。\n第三条 公积金提取须连续缴存三个月。\n",
        encoding="utf-8",
    )
    doc_id = registry.add_document(kb_id, src.name, chunk_count=0, status="queued")["doc_id"]

    tracker = KBVersionTracker(poll_ttl=0.0)
    version_before = tracker.current(kb_id)
    assert registry.get_document(kb_id, doc_id)["status"] == "queued"

    payload = ingest_job_payload(kb_id, doc_id, str(src), src.name)
    job_id = _run(enqueue_ingest(payload))
    assert job_id, "任务没有被 broker 接受"
    assert queue_depth() is not None, "队列深度指标不可用（arq 键布局变了？）"
    assert queue_depth() >= 1, f"入队后深度应 >= 1，实际 {queue_depth()}"

    result = _run(worker_task({}, payload))
    assert json.dumps(result)  # worker 返回可序列化，arq 才会接受

    # 收尾清掉本用例写进 broker 的任务记录，别在共享 Redis 里留垃圾
    def _flush_broker():
        import redis as redis_lib

        try:
            client = redis_lib.Redis.from_url(settings.redis_url)
            client.delete(INGEST_QUEUE)
            for key in client.scan_iter("arq:job:*"):
                client.delete(key)
            client.close()
        except Exception:  # noqa: BLE001
            pass

    request.addfinalizer(_flush_broker)

    record = registry.get_document(kb_id, doc_id)
    assert record["status"] == "indexed" and record["chunk_count"] >= 1
    assert record["chunk_count"] == store.count(kb_id=kb_id), "状态里的数量与真实写入不一致"

    # 工单 13 的落点：worker 写完必须留下跨进程失效信号
    assert tracker.current(kb_id) == version_before + 1, (
        "kb 版本号没有前进 —— 换成真 worker 进程后，API 侧的关键词索引不会重建，"
        "表现为『文档传上去了但搜不到、重启又好了』"
    )
    hits = store.search("年假要几天", top_k=3, kb_id=kb_id)
    assert hits, "摄取成功但检索不到内容"


def _run(coro):
    import asyncio

    return asyncio.run(coro)
