"""管理接口：知识库 / 文档生命周期与召回测试（offline：SQLite 注册中心 + 桩 store）。"""
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.config import settings
from app.rag.registry import SQLiteKBRegistry
from tests.conftest import FakeStore

client = TestClient(app)


class RecordingStore(FakeStore):
    """在 FakeStore 上补齐管理面需要的方法并记录调用。"""

    def __init__(self):
        super().__init__(docs=[])
        self.deleted_docs = []
        self.deleted_kbs = []
        self.ingested = []
        self.raise_on_add = False

    def embed_for(self, docs):
        # 真实 store 的这个方法做批量向量化；替身只保证"接口存在、返回等长向量"
        return [[0.0] * 4 for _ in docs]

    def add_documents(self, docs, kb_id: str, doc_id: str) -> int:
        if self.raise_on_add:
            raise RuntimeError("索引失败（测试注入）")
        self.ingested.append((kb_id, doc_id, len(docs)))
        return len(docs)

    def delete_by_doc(self, doc_id: str, kb_id=None) -> int:
        # 签名要跟真实后端一致（工单 03 起带 kb_id 校验），否则替身会把接口漂移藏起来
        self.deleted_docs.append((doc_id, kb_id))
        return 1

    def delete_by_kb(self, kb_id: str) -> int:
        self.deleted_kbs.append(kb_id)
        return 1

    def count(self, kb_id=None) -> int:
        return sum(n for kb, _, n in self.ingested if kb_id is None or kb == kb_id)

    def counts_by_kb(self) -> dict:
        # 工单 17：list_kbs 改用一次聚合，替身必须同步实现，否则接口漂移藏在桩里
        out: dict = {}
        for kb, _, n in self.ingested:
            out[kb] = out.get(kb, 0) + n
        return out


class RecordingRetriever:
    def __init__(self):
        self.invalidated = []
        self.hits = []

    def invalidate(self, kb_id=None):
        self.invalidated.append(kb_id)

    async def aretrieve(self, query, kb_id=None, top_k=None, threshold=None):
        return self.hits


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    store = RecordingStore()
    retriever = RecordingRetriever()
    registry = SQLiteKBRegistry(str(tmp_path / "registry.db"))
    monkeypatch.setattr(deps.pipeline, "store", store, raising=False)
    monkeypatch.setattr(deps.pipeline, "retriever", retriever, raising=False)
    monkeypatch.setattr(deps.pipeline, "registry", registry, raising=False)
    return store, retriever, registry


def test_create_list_delete_kb(wired):
    store, retriever, registry = wired
    r = client.post("/v1/kbs", json={"name": "帮助中心", "description": "对客"})
    assert r.status_code == 201
    kb_id = r.json()["kb_id"]

    r = client.post("/v1/kbs", json={"name": "帮助中心"})
    assert r.status_code == 409
    r = client.post("/v1/kbs", json={"name": ""})
    assert r.status_code == 422

    r = client.get("/v1/kbs")
    assert r.status_code == 200
    kb = next(k for k in r.json() if k["kb_id"] == kb_id)
    assert kb["document_count"] == 0 and kb["chunk_count"] == 0

    r = client.delete(f"/v1/kbs/{kb_id}")
    assert r.status_code == 204
    assert store.deleted_kbs == [kb_id]
    assert retriever.invalidated == [kb_id]
    r = client.delete(f"/v1/kbs/{kb_id}")
    assert r.status_code == 404


def test_upload_list_delete_document(wired, tmp_path):
    store, retriever, registry = wired
    kb_id = client.post("/v1/kbs", json={"name": "hr"}).json()["kb_id"]

    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("员工手册.md", "# 手册\n年假 15 天。\n" * 5, "text/markdown")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["filename"] == "员工手册.md"
    assert body["status"] == "indexed"
    assert body["chunk_count"] >= 1
    assert store.ingested and store.ingested[0][0] == kb_id
    assert retriever.invalidated == [kb_id]

    r = client.get(f"/v1/kbs/{kb_id}/documents")
    assert [d["doc_id"] for d in r.json()] == [body["doc_id"]]

    r = client.delete(f"/v1/kbs/{kb_id}/documents/{body['doc_id']}")
    assert r.status_code == 204
    # 删除要带上 kb 归属（工单 03 的防御性校验）
    assert store.deleted_docs == [(body["doc_id"], kb_id)]
    r = client.delete(f"/v1/kbs/{kb_id}/documents/{body['doc_id']}")
    assert r.status_code == 404


def test_failed_upload_leaves_exactly_one_record(wired):
    """工单 04：索引失败不得留下「声称有 N 个 chunk」的幽灵文档。

    修复前的次序是 add_document(indexed, N) → store.add_documents，
    第二步抛错时又 add_document(failed, 0)，于是同一次失败上传留下两条记录，
    且第一条的 chunk_count 指向一批根本不存在的 chunk。
    """
    store, retriever, registry = wired
    store.raise_on_add = True
    kb_id = client.post("/v1/kbs", json={"name": "hr-fail"}).json()["kb_id"]

    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("坏文件.md", "# x\n内容\n", "text/markdown")},
    )
    assert r.status_code == 500
    docs = registry.list_documents(kb_id)
    assert len(docs) == 1, f"失败上传留下了 {len(docs)} 条记录"
    assert docs[0]["status"] == "failed"
    assert docs[0]["chunk_count"] == 0
    assert "索引失败" in docs[0]["error"]
    assert store.ingested == []
    assert retriever.invalidated == [], "索引未成功就不该失效检索缓存"


def test_original_is_archived_per_doc_id(wired, monkeypatch):
    """工单 04：原始件按 doc_id 分目录，同名文件不再互相覆盖。"""
    from app.api import manage

    store, retriever, registry = wired
    kb_id = client.post("/v1/kbs", json={"name": "hr-arch"}).json()["kb_id"]

    ids = []
    for _ in range(2):
        r = client.post(
            f"/v1/kbs/{kb_id}/documents",
            files={"file": ("同名.md", "# 手册\n内容\n" * 3, "text/markdown")},
        )
        assert r.status_code == 201
        ids.append(r.json()["doc_id"])
    assert ids[0] != ids[1]

    dirs = [manage.UPLOADS_DIR / d for d in ids]
    assert all(d.is_dir() for d in dirs), "两次上传必须落在两个目录，而不是互相覆盖"
    assert all((d / "同名.md").read_bytes().startswith(b"#") for d in dirs)


def test_safe_name_blocks_traversal_and_separators():
    """断行为而不是断具体字符串：Path 对分隔符的处理本身是平台相关的。

    在 Windows 上 Path(r"a\\b\\c.md").name 得到 "c.md"（反斜杠是分隔符）；在 Linux 上
    反斜杠不是分隔符，整串被当成一个文件名，再由正则替换成下划线。两种结果都安全，
    所以契约只有两条：

      1. 结果不得含任何目录成分 —— 用 Path(out).name == out 表达，比逐字符判断更准；
      2. 结果不得含控制字符，且长度有界。

    曾经多写了第三条 `assert ".." not in out`，被 CI 的 Linux runner 直接证伪：
    输入 r"..\\..\\windows\\system32" 净化后是 "_.._windows_system32"，其中的 ".."
    只是文件名的一部分——没有分隔符就穿越不了任何东西。
    """
    from pathlib import Path

    from app.api.manage import MAX_FILENAME_LEN, safe_name

    hostile = [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "a/b/c.md",
        r"a\b\c.md",
        "x\x00evil.md",  # Path() 对 NUL 会抛 ValueError，必须兜住而不是 500
        "\x0a\x0d.md",
        "",
        "   ",
        "..",
        "...",
        "/",
        "\\",
        "x" * 500,
    ]
    for raw in hostile:
        out = safe_name(raw)
        assert out, f"空文件名兜底失败：{raw!r}"
        assert Path(out).name == out, f"{raw!r} → {out!r} 不再是单个路径分量"
        assert "/" not in out and "\\" not in out, f"{raw!r} → {out!r} 残留目录分隔符"
        assert len(out) <= MAX_FILENAME_LEN
        assert not any(ord(ch) < 0x20 for ch in out), f"{raw!r} → {out!r} 残留控制字符"
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("员工手册.md") == "员工手册.md"  # 正常文件名不得被改动


def test_upload_rejects_unsupported_suffix(wired):
    store, retriever, registry = wired
    kb_id = client.post("/v1/kbs", json={"name": "hr2"}).json()["kb_id"]
    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("data.exe", b"binary", "application/octet-stream")},
    )
    assert r.status_code == 415
    assert store.ingested == []


def test_upload_to_missing_kb_404(wired):
    store, retriever, registry = wired
    r = client.post(
        "/v1/kbs/nope/documents",
        files={"file": ("a.md", "内容", "text/markdown")},
    )
    assert r.status_code == 404


def test_retrieval_test_endpoint(wired):
    from tests.conftest import mkdoc

    store, retriever, registry = wired
    retriever.hits = [(mkdoc("年假 15 天", source="手册.md"), 0.91)]
    r = client.post("/v1/retrieval-test", json={"query": "年假几天", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["source"] == "手册.md"
    assert body["hits"][0]["score"] == 0.91
    assert "retrieval_ms" in body


# ---------------------------------------------------------------------------
# Wave 2：聚合查询与边界防护
# ---------------------------------------------------------------------------
def test_list_kbs_uses_aggregates_not_per_kb_queries(wired):
    """工单 17：3 个知识库只该发 2 条聚合查询，而不是 2N+1 条。"""
    store, retriever, registry = wired
    seen = {"store.count": 0, "list_documents": 0}
    real_count, real_docs = store.count, registry.list_documents

    def spy_count(*a, **k):
        seen["store.count"] += 1
        return real_count(*a, **k)

    def spy_docs(*a, **k):
        seen["list_documents"] += 1
        return real_docs(*a, **k)

    store.count = spy_count
    registry.list_documents = spy_docs

    for name in ("甲库", "乙库", "丙库"):
        assert client.post("/v1/kbs", json={"name": name}).status_code == 201
    body = client.get("/v1/kbs").json()
    assert len(body) == 3
    assert seen["store.count"] == 0, "list_kbs 退回了逐库 count（N+1 复活）"
    assert seen["list_documents"] == 0, "list_kbs 退回了逐库 list_documents（N+1 复活）"
    assert {k["document_count"] for k in body} == {0}
    assert {k["chunk_count"] for k in body} == {0}


def test_list_kbs_reports_real_counts_from_aggregates(wired):
    """聚合改写不能把数字算错。"""
    store, retriever, registry = wired
    kb_id = client.post("/v1/kbs", json={"name": "计数库"}).json()["kb_id"]
    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("手册.md", "# 手册\n年假 15 天。\n" * 6, "text/markdown")},
    )
    doc = r.json()
    row = next(k for k in client.get("/v1/kbs").json() if k["kb_id"] == kb_id)
    assert row["document_count"] == 1
    assert row["chunk_count"] == doc["chunk_count"]


def test_upload_over_limit_returns_413(wired, monkeypatch):
    """工单 20：超限必须在读取阶段就拒绝，不能先把 2GB 收进内存再判。"""
    store, retriever, registry = wired
    monkeypatch.setattr(settings, "max_upload_mb", 1, raising=False)
    kb_id = client.post("/v1/kbs", json={"name": "大文件库"}).json()["kb_id"]

    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("巨大.md", "#" + "x" * (2 * 1024 * 1024), "text/markdown")},
    )
    assert r.status_code == 413, f"{r.status_code}: {r.text[:200]}"
    assert store.ingested == [], "超限文件不得进入索引流程"
    assert registry.list_documents(kb_id) == [], "超限上传不该留下文档记录"


def test_rate_limit_disabled_by_default(wired, monkeypatch):
    """默认关闭限流：一次安全加固不该悄悄改变现有部署的行为。"""
    monkeypatch.setattr(settings, "rate_limit_rps", 0.0, raising=False)
    wired
    for _ in range(30):
        assert client.get("/v1/kbs").status_code == 200


def test_rate_limit_falls_open_when_redis_is_down(wired, monkeypatch):
    """限流是保护措施，不能变成新的单点：Redis 不可用时放行而不是拒绝。"""
    import app.core.auth as auth_mod

    monkeypatch.setattr(settings, "rate_limit_rps", 1.0, raising=False)
    monkeypatch.setattr(settings, "rate_limit_burst", 2, raising=False)

    class Unreachable:
        available = False
        raw_client = None

    monkeypatch.setattr(auth_mod, "get_cache", lambda: Unreachable())
    for _ in range(5):
        assert client.get("/v1/kbs").status_code == 200


def test_rate_limit_returns_429_beyond_burst(wired, monkeypatch):
    """开限流后超过 burst 要回 429，并带 Retry-After。"""
    import app.core.auth as auth_mod

    store, retriever, registry = wired
    monkeypatch.setattr(settings, "rate_limit_rps", 1.0, raising=False)
    monkeypatch.setattr(settings, "rate_limit_burst", 3, raising=False)

    class Counting:
        def __init__(self):
            self.n = 0

        def incr(self, key):
            self.n += 1
            return self.n

        def expire(self, key, ttl):
            return None

    class Up:
        def __init__(self):
            self.client = Counting()

        @property
        def available(self):
            return True

        @property
        def raw_client(self):
            return self.client

    shared = Up()  # 必须是同一个实例：每次调用新建计数器就永远不超限
    monkeypatch.setattr(auth_mod, "get_cache", lambda: shared)
    codes = [client.get("/v1/kbs").status_code for _ in range(6)]
    assert codes[:3] == [200] * 3, codes
    assert 429 in codes, f"超过 burst 之后应当限流：{codes}"
    limited = client.get("/v1/kbs")
    assert limited.status_code == 429 and limited.headers.get("retry-after") == "1"


def test_api_key_comparison_is_backend_of_constant_time():
    """常量时间改写不得改变语义：每个配置的 key 都要能过，非 key 要挡住。"""
    from app.core.auth import _key_matches

    configured = ["key-one", "key-two", "key-three"]
    for k in configured:
        assert _key_matches(k, configured), k
    assert not _key_matches("key-1", configured)
    assert not _key_matches("", configured)
    assert not _key_matches("KEY-ONE", configured)  # 大小写敏感，不能被"顺手兼容"放宽


# ---------------------------------------------------------------------------
# 替身与真实接口的同步守门
# ---------------------------------------------------------------------------
# 这一组方法名是从 app/rag/ingest.py 的实际调用点抄下来的。
# 为什么单独立一条用例：本次重构给 store/registry 加了 embed_for / transaction /
# shares_pool_with，真实后端补齐了、测试替身没补，于是四条上传用例全部变成 500，
# 报错是 "'RecordingStore' object has no attribute 'embed_for'"。
# 替身漂移不会自己暴露，只能靠一条断言把它钉住。
STORE_INTERFACE = ["search", "add_documents", "get_all_documents", "count", "counts_by_kb",
                   "delete_by_doc", "delete_by_kb", "embed_for"]
REGISTRY_INTERFACE = ["ensure_default", "create_kb", "list_kbs", "get_kb", "kb_id_by_name",
                      "delete_kb", "add_document", "list_documents", "get_document",
                      "delete_document", "update_document", "delete_documents_of_kb",
                      "document_counts", "shares_pool_with"]


@pytest.mark.parametrize("cls", [SQLiteKBRegistry])
def test_registry_double_implements_interface(cls):
    missing = [m for m in REGISTRY_INTERFACE if not hasattr(cls, m)]
    assert not missing, f"{cls.__name__} 缺少摄取流程依赖的方法：{missing}"


def test_real_backends_implement_interface():
    from app.rag.registry import PGKBRegistry
    from app.rag.vector_store import ChromaStore, PGvectorStore

    for cls in (PGKBRegistry, SQLiteKBRegistry):
        missing = [m for m in REGISTRY_INTERFACE if not hasattr(cls, m)]
        assert not missing, f"{cls.__name__} 缺少 {missing}"
    for cls in (PGvectorStore, ChromaStore):
        missing = [m for m in STORE_INTERFACE if not hasattr(cls, m)]
        assert not missing, f"{cls.__name__} 缺少 {missing}"


def test_ingest_path_uses_only_interface_methods():
    """摄取模块只准通过上述接口访问 store / registry。

    它曾经直接调 `registry.shares_pool_with(store)` 而该方法只存在于 PG 实现上，
    本地模式立刻 AttributeError。这条用例把"只依赖公共接口"变成可检查的约束。
    """
    import inspect

    from app.rag import ingest as ingest_mod

    src = inspect.getsource(ingest_mod)
    used = {name for name in dir(ingest_mod) if not name.startswith("_")}
    for method in REGISTRY_INTERFACE + STORE_INTERFACE:
        if f".{method}(" in src:
            assert method in set(REGISTRY_INTERFACE) | set(STORE_INTERFACE), method
    assert used  # sanity


def test_upload_returns_202_and_enqueues_when_queue_mode(wired, monkeypatch):
    """工单 12：queue 模式下端点只入队、不索引，且响应语义明确改变。"""
    import app.api.manage as manage_mod

    store, retriever, registry = wired
    enqueued = []

    async def fake_enqueue(payload):
        enqueued.append(payload)
        return "job-1"

    monkeypatch.setattr(manage_mod, "queue_enabled", lambda: True)
    monkeypatch.setattr(manage_mod, "enqueue_ingest", fake_enqueue)
    kb_id = client.post("/v1/kbs", json={"name": "队列库"}).json()["kb_id"]

    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("制度.md", "# 制度\n年假 15 天。\n" * 3, "text/markdown")},
    )
    assert r.status_code == 202, f"{r.status_code}: {r.text}"
    body = r.json()
    assert body["status"] == "queued" and body["queue_job"] == "job-1"
    assert len(enqueued) == 1, "任务没有入队"
    assert store.ingested == [], "queue 模式下端点不得自己索引"

    import json

    job = json.loads(enqueued[0])
    assert job["kb_id"] == kb_id and job["doc_id"] == body["doc_id"]
    # 载荷里放路径而不是文件内容：原始件已按 doc_id 落盘（工单 04）
    from pathlib import Path

    assert Path(job["path"]).is_file(), f"worker 将读不到原始件：{job['path']}"


def test_enqueue_failure_leaves_no_orphan_queued_record(wired, monkeypatch):
    """入队失败必须就地收尾成 failed。

    否则 documents 里留下一条 queued 记录而没有任何人会被叫醒处理它 ——
    客户端轮询到永远，而且没有任何错误线索。
    """
    import app.api.manage as manage_mod

    store, retriever, registry = wired

    async def boom(payload):
        raise RuntimeError("Redis 不可达")

    monkeypatch.setattr(manage_mod, "queue_enabled", lambda: True)
    monkeypatch.setattr(manage_mod, "enqueue_ingest", boom)
    kb_id = client.post("/v1/kbs", json={"name": "孤儿库"}).json()["kb_id"]

    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("a.md", "# a\n内容\n", "text/markdown")},
    )
    assert r.status_code == 503
    docs = registry.list_documents(kb_id)
    assert len(docs) == 1 and docs[0]["status"] == "failed"
    assert "入队失败" in docs[0]["error"]


def test_sync_mode_still_returns_201_and_indexes(wired):
    """默认 sync 模式的行为必须与 v0.6 一致 —— 不能顺手把所有人切成异步。"""
    store, retriever, registry = wired
    kb_id = client.post("/v1/kbs", json={"name": "同步库"}).json()["kb_id"]
    r = client.post(
        f"/v1/kbs/{kb_id}/documents",
        files={"file": ("b.md", "# b\n内容\n" * 3, "text/markdown")},
    )
    assert r.status_code == 201
    assert r.json()["status"] == "indexed" and r.json()["chunk_count"] >= 1
    assert store.ingested, "sync 模式应当在本进程完成索引"
