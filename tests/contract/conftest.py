"""契约测试夹具：两个后端各建一套真实资源，用完清干净。

三条硬规则（都是这次修复踩出来的）：
1. **拿不到 Postgres 时，CI 里必须红，不能 skip。** 否则「契约测试全绿」可以全靠
   跳过得到，这个门禁就白加了。本地开发没起 Docker 才降级为 skip。
2. **不下载 embedding 模型。** 契约测的是 SQL / schema / kb 过滤 / 分数口径 /
   删除级联，与向量质量无关，用确定性伪向量既快又可复现。真模型冒烟单独打
   `contract_model` 标记，默认不跑。
3. **每个用例自己清自己的数据。** 两个后端共用一个 PG 实例，脏数据会让下一个
   用例的 count() 断言随机失败。
"""
import hashlib
import math
import random
from typing import List, Optional

import pytest

from app.config import settings

BACKENDS = ["pg", "chroma"]


# ---------------------------------------------------------------------------
# PG 可用性探测
# ---------------------------------------------------------------------------
_pg_state: Optional[dict] = None


def _probe_pg() -> dict:
    """连一次 DSN，把结果缓存住（每个 pytest 进程只探一次）。"""
    global _pg_state
    if _pg_state is not None:
        return _pg_state
    try:
        from psycopg import connect

        with connect(settings.postgres_dsn, autocommit=True, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        _pg_state = {"ok": True}
    except Exception as exc:  # noqa: BLE001 - 探测失败的原因要原样报告给使用者
        _pg_state = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return _pg_state


def _require_pg_or_skip() -> None:
    """PG 不可用时的处理：CI 里判失败，本地才跳过。

    为什么不一律 skip：这套测试存在的理由就是「CI 全绿但生产是坏的」，
    如果服务没起来就静默跳过，绿灯的含义会和以前一模一样。
    """
    state = _probe_pg()
    if state["ok"]:
        return
    detail = (
        f"PostgreSQL 不可用（{state['error']}）。本地请先 docker compose up -d；"
        "或跑离线套件：pytest tests -m 'not contract'"
    )
    import os

    if os.environ.get("CI") or os.environ.get("RAG_CONTRACT_STRICT"):
        pytest.fail(detail, pytrace=False)
    pytest.skip(detail)


# ---------------------------------------------------------------------------
# 确定性伪向量
# ---------------------------------------------------------------------------
class StubEmbeddings:
    """按文本内容生成固定的单位向量（LangChain Embeddings 鸭子类型）。

    同一段文本 → 同一向量，所以两个后端拿到的是完全相同的输入，
    跨后端比分数才是有意义的（比的正是两侧的距离→分数换算，而不是模型差异）。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rnd = random.Random(seed)
        v = [rnd.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# 构造真实后端
# ---------------------------------------------------------------------------
def build_registry(backend: str, tmp_path, monkeypatch):
    """返回该后端的注册中心实例（绕开进程级单例工厂，测试之间不串）。"""
    if backend == "pg":
        _require_pg_or_skip()
        from app.rag.registry import PGKBRegistry

        return PGKBRegistry(settings.postgres_dsn)
    from app.rag.registry import SQLiteKBRegistry

    return SQLiteKBRegistry(str(tmp_path / "registry.db"))


def build_store(backend: str, tmp_path, monkeypatch, collection: str):
    """返回该后端的向量库实例，embedding 换成伪向量。

    monkeypatch 的是 `app.rag.vector_store.BGEEmbeddings` 这个名字——两个 Store 的
    __init__ 里都会 `BGEEmbeddings()`，而 PGvectorStore 一旦真构造就会去下载
    ~95MB 的 bge 模型，CI 不该为此付网络与磁盘代价。
    """
    from app.rag import vector_store as vs

    monkeypatch.setattr(
        vs, "BGEEmbeddings", lambda *a, **k: StubEmbeddings(settings.embed_dim)
    )
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "collection_name", collection)

    if backend == "pg":
        _require_pg_or_skip()
        return vs.PGvectorStore()
    return vs.ChromaStore()


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def registry(backend, tmp_path, monkeypatch):
    before = _pg_snapshot(backend)
    reg = build_registry(backend, tmp_path, monkeypatch)
    yield reg
    _restore_pg_snapshot(backend, before)


def _safe_collection(name: str) -> str:
    """Chroma 的集合名校验很严：只允许 [a-zA-Z0-9._-]，且首尾必须是字母或数字。

    测试节点名长这样：test_kb_isolation_in_search[chroma] —— 直接拿来当集合名会
    被拒。所以净化字符 + 拼一段哈希保证唯一，避免不同用例复用同一个集合。
    """
    import re

    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-")[:80] or "case"
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"contract-{slug}-{digest}"


@pytest.fixture
def store(backend, tmp_path, monkeypatch, request):
    before = _pg_snapshot(backend)
    st = build_store(
        backend, tmp_path, monkeypatch, collection=_safe_collection(request.node.name)
    )
    yield st
    _restore_pg_snapshot(backend, before)


# ---------------------------------------------------------------------------
# PG 数据隔离：只回收本用例新增的行
# ---------------------------------------------------------------------------
# 为什么不用「DELETE FROM chunks」这种一把梭的清理：那等于假设这个库是专用测试库。
# 一旦有人把 POSTGRES_DSN 指到一个带真实数据的实例上，跑一遍契约测试就把人家的库清了。
# 所以改成用例前后各拍一次快照，只删差集里的行。
def _pg_snapshot(backend: str) -> dict:
    if backend != "pg" or not _probe_pg()["ok"]:
        return {}
    from psycopg import connect

    with connect(settings.postgres_dsn, autocommit=True) as conn:
        # 快照发生在 fixture 建表之前——第一次跑时 kbs/chunks 可能还不存在，
        # 那就等价于「基线为空」，不能用它把用例打成 ERROR
        exists = conn.execute("SELECT to_regclass('public.kbs')").fetchone()[0]
        if exists is None:
            return {"kbs": set(), "chunks": set(), "docs": set()}
        kbs = {r[0] for r in conn.execute("SELECT kb_id FROM kbs").fetchall()}
        chunks = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
        docs = {r[0] for r in conn.execute("SELECT doc_id FROM documents").fetchall()}
    return {"kbs": kbs, "chunks": chunks, "docs": docs}


def _restore_pg_snapshot(backend: str, before: dict) -> None:
    if not before:
        return
    try:
        from psycopg import connect

        with connect(settings.postgres_dsn, autocommit=True) as conn:
            now_kbs = {r[0] for r in conn.execute("SELECT kb_id FROM kbs").fetchall()}
            now_chunks = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
            new_chunks = list(now_chunks - before["chunks"])
            new_kbs = list(now_kbs - before["kbs"])
            if new_chunks:
                conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (new_chunks,))
            if new_kbs:
                conn.execute("DELETE FROM documents WHERE kb_id = ANY(%s)", (new_kbs,))
                conn.execute("DELETE FROM kbs WHERE kb_id = ANY(%s)", (new_kbs,))
    except Exception:  # noqa: BLE001 - 清理失败不应把已通过的结果改写成错误
        pass
