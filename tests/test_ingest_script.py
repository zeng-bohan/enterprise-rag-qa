"""ingest_docs.py 的单文件流程：成功与失败都只留一条文档记录。

回归点：旧实现先按 chunk 数登记、写入失败后再 add_document 一次，同一次失败
在文档列表里出现两条记录（一条 indexed 且声称有 chunk，一条 failed）。
"""
from pathlib import Path

import pytest

from app.rag import ingest as ingest_mod
from app.rag.registry import SQLiteKBRegistry
from scripts.ingest_docs import ingest_file


class _Store:
    """最小向量库桩：shares_pool_with 走基类默认 False，即本地补偿写路径。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.written = 0

    def embed_for(self, docs):
        return [[0.0] * 4 for _ in docs]

    def add_documents(self, docs, kb_id, doc_id):
        if self.fail:
            raise RuntimeError("模拟向量化失败")
        self.written += len(docs)
        return len(docs)


class _Pipe:
    def __init__(self, registry, store) -> None:
        self.registry = registry
        self.store = store


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    # 版本号写入只影响跨进程失效，单测里不连 Redis
    monkeypatch.setattr(ingest_mod, "bump_kb_version", lambda kb_id: None)


def _write_doc(tmp_path: Path) -> Path:
    path = tmp_path / "员工手册.md"
    path.write_text("员工年假 15 天。\n" + "请假须提前三个工作日申请。\n" * 30, encoding="utf-8")
    return path


def test_ingest_file_success_creates_one_indexed_record(tmp_path):
    reg = SQLiteKBRegistry(str(tmp_path / "r.db"))
    kb_id = reg.create_kb("kb")["kb_id"]
    store = _Store()

    record = ingest_file(_Pipe(reg, store), kb_id, _write_doc(tmp_path))

    docs = reg.list_documents(kb_id)
    assert len(docs) == 1
    assert docs[0]["status"] == "indexed"
    assert docs[0]["chunk_count"] == record["chunk_count"] > 0
    assert store.written == record["chunk_count"]


def test_ingest_file_failure_leaves_exactly_one_failed_record(tmp_path):
    reg = SQLiteKBRegistry(str(tmp_path / "r.db"))
    kb_id = reg.create_kb("kb")["kb_id"]

    with pytest.raises(RuntimeError, match="模拟向量化失败"):
        ingest_file(_Pipe(reg, _Store(fail=True)), kb_id, _write_doc(tmp_path))

    docs = reg.list_documents(kb_id)
    assert len(docs) == 1, f"失败后留下 {len(docs)} 条记录"
    assert docs[0]["status"] == "failed"
    assert docs[0]["chunk_count"] == 0
    assert "模拟向量化失败" in docs[0]["error"]
