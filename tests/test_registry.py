"""注册中心：知识库 / 文档元数据的 CRUD 与级联删除（SQLite 实现）。"""
import pytest

from app.rag.registry import KBError, SQLiteKBRegistry


@pytest.fixture()
def registry(tmp_path):
    return SQLiteKBRegistry(str(tmp_path / "registry.db"))


def test_create_and_get_kb(registry):
    kb = registry.create_kb("帮助中心", "对客知识库")
    assert kb["name"] == "帮助中心"
    got = registry.get_kb(kb["kb_id"])
    assert got["description"] == "对客知识库"


def test_create_duplicate_kb_name_raises(registry):
    registry.create_kb("帮助中心")
    with pytest.raises(KBError):
        registry.create_kb("帮助中心")


def test_get_missing_kb_raises(registry):
    with pytest.raises(KBError):
        registry.get_kb("nope")


def test_list_kbs_sorted_and_complete(registry):
    registry.create_kb("甲")
    registry.create_kb("乙")
    names = [kb["name"] for kb in registry.list_kbs()]
    assert names == ["甲", "乙"]


def test_kb_id_by_name(registry):
    kb = registry.create_kb("帮助中心")
    assert registry.kb_id_by_name("帮助中心") == kb["kb_id"]
    assert registry.kb_id_by_name("不存在") is None


def test_document_lifecycle(registry):
    kb = registry.create_kb("帮助中心")
    doc = registry.add_document(kb["kb_id"], "员工手册.md", chunk_count=12)
    assert doc["status"] == "indexed"
    docs = registry.list_documents(kb["kb_id"])
    assert len(docs) == 1 and docs[0]["chunk_count"] == 12
    got = registry.get_document(kb["kb_id"], doc["doc_id"])
    assert got["filename"] == "员工手册.md"
    registry.delete_document(kb["kb_id"], doc["doc_id"])
    assert registry.list_documents(kb["kb_id"]) == []


def test_delete_missing_document_raises(registry):
    kb = registry.create_kb("帮助中心")
    with pytest.raises(KBError):
        registry.delete_document(kb["kb_id"], "missing")


def test_delete_kb_cascades_documents(registry):
    kb = registry.create_kb("帮助中心")
    registry.add_document(kb["kb_id"], "a.md", 3)
    registry.delete_kb(kb["kb_id"])
    assert registry.list_kbs() == []
    assert registry.kb_id_by_name("帮助中心") is None


def test_ensure_default_is_idempotent(registry):
    first = registry.ensure_default()
    second = registry.ensure_default()
    assert first == second
    assert len(registry.list_kbs()) == 1
