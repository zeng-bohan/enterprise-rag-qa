"""管理接口：知识库 / 文档生命周期与召回测试（offline：SQLite 注册中心 + 桩 store）。"""
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
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

    def add_documents(self, docs, kb_id: str, doc_id: str) -> int:
        self.ingested.append((kb_id, doc_id, len(docs)))
        return len(docs)

    def delete_by_doc(self, doc_id: str) -> int:
        self.deleted_docs.append(doc_id)
        return 1

    def delete_by_kb(self, kb_id: str) -> int:
        self.deleted_kbs.append(kb_id)
        return 1

    def count(self, kb_id=None) -> int:
        return sum(n for kb, _, n in self.ingested if kb_id is None or kb == kb_id)


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
    assert store.deleted_docs == [body["doc_id"]]
    r = client.delete(f"/v1/kbs/{kb_id}/documents/{body['doc_id']}")
    assert r.status_code == 404


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
