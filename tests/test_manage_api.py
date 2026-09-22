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
        self.raise_on_add = False

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

    在 Windows 上 Path(r"a\\b\\c.md").name 得到 "c.md"（反斜杠是分隔符），
    在 Linux 上得到 "a\\b\\c.md" 再由正则替换成下划线。两种都安全，
    所以契约只有一条：结果里绝不能还有目录成分或控制字符。
    """
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
        "x" * 500,
    ]
    for raw in hostile:
        out = safe_name(raw)
        assert out, f"空文件名兜底失败：{raw!r}"
        assert "/" not in out and "\\" not in out, f"{raw!r} → {out!r} 残留目录成分"
        assert ".." not in out, f"{raw!r} → {out!r} 仍可穿越"
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
