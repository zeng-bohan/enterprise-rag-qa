"""注册中心契约测试：PGKBRegistry 与 SQLiteKBRegistry 必须满足同一份契约。

这组用例的存在价值可以一句话说明：在工单 05 之前，`tests/` 里没有任何一处
引用过 PGKBRegistry，于是 `create_kb` / `add_document` 里 psycopg3 不认的 `?`
占位符（工单 01）能一路留在 main 分支且 CI 全绿。

因此这里刻意不测「能不能跑」，测的是**两个后端的可观察形状必须一致**：
返回哪些键、抛哪一类异常、删除的级联范围。
"""
import pytest

from app.rag.registry import BackendError, KBError

pytestmark = pytest.mark.contract

# 两个后端返回的 dict 键集合必须逐个相等——换后端不该换 API 契约。
# 修复前 PG 侧的读方法一律漏掉 created_at / updated_at，SQLite 侧带着。
KB_KEYS = {"kb_id", "name", "description", "created_at"}
DOC_KEYS = {
    "doc_id",
    "kb_id",
    "filename",
    "status",
    "chunk_count",
    "error",
    "created_at",
    "updated_at",
}


def test_create_and_get_kb_shape_is_backend_independent(registry):
    created = registry.create_kb("契约-人力", "用于契约测试")
    assert set(created) == KB_KEYS, f"create_kb 返回形状漂移：{set(created)}"
    assert created["name"] == "契约-人力"

    fetched = registry.get_kb(created["kb_id"])
    assert set(fetched) == KB_KEYS, f"get_kb 返回形状漂移：{set(fetched)}"
    assert fetched["kb_id"] == created["kb_id"]
    # created_at 必须是字符串（SQLite 存文本、PG 是 TIMESTAMPTZ，两边要归一口径）
    assert isinstance(fetched["created_at"], str) and "T" in fetched["created_at"]


def test_list_kbs_shape_and_membership(registry):
    a = registry.create_kb("契约-A")
    b = registry.create_kb("契约-B")
    kbs = registry.list_kbs()
    assert all(set(k) == KB_KEYS for k in kbs), "list_kbs 与 get_kb 形状不一致"
    names = {k["name"] for k in kbs}
    assert {a["name"], b["name"]} <= names


def test_duplicate_kb_name_is_business_conflict_not_backend_failure(registry):
    """工单 02 的核心断言：重名必须是 KBError，后端故障必须是 BackendError。

    修复前 PG 侧用 `except Exception` 把一切揉成 KBError("…（重名？）")，
    于是工单 01 那个 SQL 语法错误也长得像业务冲突。
    """
    registry.create_kb("契约-重名")
    with pytest.raises(KBError):
        registry.create_kb("契约-重名")


def test_missing_kb_raises_kberror(registry):
    with pytest.raises(KBError):
        registry.get_kb("does-not-exist")
    with pytest.raises(KBError):
        registry.add_document("does-not-exist", "x.md", chunk_count=3)


def test_add_document_shape_and_status_machine(registry):
    """indexing → indexed / failed 的推进（工单 04 的写入次序依赖它）。"""
    kb = registry.create_kb("契约-文档")["kb_id"]
    doc = registry.add_document(kb, "员工手册.md", chunk_count=0, status="indexing")
    assert set(doc) == DOC_KEYS, f"add_document 返回形状漂移：{set(doc)}"
    assert doc["status"] == "indexing"

    done = registry.update_document(kb, doc["doc_id"], status="indexed", chunk_count=7)
    assert done["status"] == "indexed"
    assert done["chunk_count"] == 7
    assert set(done) == DOC_KEYS

    failed = registry.update_document(kb, doc["doc_id"], status="failed", error="解析炸了")
    assert failed["status"] == "failed" and failed["error"] == "解析炸了"
    # 没显式给 chunk_count 时保持原值，不能被悄悄清零
    assert failed["chunk_count"] == 7


def test_delete_document_cascades_only_itself(registry):
    kb = registry.create_kb("契约-删除")["kb_id"]
    d1 = registry.add_document(kb, "a.md", chunk_count=1)["doc_id"]
    d2 = registry.add_document(kb, "b.md", chunk_count=1)["doc_id"]

    registry.delete_document(kb, d1)
    left = {d["doc_id"] for d in registry.list_documents(kb)}
    assert left == {d2}
    with pytest.raises(KBError):
        registry.delete_document(kb, d1)


def test_delete_kb_removes_its_documents(registry):
    kb = registry.create_kb("契约-删库")["kb_id"]
    registry.add_document(kb, "a.md", chunk_count=1)
    registry.delete_kb(kb)
    assert registry.list_documents(kb) == []
    assert kb not in {k["kb_id"] for k in registry.list_kbs()}


def test_default_kb_is_idempotent(registry):
    """ensure_default 会被服务启动调用，重复调用不得产生两条记录。"""
    first = registry.ensure_default()
    second = registry.ensure_default()
    assert first == second


def test_registry_errors_are_not_swallowed_by_translation(registry):
    """KBError 与 BackendError 不能混为一谈（断的是异常类型这条边界）。"""
    assert not issubclass(BackendError, KBError)
    assert issubclass(KBError, RuntimeError) and issubclass(BackendError, RuntimeError)
