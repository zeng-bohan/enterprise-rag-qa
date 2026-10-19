"""本地模式（Chroma + SQLite）注册中心的并发用例。

管理接口把注册中心调用丢进 4 线程的元数据池。sqlite3 的 execute 与 commit 是
两步，单连接被多线程共享时，两个线程的语句会卷进同一个隐式事务：先提交的把
两条一起提交，后一个再 commit 就抛「cannot commit - no transaction is active」
或 InterfaceError。修复前 8 线程 × 20 次建库稳定复现 7 个线程报错。
"""
import threading

from app.rag.registry import SQLiteKBRegistry


def test_concurrent_create_kb_does_not_raise(tmp_path):
    reg = SQLiteKBRegistry(str(tmp_path / "registry.db"))
    errors = []
    per_thread, threads_n = 15, 8

    def work(tid: int) -> None:
        try:
            for i in range(per_thread):
                reg.create_kb(f"kb-{tid}-{i}", "")
        except Exception as exc:  # noqa: BLE001 - 用例要把原始异常类型报出来
            errors.append((tid, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"并发建库报错：{errors[:3]}"
    assert len(reg.list_kbs()) == per_thread * threads_n


def test_concurrent_document_writes_keep_records_consistent(tmp_path):
    reg = SQLiteKBRegistry(str(tmp_path / "registry.db"))
    kb_id = reg.create_kb("docs")["kb_id"]
    errors = []

    def work(tid: int) -> None:
        try:
            for i in range(10):
                rec = reg.add_document(kb_id, f"f-{tid}-{i}.md", chunk_count=0, status="queued")
                reg.update_document(kb_id, rec["doc_id"], status="indexed", chunk_count=3)
        except Exception as exc:  # noqa: BLE001
            errors.append((tid, type(exc).__name__, str(exc)))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"并发写文档报错：{errors[:3]}"
    docs = reg.list_documents(kb_id)
    assert len(docs) == 60
    assert all(d["status"] == "indexed" and d["chunk_count"] == 3 for d in docs)
