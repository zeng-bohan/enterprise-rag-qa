"""文档摄取：解析 → 切片 → 向量化 → 事务写入 → 检索失效（工单 12/13/18）。

抽成独立模块的理由
------------------
这段流程有两个调用方：HTTP 端点（同步模式）与 arq worker（异步模式）。
原先它内联在 `app/api/manage.py` 里，一旦加 worker 就必须复制一份，而两份的失败
语义迟早会分叉。所以逻辑放这里，两个调用方只负责"在哪儿执行"。

三个非显然的设计点
------------------
1. **先算向量，再开事务**（`embed_for` 与 `add_documents_on` 分开）。批量 ONNX
   推理可能几十秒，如果这段时间连接被事务按着，8 条连接能被几个上传占光，
   在线检索会饿死在连接池上。
2. **双 PG 才走真事务**。本地模式是 Chroma + SQLite 两个引擎，跨引擎没有事务，
   只能顺序写 + 失败补偿。契约测试断言的是**可观察结果**（要么全成、要么
   状态 failed 且零 chunk），不是"有没有 BEGIN"，这样两个后端才能共用一套断言。
3. **写完必须发失效信号**（`bump_kb_version`）。否则把摄取移进 worker 进程后，
   API 进程里的关键词索引永远不会重建 —— 表现为"文档传上去了但搜不到"，
   且重启又好了。这是工单 13 存在的全部原因。
"""
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.core.kb_version import bump_kb_version
from app.rag.chunker import split_documents
from app.rag.document_loader import load_document


def prepare_chunks(path: Path) -> List[Document]:
    """文件 → chunk 列表（解析 + 切片），不做任何写入。"""
    return split_documents(load_document(path))


def _write_atomic(store, registry, kb_id: str, doc_id: str, chunks: List[Document],
                  vectors: List[List[float]]) -> None:
    """chunks 与 documents 状态推进放进同一个事务（仅双 PG 后端）。"""
    with store.transaction() as conn:
        store.add_documents_on(conn, chunks, kb_id, doc_id, vectors)
        registry.update_document_on(conn, kb_id, doc_id, status="indexed", chunk_count=len(chunks))
        # 提交由 transaction() 的上下文负责；这里再 conn.commit() 会变成事务里套事务


def _write_best_effort(store, registry, kb_id: str, doc_id: str, chunks: List[Document],
                       vectors: Optional[List[List[float]]]) -> None:
    """跨引擎没有事务：顺序写入，失败方由调用方落 failed 状态。"""
    if vectors is None:
        vectors = store.embed_for(chunks)
    store.add_documents(chunks, kb_id=kb_id, doc_id=doc_id)
    registry.update_document(kb_id, doc_id, status="indexed", chunk_count=len(chunks))


def ingest_document(store, registry, kb_id: str, doc_id: str, path: Path,
                    chunk_limit: int = 0) -> dict:
    """把一个已落盘的文件索引进知识库，返回最终文档记录。

    chunk_limit > 0 时截断（压测 / 超大文件冒烟用），0 表示不限制。
    """
    chunks = prepare_chunks(path)
    if chunk_limit:
        chunks = chunks[:chunk_limit]
    if not chunks:
        registry.update_document(kb_id, doc_id, status="failed", chunk_count=0,
                                 error="解析结果为空（扫描件或缺少 ToUnicode 映射的 PDF？）")
        raise ValueError("文档解析后没有可索引的内容")

    if registry.shares_pool_with(store):  # type: ignore[attr-defined]
        # 推理在事务之外完成，连接只在写入期间被占用
        vectors = store.embed_for(chunks)
        _write_atomic(store, registry, kb_id, doc_id, chunks, vectors)
        record = registry.get_document(kb_id, doc_id)
    else:
        _write_best_effort(store, registry, kb_id, doc_id, chunks, None)
        record = registry.get_document(kb_id, doc_id)

    bump_kb_version(kb_id)
    return record


def delete_document(store, registry, retriever, kb_id: str, doc_id: str) -> Tuple[int, int]:
    """删除文档的完整语义：元数据 + chunk 血缘 + 检索失效。

    返回 (registry 删除, store 删除) 两个计数，便于调用方核对是否真的对齐 ——
    半态（元数据没了但 chunk 还在）是这里最危险的故障。
    """
    registry.delete_document(kb_id, doc_id)
    removed_chunks = store.delete_by_doc(doc_id, kb_id=kb_id)
    bump_kb_version(kb_id)
    if retriever is not None:
        retriever.invalidate(kb_id)
    return removed_chunks
