"""管理接口层：知识库 / 文档生命周期 + 召回测试。

对齐主流企业知识库产品的管理面（Dify / FastGPT 的「知识库」页）：

  POST   /v1/kbs                             创建知识库
  GET    /v1/kbs                             知识库列表（含文档数 / chunk 数）
  DELETE /v1/kbs/{kb_id}                     删除知识库（连带 chunks）
  POST   /v1/kbs/{kb_id}/documents           上传文档（同步解析→切片→索引）
  GET    /v1/kbs/{kb_id}/documents           文档列表（状态 / chunk 数）
  DELETE /v1/kbs/{kb_id}/documents/{doc_id}  删除文档（连带其 chunks）
  POST   /v1/retrieval-test                  召回测试：只检索不生成（调参用）

实现约定：
- 元数据走 registry（PG 表 / SQLite），chunk 血缘写进向量库（kb_id / doc_id）；
- 删除 = registry 元数据 + store 血缘清理两步，BM25 索引随后失效；
- 上传为同步索引（小文档秒级）；大文件异步摄取在 roadmap；
- 阻塞 IO（registry / store / 解析切片）经 run_cpu 进线程池，不卡事件循环。
"""
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import settings
from app.core.auth import require_api_key
from app.core.executor import run_cpu
from app.api.deps import pipeline as qa_pipeline
from app.rag.chunker import split_documents
from app.rag.document_loader import SUPPORTED_SUFFIXES, load_document
from app.rag.generator import build_context
from app.rag.registry import KBError
from app.schemas import CreateKBRequest, RetrievalTestRequest

router = APIRouter(prefix="/v1", tags=["manage"], dependencies=[Depends(require_api_key)])


@router.post("/kbs", status_code=201)
async def create_kb(payload: CreateKBRequest) -> dict:
    try:
        return await run_cpu(qa_pipeline.registry.create_kb, payload.name.strip(), payload.description)
    except KBError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/kbs")
async def list_kbs() -> list[dict]:
    kbs = await run_cpu(qa_pipeline.registry.list_kbs)
    out = []
    for kb in kbs:
        docs = await run_cpu(qa_pipeline.registry.list_documents, kb["kb_id"])
        chunk_count = await run_cpu(qa_pipeline.store.count, kb["kb_id"])
        out.append({**kb, "document_count": len(docs), "chunk_count": chunk_count})
    return out


@router.delete("/kbs/{kb_id}", status_code=204)
async def delete_kb(kb_id: str) -> None:
    try:
        await run_cpu(qa_pipeline.registry.delete_kb, kb_id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await run_cpu(qa_pipeline.store.delete_by_kb, kb_id)
    qa_pipeline.retriever.invalidate(kb_id)


@router.post("/kbs/{kb_id}/documents", status_code=201)
async def upload_document(kb_id: str, file: UploadFile) -> dict:
    """上传并同步索引一个文档（multipart/form-data，字段名 file）。"""
    try:
        await run_cpu(qa_pipeline.registry.get_kb, kb_id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型: {suffix or '(无后缀)'}")

    content = await file.read()
    uploads = Path(settings.chroma_dir).parent / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    tmp = uploads / Path(file.filename).name
    tmp.write_bytes(content)

    def _ingest() -> dict:
        docs = load_document(tmp)
        chunks = split_documents(docs)
        record = qa_pipeline.registry.add_document(kb_id, file.filename, chunk_count=len(chunks))
        qa_pipeline.store.add_documents(chunks, kb_id=kb_id, doc_id=record["doc_id"])
        qa_pipeline.retriever.invalidate(kb_id)
        return record

    try:
        return await run_cpu(_ingest)
    except HTTPException:
        raise
    except Exception as exc:
        # 解析 / 索引失败：记录 failed 文档（可追溯），对外 500
        qa_pipeline.registry.add_document(
            kb_id, file.filename, chunk_count=0, status="failed", error=str(exc)[:500]
        )
        raise HTTPException(status_code=500, detail=f"文档索引失败: {exc}")


@router.get("/kbs/{kb_id}/documents")
async def list_documents(kb_id: str) -> list[dict]:
    try:
        await run_cpu(qa_pipeline.registry.get_kb, kb_id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return await run_cpu(qa_pipeline.registry.list_documents, kb_id)


@router.delete("/kbs/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(kb_id: str, doc_id: str) -> None:
    try:
        await run_cpu(qa_pipeline.registry.delete_document, kb_id, doc_id)
    except KBError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    await run_cpu(qa_pipeline.store.delete_by_doc, doc_id)
    qa_pipeline.retriever.invalidate(kb_id)


@router.post("/retrieval-test")
async def retrieval_test(payload: RetrievalTestRequest) -> dict:
    """召回测试：只检索不生成（对齐 Dify / FastGPT 的「命中测试」）。"""
    t0 = time.perf_counter()
    hits = await qa_pipeline.retriever.aretrieve(payload.query, kb_id=payload.kb_id, top_k=payload.top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    _, citations = build_context(hits)
    return {"hits": citations, "retrieval_ms": retrieval_ms}
