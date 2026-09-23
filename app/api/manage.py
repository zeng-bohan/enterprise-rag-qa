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
- 上传为同步索引（小文档秒级）；大文件异步摄取在工单 12；
- 阻塞 IO 分三个池（app/core/executor.py，工单 14）：注册中心与纯 SQL 计数走
  run_meta，解析/切片/批量 embed 的摄取走 run_ingest，只有推理与检索计算走 run_cpu。
  分池的理由是避免互相排队 —— 一次上传拖爆在线查询 P95 是原来最容易踩的模式。

错误语义（工单 02）：
- KBError      → 404 / 409：业务上就不该有这个结果；
- BackendError → 503：存储后端连不上 / SQL 非法 / 表缺失。
  这条区分是本次修复加进来的——之前所有异常都被揉成「重名？」之类的业务话术，
  一个 psycopg 客户端语法错误能把人往数据问题上带偏半小时。
"""
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.config import settings
from app.core.auth import require_api_key, require_rate_limit
from app.core.executor import run_cpu, run_ingest, run_meta
from app.core.metrics import counted
from app.api.deps import pipeline as qa_pipeline
from app.api.errors import backend_errors
from app.rag.chunker import split_documents
from app.rag.document_loader import SUPPORTED_SUFFIXES, load_document
from app.rag.generator import build_context
from app.rag.registry import KBError
from app.schemas import CreateKBRequest, RetrievalTestRequest

router = APIRouter(
    prefix="/v1",
    tags=["manage"],
    dependencies=[Depends(require_api_key), Depends(require_rate_limit)],
)

# 原始上传件的落盘目录：<chroma_dir 的父目录>/uploads/<doc_id>/<安全文件名>
# 按 doc_id 分目录（工单 04）：同名文件不再互相覆盖，且每个文档的原始件可被独立
# 定位——换切片策略后能原地重灌，不必要求用户重新上传。
UPLOADS_DIR = Path(settings.chroma_dir).parent / "uploads"
MAX_FILENAME_LEN = 120
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_name(raw: str) -> str:
    """把用户提供的文件名压成单层安全文件名。

    Path(...).name 已经挡住了路径穿越（../../etc/passwd → passwd），这里再清掉
    分隔符与控制字符，避免同一批上传里 "a/b.md" 与 "a\\b.md" 落到同一个名字上。

    注意 Path() 本身在含 NUL 字节的输入上会抛 ValueError——那是攻击者可控的输入，
    绝不能让它变成一个 500，所以整段兜底。
    """
    try:
        name = Path(raw or "document").name
    except (ValueError, OSError):
        name = raw or "document"
    name = _UNSAFE.sub("_", name).strip().strip(".")
    if not name:
        name = "document"
    return name[:MAX_FILENAME_LEN]


@router.post("/kbs", status_code=201)
async def create_kb(payload: CreateKBRequest) -> dict:
    name = payload.name.strip()
    with backend_errors():
        try:
            return await run_meta(qa_pipeline.registry.create_kb, name, payload.description)
        except KBError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.get("/kbs")
async def list_kbs() -> list[dict]:
    """知识库列表，带文档数与 chunk 数。

    工单 17：以前是 N+1 —— 每个知识库各发一次 list_documents 和一次 store.count，
    N 个库就是 2N+1 次往返，管理页一开就是几十次查询。现在两条聚合查询取全量，
    在内存里按 kb_id 拼接。
    """
    with backend_errors():
        kbs = await run_meta(qa_pipeline.registry.list_kbs)
        doc_counts = await run_meta(qa_pipeline.registry.document_counts)
        chunk_counts = await run_meta(qa_pipeline.store.counts_by_kb)
    return [
        {
            **kb,
            "document_count": doc_counts.get(kb["kb_id"], 0),
            "chunk_count": chunk_counts.get(kb["kb_id"], 0),
        }
        for kb in kbs
    ]


@router.delete("/kbs/{kb_id}", status_code=204)
async def delete_kb(kb_id: str) -> None:
    with backend_errors():
        try:
            await run_meta(qa_pipeline.registry.delete_kb, kb_id)
        except KBError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        await run_meta(qa_pipeline.store.delete_by_kb, kb_id)
    qa_pipeline.retriever.invalidate(kb_id)


async def _read_capped(file: UploadFile) -> bytes:
    """分块读到 MAX_UPLOAD_MB 为止，超了就 413。

    原来是一句 `await file.read()`：客户端 POST 一个 2GB 文件就会把它整个吃进内存，
    一个请求即可打爆进程。分块读的好处是**在超限那一刻就停止读取并返回**，
    不必先把整个请求体收完再拒绝。
    """
    limit = settings.max_upload_mb * 1024 * 1024
    chunks, total = [], 0
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        total += len(part)
        if limit and total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过上限 {settings.max_upload_mb}MB（已收到 {total} 字节）",
            )
        chunks.append(part)
    if limit and total > limit:
        raise HTTPException(status_code=413, detail=f"文件超过上限 {settings.max_upload_mb}MB")
    return b"".join(chunks)


@router.post("/kbs/{kb_id}/documents", status_code=201)
async def upload_document(kb_id: str, file: UploadFile) -> dict:
    """上传并同步索引一个文档（multipart/form-data，字段名 file）。"""
    with backend_errors():
        try:
            await run_meta(qa_pipeline.registry.get_kb, kb_id)
        except KBError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"不支持的文件类型: {suffix or '(无后缀)'}")

        content = await _read_capped(file)

        def _store_original(doc_id: str) -> Path:
            d = UPLOADS_DIR / doc_id
            d.mkdir(parents=True, exist_ok=True)
            path = d / safe_name(file.filename or "")
            path.write_bytes(content)
            return path

        def _ingest() -> dict:
            """先登记 indexing 记录，再索引，最后改写状态。

            旧写法是「add_document(indexed) → store.add_documents，失败再
            add_document(failed)」，于是同一次失败上传留下两条 documents 记录，
            且第一条声称有 N 个 chunk 而实际一个都没写进去（幽灵文档）。
            现在记录只创建一次，状态由 update_document 推进。
            """
            record = qa_pipeline.registry.add_document(
                kb_id, file.filename or "document", chunk_count=0, status="indexing"
            )
            doc_id = record["doc_id"]
            try:
                path = _store_original(doc_id)
                docs = load_document(path)
                chunks = split_documents(docs)
                qa_pipeline.store.add_documents(chunks, kb_id=kb_id, doc_id=doc_id)
            except Exception as exc:
                # 解析 / 索引失败：状态落 failed（可追溯），对外 500
                qa_pipeline.registry.update_document(
                    kb_id, doc_id, status="failed", chunk_count=0, error=str(exc)[:500]
                )
                raise HTTPException(status_code=500, detail=f"文档索引失败: {exc}")
            record = qa_pipeline.registry.update_document(
                kb_id, doc_id, status="indexed", chunk_count=len(chunks)
            )
            qa_pipeline.retriever.invalidate(kb_id)
            return record

        try:
            return await run_ingest(_ingest)
        except HTTPException:
            raise


@router.get("/kbs/{kb_id}/documents")
async def list_documents(kb_id: str) -> list[dict]:
    with backend_errors():
        try:
            await run_meta(qa_pipeline.registry.get_kb, kb_id)
            return await run_meta(qa_pipeline.registry.list_documents, kb_id)
        except KBError as exc:
            raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/kbs/{kb_id}/documents/{doc_id}", status_code=204)
async def delete_document(kb_id: str, doc_id: str) -> None:
    with backend_errors():
        try:
            await run_meta(qa_pipeline.registry.delete_document, kb_id, doc_id)
        except KBError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        await run_meta(qa_pipeline.store.delete_by_doc, doc_id, kb_id)
        original = UPLOADS_DIR / doc_id
        if original.is_dir():
            for f in original.iterdir():
                f.unlink(missing_ok=True)
            original.rmdir()
    qa_pipeline.retriever.invalidate(kb_id)


@router.post("/retrieval-test")
async def retrieval_test(payload: RetrievalTestRequest) -> dict:
    """召回测试：只检索不生成（对齐 Dify / FastGPT 的「命中测试」）。"""
    t0 = time.perf_counter()
    with counted(), backend_errors():
        hits = await qa_pipeline.retriever.aretrieve(payload.query, kb_id=payload.kb_id, top_k=payload.top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    _, citations = build_context(hits)
    return {"hits": citations, "retrieval_ms": retrieval_ms}
