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

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile

from app.config import settings
from app.core.auth import require_api_key, require_rate_limit
from app.core.executor import run_cpu, run_ingest, run_meta
from app.core.metrics import counted
from app.api.deps import pipeline as qa_pipeline
from app.api.errors import backend_errors
from app.core.queue import enqueue_ingest, ingest_job_payload, queue_enabled
from app.rag.document_loader import SUPPORTED_SUFFIXES
from app.rag.ingest import ingest_document
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
async def upload_document(kb_id: str, file: UploadFile, response: Response) -> dict:
    """上传文档。

    两种模式（`INGEST_MODE`，工单 12）：

    - `sync`（默认）：请求内完成解析→切片→向量化→写入，返回 **201**，
      响应里 `status=indexed`、`chunk_count` 立即可得。行为与 v0.6 一致。
    - `queue`：原始件落盘 + 登记 `queued` 记录 + 入队，返回 **202**，
      索引由独立 worker 进程完成，客户端轮询文档详情。

    为什么默认还是 sync：202 是破坏性变更（chunk_count 不再立即可得），
    不该由一次"性能优化"顺手改变所有调用方的行为。生产 compose 里显式开 queue。

    状态机：queued → indexing → indexed | failed。记录只创建一次，
    后续都是 update_document —— 修复前失败路径会再 add_document 一次，
    于是同一次失败上传留下两条记录，第一条还声称有 N 个 chunk。
    """
    with backend_errors():
        try:
            await run_meta(qa_pipeline.registry.get_kb, kb_id)
        except KBError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        filename = file.filename or "document"
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"不支持的文件类型: {suffix or '(无后缀)'}")

        content = await _read_capped(file)

        def _register() -> tuple[dict, Path]:
            """登记文档并把原始件按 doc_id 落盘（工单 04）。"""
            record = qa_pipeline.registry.add_document(
                kb_id, filename, chunk_count=0, status="queued"
            )
            target = UPLOADS_DIR / record["doc_id"]
            target.mkdir(parents=True, exist_ok=True)
            path = target / safe_name(filename)
            path.write_bytes(content)
            return record, path

        record, stored_path = await run_meta(_register)
        doc_id = record["doc_id"]

        if queue_enabled():
            payload = ingest_job_payload(kb_id, doc_id, str(stored_path), filename)
            try:
                job_id = await enqueue_ingest(payload)
            except Exception as exc:
                # 入队失败必须就地收尾：留下一条 queued 记录等于造一个孤儿
                # —— 没有任何人会被叫醒处理它，客户端会轮询到永远。
                await run_meta(
                    qa_pipeline.registry.update_document, kb_id, doc_id,
                    status="failed", chunk_count=0, error=f"入队失败: {exc}"[:500],
                )
                raise HTTPException(status_code=503, detail=f"摄取队列不可用: {exc}")
            response.status_code = 202
            return {**record, "status": "queued", "queue_job": job_id}

        def _sync_ingest() -> dict:
            try:
                return ingest_document(
                    qa_pipeline.store, qa_pipeline.registry, kb_id, doc_id, stored_path
                )
            except Exception as exc:
                if qa_pipeline.registry.get_document(kb_id, doc_id)["status"] != "failed":
                    qa_pipeline.registry.update_document(
                        kb_id, doc_id, status="failed", chunk_count=0, error=str(exc)[:500]
                    )
                raise HTTPException(status_code=500, detail=f"文档索引失败: {exc}")

        try:
            result = await run_ingest(_sync_ingest)
        except HTTPException:
            raise
        qa_pipeline.retriever.invalidate(kb_id)  # 同进程快速通道，不等轮询窗口
        return result


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
