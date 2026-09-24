"""arq worker：把文档摄取移出 HTTP 进程（工单 12）。

启动：
    .venv/Scripts/python -m arq app.worker.WorkerSettings

 worker 与 API 进程的唯一接口是**数据库 + kb 版本号**：
 - 它写 chunks 与 documents 状态；
 - `ingest_document` 内部会 `bump_kb_version`，API 进程据此重建关键词索引（工单 13）。
 没有这个信号，就会出现"文档传上去了但搜不到、重启又好了"这种最难查的故障。
"""
import json
import logging
import os
from pathlib import Path

from arq.connections import RedisSettings

from app.config import settings
from app.core.queue import INGEST_QUEUE, JOB_INGEST_DOCUMENT, queue_depth
from app.rag import ingest as ingest_mod

logger = logging.getLogger("rag.worker")

_store = None
_registry = None


def _deps():
    """惰性构造向量库与注册中心。

    刻意不在模块导入期做：`get_store()` 会加载 ONNX 模型（几百 MB 级），
    让 `arq app.worker.WorkerSettings` 这种"只读配置"的调用也付出模型加载成本，
    并且任何导入本模块的测试都会被拖进网络与磁盘。
    """
    global _store, _registry
    if _store is None:
        from app.rag.registry import get_registry
        from app.rag.vector_store import get_store

        _store = get_store()
        _registry = get_registry()
    return _store, _registry


async def ingest_document(ctx: dict, payload: str) -> dict:
    """处理一个摄取任务。失败时把状态落成 failed 后再抛出，交给 arq 重试。"""
    store, registry = _deps()
    job = json.loads(payload)
    kb_id, doc_id = job["kb_id"], job["doc_id"]
    path = Path(job["path"])

    registry.update_document(kb_id, doc_id, status="indexing")
    try:
        record = ingest_mod.ingest_document(store, registry, kb_id, doc_id, path)
    except Exception as exc:  # noqa: BLE001 - 必须落状态，否则文档永远停在 indexing
        registry.update_document(
            kb_id, doc_id, status="failed", chunk_count=0, error=str(exc)[:500]
        )
        logger.warning("摄取失败 doc_id=%s file=%s err=%s", doc_id, job.get("filename"), exc)
        raise
    logger.info("摄取完成 doc_id=%s chunks=%s", doc_id, record["chunk_count"])
    return {
        "doc_id": doc_id,
        "kb_id": kb_id,
        "status": record["status"],
        "chunk_count": record["chunk_count"],
    }


async def startup(ctx: dict) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _deps()  # 预热：模型加载放在启动期，而不是压在第一个任务上
    logger.info("rag worker ready, queue=%s", INGEST_QUEUE)


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = INGEST_QUEUE
    functions = [ingest_document]
    on_startup = startup
    # 并发摄取数：受限于 ONNX 线程配额，给得大只会让在线检索抢不到核
    max_jobs = int(os.environ.get("RAG_WORKER_CONCURRENCY", "2"))
    job_timeout = int(os.environ.get("RAG_WORKER_JOB_TIMEOUT", "900"))
    max_tries = 2
    retry_jobs = True

