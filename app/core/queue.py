"""文档摄取的异步队列（arq + Redis，工单 12）。

为什么要有这条路径
------------------
修复前上传接口在请求线程里同步做"解析 → 切片 → 批量向量化 → 写库"。小文档几秒
返回没问题，但一个几百页的 PDF 会把这条请求按住几十秒到几分钟，而进程内任何
线程配额都挡不住"一个任务占满池"这种模式（工单 14 只能限制爆炸半径）。
所以把摄取移出 HTTP 请求，交给独立 worker 进程。

为什么默认还是 sync
------------------
`INGEST_MODE=queue` 会把上传的响应语义从 201 改成 202、并且 chunk_count 不再
立即可得 —— 这是**破坏性变更**。所以：

- 默认 `sync`：现有部署、本地开发、离线测试行为不变；
- `docker-compose.yml` 里显式设成 `queue`，并附带 worker 服务 —— 生产路径拿到
  异步能力，而升级与否由部署方决定。

没有引入新的基础设施：Redis 已经在栈里（缓存用的就是它）。
"""
import json
from typing import Optional

from app.config import settings

INGEST_QUEUE = "rag:ingest"
JOB_INGEST_DOCUMENT = "ingest_document"


def queue_enabled() -> bool:
    return settings.ingest_mode == "queue"


def ingest_job_payload(kb_id: str, doc_id: str, path: str, filename: str) -> dict:
    """任务载荷。带 path 而不是文件内容：原始件已按 doc_id 落盘（工单 04），
    把 PDF 塞进队列消息只会让 Redis 存一堆二进制、且 worker 重启后无法追溯。"""
    return json.dumps({"kb_id": kb_id, "doc_id": doc_id, "path": path, "filename": filename})


_pool = None


async def get_pool():
    """进程级 arq 连接池（惰性创建）。

    每请求新建 Redis 连接会让上传路径平白多一次握手，所以缓存一个；
    失败时不缓存，下一次调用可以重试（否则会把第一次抖动固化成永久不可用）。
    """
    global _pool
    if _pool is None:
        from arq import create_pool
        from arq.connections import RedisSettings

        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool


async def enqueue_ingest(payload: str) -> str:
    """把摄取任务入队并返回 job id。

    端点必须等这一步成功再返回 202。否则会出现最糟的一种半态：documents 表里
    留下一条 queued 记录，但没有任何人会被叫醒去处理它 —— 调用方轮询到永远。
    """
    pool = await get_pool()
    job = await pool.enqueue_job(JOB_INGEST_DOCUMENT, payload, _queue_name=INGEST_QUEUE)
    if job is None:  # pragma: no cover - arq 仅在去重命中时返回 None
        raise RuntimeError("摄取任务入队失败")
    return job.job_id


def queue_depth() -> int | None:
    """待处理摄取任务数（观测用）。

    两个坑都是真跑踩到的：
    1. arq 把队列存在一个 **zset** 里（不是 list），所以 LLEN 会抛
       WRONGTYPE —— 第一版就是这么写的，只有连真 Redis 才会暴露；
    2. 队列键的内部布局属于 arq 的实现细节，版本可能变。
       因此这里对键类型做检查：不是 zset 就返回 None，而不是让指标采集把
       一次上传或一个 /metrics 抓取变成 500。
    """
    import redis as redis_lib

    try:
        client = redis_lib.Redis.from_url(settings.redis_url)
    except Exception:  # noqa: BLE001
        return None
    try:
        kind = client.type(INGEST_QUEUE)
        kind = kind.decode() if isinstance(kind, bytes) else kind
        if kind in ("none", "zset"):
            return int(client.zcard(INGEST_QUEUE))
        return None
    except Exception:  # noqa: BLE001 - 观测指标不该影响主流程
        return None
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
