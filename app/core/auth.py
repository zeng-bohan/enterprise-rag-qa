"""API Key 鉴权 + 速率限制。

鉴权
----
- API_KEYS 留空 = 关闭鉴权（本地开发 / 评测脚本场景，行为与 v0.5 完全一致）；
- 配置后（逗号分隔多个 key），/v1 下所有业务接口必须携带合法 X-API-Key；
- /health 与 /metrics 保持开放：健康检查与 Prometheus 抓取不走鉴权（生产惯例）。

工单 20 的两处收紧：

1. **常量时间比较**。原来是 `x_api_key not in configured`，短路求值下命中位置越靠前
   返回越快。对本地维护的 key 列表这是理论问题，但修它的成本是两行，而"用比较密钥
   的写法去比较密钥"这件事本身就该是默认。
2. **速率限制**。RAG 服务的单次请求会放大成 1 次 embedding + 1 次 rerank + 最多
   2 次 LLM 调用，没有限流等于把一个便宜的 HTTP 入口换算成一张昂贵的账单，
   同时把 CPU 线程池拱手让人。

限流的取舍
----------
令牌桶放在 Redis 里（INCR + EXPIRE 的近似实现），因为多 worker 部署下进程内计数
each worker 各算一份，限额会被 worker 数倍乘。Redis 不可用时**放行而不是拒绝**：
限流是保护措施，不该成为新的单点故障——缓存层已经是这个策略，这里保持一致。
"""
import hmac
import time

from fastapi import Header, HTTPException, Request

from app.config import settings
from app.core.cache import get_cache
from app.core.metrics import RATE_LIMITED


def _configured_keys() -> list[str]:
    return [k.strip() for k in settings.api_keys.split(",") if k.strip()]


def _key_matches(provided: str, configured: list[str]) -> bool:
    """逐个常量时间比较，命中与否不通过提前返回泄露时间差。"""
    matched = False
    for key in configured:
        if hmac.compare_digest(provided, key):
            matched = True
    return matched


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = _configured_keys()
    if not configured:
        return
    if not x_api_key or not _key_matches(x_api_key, configured):
        raise HTTPException(status_code=401, detail="缺少或无效的 X-API-Key")


def _client_identity(request: Request) -> str:
    """限流桶的键：有鉴权用 API Key，否则退化为客户端 IP。"""
    configured = _configured_keys()
    if configured:
        key = request.headers.get("x-api-key")
        if key:
            return f"key:{key[:16]}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # 只取链上第一个（最接近真实客户端），且它可被伪造 —— 所以这是限流的
        # 粗粒度保护，不是身份识别
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _allow(bucket: str, limit_per_sec: float, burst: int) -> bool:
    """近似的秒窗令牌桶：Redis INCR + 首次 EXPIRE。

    不是严格平滑的令牌桶（窗口边界上最多放行 2×limit 的量），但实现简单、
    无 Lua 脚本、且在"挡住明显滥用"这个目标上足够。要精确限速应交给网关。
    """
    cache = get_cache()
    if not cache.available:
        return True  # Redis 挂了不能把全站锁死
    client = cache.raw_client
    window = int(time.time())
    key = f"ratelimit:{bucket}:{window}"
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, 2)
    except Exception:
        return True
    return count <= max(1, int(burst if burst else limit_per_sec))


def require_rate_limit(request: Request) -> None:
    """依赖：RATE_LIMIT_RPS=0 时直接放行（默认关闭，避免改变现有部署行为）。"""
    limit = settings.rate_limit_rps
    if limit <= 0:
        return
    if not _allow(_client_identity(request), limit, settings.rate_limit_burst):
        RATE_LIMITED.inc()
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试",
            headers={"Retry-After": "1"},
        )
