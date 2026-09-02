"""API Key 鉴权：X-API-Key 头校验。

- API_KEYS 留空 = 关闭鉴权（本地开发 / 评测脚本场景，行为与 v0.5 完全一致）；
- 配置后（逗号分隔多个 key），/v1 下所有业务接口必须携带合法 X-API-Key；
- /health 与 /metrics 保持开放：健康检查与 Prometheus 抓取不走鉴权（生产惯例）。
"""
from fastapi import Header, HTTPException

from app.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = [k.strip() for k in settings.api_keys.split(",") if k.strip()]
    if not configured:
        return
    if not x_api_key or x_api_key not in configured:
        raise HTTPException(status_code=401, detail="缺少或无效的 X-API-Key")
