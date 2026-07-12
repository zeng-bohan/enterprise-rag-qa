"""Redis 缓存层（进程内单例）。

缓存对象（都缓存了什么、为什么）：
1. 查询改写结果：改写的输入只有问题本身，近似纯函数 → 按问题哈希缓存，
   命中后跳过改写 LLM 调用（省约 1s 延迟 + token 成本）；
2. 问答结果：按「问题 + 命中文档内容哈希集」做键（语义缓存）——检索结果变了
   键就变，旧缓存自动失效，避免答非所问的脏缓存；
3. Embedding 向量：按 chunk 内容哈希缓存，全量重建索引时免重复计算。

容错设计：Redis 不可用时降级为直查直算，缓存层永远不阻塞主链路。
"""
import hashlib
import json
from typing import Optional

import redis

from app.config import settings


class Cache:
    def __init__(self, url: str, ttl: int = 3600) -> None:
        self._ttl = ttl
        self._client: Optional[redis.Redis] = None
        try:
            client = redis.Redis.from_url(url, socket_timeout=2, decode_responses=True)
            client.ping()
            self._client = client
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    def get_json(self, key: str) -> Optional[dict]:
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set_json(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        if not self._client:
            return
        try:
            self._client.setex(key, ttl or self._ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass


# 模块级惰性单例：Embedding / 改写 / 生成 各模块共享同一个连接池
_cache: Optional[Cache] = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(settings.redis_url)
    return _cache
