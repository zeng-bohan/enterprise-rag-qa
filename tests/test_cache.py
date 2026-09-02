"""缓存层：降级语义（Redis 不可用不阻塞主链路）与键确定性。"""
from unittest import mock

from app.core.cache import Cache


def test_unreachable_redis_degrades_gracefully():
    cache = Cache("redis://127.0.0.1:1/0")  # 不可达端口，ping 失败
    assert cache.available is False
    assert cache.get_json("any") is None
    cache.set_json("any", {"a": 1})  # 不抛异常即符合降级语义


def test_key_is_deterministic_and_order_sensitive():
    assert Cache.key("answer", "问题", "abc") == Cache.key("answer", "问题", "abc")
    assert Cache.key("answer", "问题") != Cache.key("问题", "answer")


def test_key_hashes_parts_into_single_token():
    k = Cache.key("rewrite", "年假几天")
    assert isinstance(k, str) and len(k) == 32  # md5 hexdigest


def test_available_client_roundtrip(monkeypatch):
    store = {}
    recorded = {}

    class FakeRedis:
        def ping(self):
            return True

        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            recorded["ttl"] = ttl
            store[key] = value

    monkeypatch.setattr(
        "redis.Redis.from_url",
        staticmethod(lambda url, **kwargs: FakeRedis()),
    )

    cache = Cache("redis://127.0.0.1:6379/0")
    assert cache.available is True
    cache.set_json("k1", {"a": "中文"}, ttl=60)
    assert recorded["ttl"] == 60
    assert cache.get_json("k1") == {"a": "中文"}
    assert cache.get_json("missing") is None


def test_corrupt_value_in_redis_returns_none(monkeypatch):
    class FakeRedis:
        def ping(self):
            return True

        def get(self, key):
            return "不是json"

    monkeypatch.setattr(
        "redis.Redis.from_url", staticmethod(lambda url, **kwargs: FakeRedis())
    )
    cache = Cache("redis://127.0.0.1:6379/0")
    assert cache.get_json("k") is None
