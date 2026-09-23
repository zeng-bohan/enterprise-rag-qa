"""并发去重（single-flight）：同一个键的并发请求只真正执行一次。

要解决的问题
------------
语义缓存的读→算→写三步之间没有任何互斥：8 个客户端同时问同一个问题，
`_cache_lookup` 8 次全部 miss，于是 8 次全部打生成 LLM、8 次全部写缓存。
这既是 `bench.py` 那种并发场景下 P95 的主因，也是白烧的 token。

和 Redis 锁的取舍
----------------
这里用的是**进程内**去重，不是分布式锁：

- 多进程部署（uvicorn --workers N）下，每个 worker 各去重一次，最坏是 N 次 LLM
  调用，而不是无限制的放大 —— 相比现状已经是数量级的改善；
- Redis 锁要处理持有者崩溃、锁续租、等待方轮询间隔，复杂度和故障面都高得多，
  收益只覆盖"跨 worker 的重复请求"这一小块。真要跨进程收敛，正确的落点是
  让摄取与生成任务都走队列（工单 12），而不是给缓存加锁。

取消语义有一个已知边界：首个请求（leader）被取消时，等待方会一起收到取消异常，
而不是自动改由某个等待方接手重发。这与"客户端断开就不该继续烧 token"是一致的，
所以按原样保留并在调用点注释说明。
"""
import asyncio
from typing import Any, Awaitable, Callable, Dict


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: Dict[str, asyncio.Future] = {}

    def inflight_count(self) -> int:
        """观测用：当前正在被执行（并有人在等）的键数量。"""
        return len(self._inflight)

    async def run(self, key: str, factory: Callable[[], Awaitable[Any]]) -> Any:
        """执行 factory()，但同一 key 的并发调用共享一次执行结果。

        等待方用 `asyncio.shield` 包一层：否则任何一个等待方被取消，会把 leader
        的 future 一起取消掉，导致"谁取消谁毁掉所有人"。
        """
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await factory()
        except BaseException as exc:  # noqa: BLE001 - 必须把失败广播给等待方，不能让他们悬等
            self._inflight.pop(key, None)
            if not future.done():
                future.set_exception(exc)
            # 主动"取回"一次：没有等待方时，事件循环会在 future 被回收时打一条
            # "exception was never retrieved" 的告警 —— 异常是我们有意广播的，不是漏处理
            future.exception()
            raise
        self._inflight.pop(key, None)
        if not future.done():
            future.set_result(result)
        return result
