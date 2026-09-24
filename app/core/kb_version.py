"""知识库变更版本号：跨进程的检索缓存失效信号（工单 13）。

问题
----
检索层有两处**进程内**缓存：
- `HybridRetriever._bm25_cache`：每个知识库一份关键词索引；
- （未来）任何按库预热的派生结构。

单进程时靠同进程直接调用 `retriever.invalidate(kb_id)` 就能保持一致 —— 这其实是
个侥幸：写入方和读取方在同一个地址空间里。一旦摄取移到 worker 进程（工单 12），
worker 改了数据库，API 进程里的索引**不会失效**，于是线上会出现"文档已上传、
检索却查不到"，而且随进程重启又神秘恢复。

做法
----
每个 kb 一个版本号，存在共享存储（Redis）里；写入方变更后 +1，读取方在取用
进程内缓存前比对版本，不一致就重建。

为什么带一个本地 TTL（`_POLL_TTL`）
-----------------------------------
每次检索都去 Redis 读一次版本号，等于给每个请求的检索路径加一次网络往返。
版本检查本身只在"缓存可能过期"时才需要，所以用一个很短的轮询窗口（默认 1 秒）：

- 1 秒内的连续请求复用已读到的版本 —— 绝大多数请求零额外开销；
- 代价是变更后最多 1 秒的可见性延迟。对文档摄取这种"分钟级才有效果"的路径
  完全可接受，换来的是在线路径不被拖慢。

Redis 不可用时退回"只信进程内 invalidate()"的旧行为：功能不因此中断，
单机部署本来也没有跨进程问题。
"""
import time
from typing import Dict, Optional

from app.core.cache import get_cache

_POLL_TTL = 1.0  # 秒：向共享存储确认版本的最小间隔
_VERSION_KEY = "kbver:{kb_id}"


class KBVersionTracker:
    """读取方：判断进程内缓存是否还跟得上数据库。"""

    def __init__(self, poll_ttl: float = _POLL_TTL) -> None:
        self._poll_ttl = poll_ttl
        self._seen: Dict[str, tuple[float, int]] = {}  # kb_id -> (读取时刻, 版本号)

    def current(self, kb_id: str) -> int:
        """取该库当前已知的版本号；拿不到共享存储时返回进程内已知值（可能偏旧）。"""
        now = time.monotonic()
        cached = self._seen.get(kb_id)
        if cached and now - cached[0] < self._poll_ttl:
            return cached[1]
        version = self._read(kb_id)
        if version is None:  # Redis 不可用：沿用上次已知值，避免每次重试
            return cached[1] if cached else 0
        self._seen[kb_id] = (now, version)
        return version

    def _read(self, kb_id: str) -> Optional[int]:
        cache = get_cache()
        if not cache.available:
            return None
        try:
            raw = cache.raw_client.get(_VERSION_KEY.format(kb_id=kb_id))
        except Exception:  # noqa: BLE001 - 共享存储抖动不该让检索失败
            return None
        return int(raw) if raw else 0

    def is_stale(self, kb_id: str, known_version: Optional[int]) -> bool:
        """进程内缓存记录的版本落后于共享版本 → 需要重建。"""
        if known_version is None:
            return True
        return self.current(kb_id) != known_version

    def forget(self, kb_id: Optional[str] = None) -> None:
        """同进程写入后的快速通道：让下一次检查立刻反映新版本。"""
        if kb_id is None:
            self._seen.clear()
        else:
            self._seen.pop(kb_id, None)


def bump_kb_version(kb_id: str) -> None:
    """写入方：文档增删后调用，让所有进程的检索缓存知道该重建了。"""
    cache = get_cache()
    if not cache.available:
        return
    try:
        cache.raw_client.incr(_VERSION_KEY.format(kb_id=kb_id))
    except Exception:  # noqa: BLE001 - 信号失败不该让写入本身回滚
        pass
