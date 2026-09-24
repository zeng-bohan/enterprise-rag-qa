"""进程级共享的 PostgreSQL 连接池（工单 10）。

为什么必须共享
--------------
修复前 PGvectorStore 与 PGKBRegistry 各建各的池（8 + 4 = 最多 12 条连接），
带来两个问题：

1. **连接数随模块叠加**，多 worker 部署下是 worker 数 × 12，很容易顶到 PG 的
   max_connections（默认 100，也就是 8 个 API worker 就快满了）；
2. 更要命的是**没法做跨表事务**。文档摄取要同时写 `chunks`（向量库）和
   `documents`（注册中心）。两条连接就是两个事务，于是"chunk 写成功、状态没更新"
   或反过来的半态是结构性的，只能靠补偿逻辑兜（工单 18 要解决的正是这个）。
   同一个池 → 同一条连接 → 一个真事务。

按 DSN 缓存，所以一个进程里只会有一个池。
"""
from typing import Dict

_pools: Dict[str, object] = {}


def configure_vector_session(conn) -> None:
    """每条新连接都会带上 HNSW 的查询期候选队列长度。

    放在建池的 configure 里、而不是每条 query 前 SET 一次：省一次往返，
    而且 autocommit 下 SET LOCAL 根本不生效。
    注册中心不需要它，但两者共用同一个池，所以由池统一设置——
    这样"谁先创建池"就不会改变检索行为（这是共享池方案唯一的隐性风险点）。
    """
    from app.config import settings

    if settings.ann_index == "hnsw":
        conn.execute(f"SET hnsw.ef_search = {int(settings.hnsw_ef_search)}")


def get_pool(dsn: str, *, max_size: int = 8, application_name: str = "rag-app"):
    """返回（并惰性创建）该 DSN 的连接池。

    min_size=1 而不是 0：第一条连接的建立含 TCP + 认证 + register_vector，
    放在冷启动路径上会让第一个真实请求替所有人付这笔钱。
    """
    from psycopg_pool import ConnectionPool

    pool = _pools.get(dsn)
    if pool is None:
        pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_size,
            kwargs={"autocommit": True, "application_name": application_name},
            configure=configure_vector_session,
            timeout=10,  # PG 不健康时快速失败，而不是每个请求都卡 30s
            check=ConnectionPool.check_connection,  # 服务端回收过的连接取回时先探一次
            open=True,
        )
        _pools[dsn] = pool
    return pool


def reset_pools() -> None:
    """测试用：丢掉缓存的池，让下一个用例可以换 DSN。"""
    for pool in _pools.values():
        try:
            pool.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应掩盖测试本身的结论
            pass
    _pools.clear()
