"""HTTP 层的异常翻译：领域异常 → 状态码。

单独放一个模块是因为 chat 和 manage 两组路由都要同一套语义，
放在 manage.py 里会让 chat 侧要么复制一份、要么反向依赖管理面。

规则只有一条，但它是本次修复新增的（工单 02）：
「业务上不该有结果」和「系统没能力回答」必须是两个状态码。
修复前所有后端异常都被揉进 KBError，一个 psycopg 客户端报的 SQL 语法错误
会以「知识库创建失败（重名？）」的面目出现，把排障方向整个带偏。
"""
from contextlib import contextmanager
from typing import Iterator

from fastapi import HTTPException

from app.rag.registry import BackendError


@contextmanager
def backend_errors() -> Iterator[None]:
    """存储后端故障 → 503；业务异常（KBError）留给各端点按自己的语义处理。"""
    try:
        yield
    except BackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
