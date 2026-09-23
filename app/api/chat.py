"""HTTP 接口层：问答（JSON / SSE 流式）。

v0.5 起 endpoint 用 async def：LLM 调用走 ainvoke，等待期间不占线程；
CPU 密集的检索计算在 app.core.executor 的有界线程池里排队（详见 docs/DESIGN.md
的压测结论）。同步 pipeline.ask 保留给离线脚本（评测 / 压测数据生成）使用。

v0.6 新增：
- POST /v1/chat/stream：SSE 流式回答，事件协议 citations → token* → done；
- 请求体支持 kb_id（多知识库）与 history（多轮，先 condense 再检索）。

工单 19 对流式路径做的三件事：
1. 关掉代理缓冲（否则 Nginx 会把整个响应攒完才发，"流式"在浏览器里表现为一次性返回）；
2. 空闲心跳（LLM 首 token 可能等十几秒，代理的 60s 空闲超时会直接掐断连接）；
3. error 之后补一个 done 终止帧（否则客户端状态机会永远停在"还在生成"）。
"""
import asyncio
import json
from contextlib import suppress
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.auth import require_api_key, require_rate_limit
from app.core.metrics import REQUESTS, counted
from app.api.deps import pipeline
from app.api.errors import backend_errors
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
    dependencies=[Depends(require_api_key), Depends(require_rate_limit)],
)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    with counted():
        result = await pipeline.aask(
            req.question, kb_id=req.kb_id, history=[m.model_dump() for m in req.history] if req.history else None
        )
    return result


def _sse(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


_END = object()

# SSE 注释帧：以 ':' 开头的行，客户端与 EventSource 规范都要求忽略其内容。
# 用它做心跳是因为它不会污染事件流（不像发一个假 event 会让前端要多判一种类型）。
_PING = ": ping\n\n"


async def _with_heartbeat(events: AsyncIterator[dict], interval: float) -> AsyncIterator:
    """把事件流包一层，空闲超过 interval 秒就产出一个 None（调用方翻译心跳帧）。

    为什么要单独起一个 task：`async for` 上没法同时"等下一个事件"和"等超时"，
    所以把上游泵进队列，下游用 wait_for(队列.get()) 取。
    产出物要么是 dict（真事件），要么是 None（该发心跳了）。
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def _pump() -> None:
        try:
            async for event in events:
                await queue.put(event)
        finally:
            await queue.put(_END)

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval) if interval > 0 else await queue.get()
            except asyncio.TimeoutError:
                yield None
                continue
            if item is _END:
                return
            yield item
    finally:
        # 客户端断开 / 上游异常时，把泵任务收掉，别让它继续读 LLM 流烧 token
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答。

    事件序列：citations（检索完成，含引用）→ token（增量文本，可多条）→ done（耗时与状态）。
    拒答与缓存命中复用同一协议：citations → 单条 token → done，前端无需特判。
    出错时协议是 citations? → error → done：**done 一定会有**，客户端可以只依赖
    "看到 done 就收尾"这一条规则。
    """

    async def gen() -> AsyncIterator[str]:
        failed = False
        try:
            with backend_errors():
                async for item in _with_heartbeat(
                    pipeline.astream(
                        req.question,
                        kb_id=req.kb_id,
                        history=[m.model_dump() for m in req.history] if req.history else None,
                    ),
                    settings.sse_heartbeat_seconds,
                ):
                    if item is None:
                        yield _PING
                        continue
                    yield _sse(item)
        except HTTPException as exc:
            # 流已经开始，状态码早发出去了 —— 只能把故障降级成 error 事件，
            # 但随后必须补 done，否则客户端状态机悬在半路
            failed = True
            yield _sse({"event": "error", "data": {"message": exc.detail}})
            yield _sse({"event": "done", "data": {"grounded": False, "cached": False, "error": True}})
        except Exception:
            failed = True
            yield _sse({"event": "error", "data": {"message": "internal error"}})
            yield _sse({"event": "done", "data": {"grounded": False, "cached": False, "error": True}})
        finally:
            REQUESTS.labels(status="error" if failed else "ok").inc()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Nginx 的 proxy_buffering 会把整个响应攒完才转发，流式直接失效；
            # 这个头让 Nginx 对这条响应关掉缓冲，不依赖运维改站点配置
            "X-Accel-Buffering": "no",
            # 明确告诉中间代理不要压缩/改写这块流（gzip 会把小帧攒成块）
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )
