"""HTTP 接口层：问答（JSON / SSE 流式）。

v0.5 起 endpoint 用 async def：LLM 调用走 ainvoke，等待期间不占线程；
CPU 密集的检索计算在 app.core.executor 的有界线程池里排队（详见 docs/DESIGN.md
的压测结论）。同步 pipeline.ask 保留给离线脚本（评测 / 压测数据生成）使用。

v0.6 新增：
- POST /v1/chat/stream：SSE 流式回答，事件协议 citations → token* → done；
- 请求体支持 kb_id（多知识库）与 history（多轮，先 condense 再检索）。
"""
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.auth import require_api_key
from app.core.metrics import REQUESTS
from app.api.deps import pipeline
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = await pipeline.aask(
            req.question, kb_id=req.kb_id, history=[m.model_dump() for m in req.history] if req.history else None
        )
    except Exception:
        REQUESTS.labels(status="error").inc()
        raise
    REQUESTS.labels(status="ok").inc()
    return result


def _sse(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式问答。

    事件序列：citations（检索完成，含引用）→ token（增量文本，可多条）→ done（耗时与状态）。
    拒答与缓存命中复用同一协议：citations → 单条 token → done，前端无需特判。
    """

    async def gen() -> AsyncIterator[str]:
        try:
            async for event in pipeline.astream(
                req.question,
                kb_id=req.kb_id,
                history=[m.model_dump() for m in req.history] if req.history else None,
            ):
                yield _sse(event)
            REQUESTS.labels(status="ok").inc()
        except Exception:
            REQUESTS.labels(status="error").inc()
            yield _sse({"event": "error", "data": {"message": "internal error"}})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
