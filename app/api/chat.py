"""HTTP 接口层。

v0.5 起 endpoint 用 async def：LLM 调用走 ainvoke，等待期间不占线程；
CPU 密集的检索计算在 app.core.executor 的有界线程池里排队（详见 docs/DESIGN.md
的压测结论）。同步 pipeline.ask 保留给离线脚本（评测 / 压测数据生成）使用。
"""
from fastapi import APIRouter

from app.core.metrics import REQUESTS
from app.rag.pipeline import RAGPipeline
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])

# 模块级单例：Embedding 模型与向量库只加载一次（加载模型是主要启动开销）
pipeline = RAGPipeline()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = await pipeline.aask(req.question)
    except Exception:
        REQUESTS.labels(status="error").inc()
        raise
    REQUESTS.labels(status="ok").inc()
    return result
