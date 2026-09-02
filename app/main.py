"""FastAPI 入口。

启动：.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
文档：http://127.0.0.1:8000/docs
指标：http://127.0.0.1:8000/metrics（Prometheus 抓取端点）

v0.6：启动时确保默认知识库存在（多知识库管理，见 app/api/manage.py）。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api import chat, manage
from app.core.executor import run_cpu
from app.rag.registry import get_registry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 默认知识库幂等创建：脚本入库 / 单库问答的落点
    await run_cpu(get_registry().ensure_default)
    yield


app = FastAPI(title="企业知识库 RAG 问答系统", version="0.6.0", lifespan=lifespan)
app.include_router(chat.router)
app.include_router(manage.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus 抓取端点：暴露请求 / 缓存 / LLM 指标（详见 app/core/metrics.py）。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
