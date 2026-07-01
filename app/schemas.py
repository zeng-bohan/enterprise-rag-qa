"""API 出入参模型。"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class Citation(BaseModel):
    index: int
    source: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    grounded: bool  # False 表示触发无答案拒答（检索结果全部低于相关度阈值）
    citations: list[Citation]
    cached: bool  # 命中 Redis 语义缓存则为 True（可观测性：压测归因用）
    retrieval_ms: float
    latency_ms: float
