"""API 出入参模型。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    kb_id: Optional[str] = Field(None, description="知识库 id；缺省用默认知识库")
    history: Optional[List[Message]] = Field(
        None, max_length=20, description="多轮对话历史（可选；提供时先做 condense 检索改写）"
    )


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


class CreateKBRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)


class RetrievalTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    kb_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
