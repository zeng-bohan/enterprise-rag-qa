"""LLM 客户端：DeepSeek（OpenAI 兼容协议）单例工厂。"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """temperature=0 保证问答输出稳定可复现（评测集需要确定性）。

    参数含义：
    - base_url 指向 DeepSeek 的 OpenAI 兼容端点；
    - timeout/max_retries 是生产可用性的兜底，网络抖动时自动重试。
    """
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
