"""RAG 主流程编排：检索 → 生成，并统计各阶段耗时。

设计要点：
- 耗时拆分（retrieval_ms / latency_ms）是性能优化的观测基础——只有先分阶段测量，
  才能定位瓶颈在向量化、检索还是 LLM 生成；
- 依赖组装集中在这里：向量库后端由 get_store() 工厂按 .env 选择，
  缓存用进程内单例 get_cache()，检索/生成各层不自己 new 资源。
"""
import time

from app.core.cache import get_cache
from app.core.metrics import INFLIGHT, LATENCY
from app.rag.generator import agenerate, generate
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import get_store


class RAGPipeline:
    def __init__(self) -> None:
        self.store = get_store()
        self.cache = get_cache()
        self.retriever = HybridRetriever(self.store)

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        result = generate(question, hits, cache=self.cache)
        result["retrieval_ms"] = round(retrieval_ms, 1)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    async def aask(self, question: str) -> dict:
        """异步版 ask（v0.5）：LLM 等待全程挂起在事件循环上，不占线程。"""
        INFLIGHT.inc()
        t0 = time.perf_counter()
        try:
            hits = await self.retriever.aretrieve(question)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            LATENCY.labels(stage="retrieval").observe(retrieval_ms / 1000)
            result = await agenerate(question, hits, cache=self.cache)
            result["retrieval_ms"] = round(retrieval_ms, 1)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return result
        finally:
            INFLIGHT.dec()
            LATENCY.labels(stage="total").observe(time.perf_counter() - t0)
