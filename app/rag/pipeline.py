"""RAG 主流程编排：检索 → 生成，并统计各阶段耗时。

设计要点：
- 耗时拆分（retrieval_ms / latency_ms）是性能优化的观测基础——只有先分阶段测量，
  才能定位瓶颈在向量化、检索还是 LLM 生成；
- 依赖组装集中在这里：向量库 / 注册中心 / 缓存由工厂按 .env 选择，
  检索/生成各层不自己 new 资源；三个依赖均可注入（测试用桩替换）；
- 多轮（v0.6）：带 history 时先 condense 成独立问题再检索；
- 流式（v0.6）：astream 产出统一事件流，事件协议见 generator.astream_answer。
"""
import time
from typing import AsyncIterator, Dict, List, Optional

from app.core.cache import get_cache
from app.core.metrics import INFLIGHT, LATENCY
from app.rag.generator import (
    acondense_question,
    agenerate,
    astream_answer,
    condense_question,
    generate,
)
from app.rag.retriever import HybridRetriever
from app.rag.registry import KBRegistry, get_registry
from app.rag.vector_store import get_store


class RAGPipeline:
    def __init__(self, store=None, registry: Optional[KBRegistry] = None, cache=None) -> None:
        self.store = store if store is not None else get_store()
        self.registry = registry if registry is not None else get_registry()
        self.cache = cache if cache is not None else get_cache()
        self.retriever = HybridRetriever(self.store)

    # ---------- 非流式 ----------

    def ask(self, question: str, kb_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> dict:
        t0 = time.perf_counter()
        search_query = condense_question(question, history) if history else question
        hits = self.retriever.retrieve(search_query, kb_id=kb_id)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        result = generate(question, hits, cache=self.cache, history=history)
        result["retrieval_ms"] = round(retrieval_ms, 1)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    async def aask(
        self, question: str, kb_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """异步版 ask（v0.5）：LLM 等待全程挂起在事件循环上，不占线程。"""
        INFLIGHT.inc()
        t0 = time.perf_counter()
        try:
            search_query = await acondense_question(question, history) if history else question
            hits = await self.retriever.aretrieve(search_query, kb_id=kb_id)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            LATENCY.labels(stage="retrieval").observe(retrieval_ms / 1000)
            result = await agenerate(question, hits, cache=self.cache, history=history)
            result["retrieval_ms"] = round(retrieval_ms, 1)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return result
        finally:
            INFLIGHT.dec()
            LATENCY.labels(stage="total").observe(time.perf_counter() - t0)

    # ---------- 流式（v0.6）----------

    async def astream(
        self, question: str, kb_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict]:
        """流式问答：产出统一事件流（citations → token* → done）。

        事件顺序固定：先检索（产出 citations），再逐 token 生成，最后 done 带耗时。
        """
        INFLIGHT.inc()
        t0 = time.perf_counter()
        try:
            search_query = await acondense_question(question, history) if history else question
            hits = await self.retriever.aretrieve(search_query, kb_id=kb_id)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            LATENCY.labels(stage="retrieval").observe(retrieval_ms / 1000)

            async for event in astream_answer(question, hits, cache=self.cache, history=history):
                if event["event"] == "done":
                    data = dict(event["data"])
                    data["retrieval_ms"] = round(retrieval_ms, 1)
                    data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    yield {"event": "done", "data": data}
                else:
                    yield event
        finally:
            INFLIGHT.dec()
            LATENCY.labels(stage="total").observe(time.perf_counter() - t0)
