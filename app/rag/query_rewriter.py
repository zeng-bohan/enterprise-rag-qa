"""查询改写：LLM 把口语化问题改写成关键词检索友好的查询。

动机：用户提问是口语（「年假要提前多久说？」），知识库文档是书面语
（「请年假须提前 3 个工作日申请」），字面重合度低导致 BM25 召回差。用 LLM 做一步
改写（保留专有名词与数字、补充同义词、去语气词），拿改写后的查询喂 BM25。

容错设计：改写失败（限流 / 网络抖动）时降级为原句，检索链路不因 LLM 故障中断。
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.cache import Cache, get_cache
from app.core.llm import get_llm
from app.core.metrics import CACHE_HITS, CACHE_MISSES, llm_done, llm_start

_SYSTEM_PROMPT = (
    "你是查询改写助手。把用户问题改写成一个用于关键词检索的短查询："
    "保留专有名词与数字，必要时补充同义词，去掉语气词。"
    "只输出改写结果本身，不要解释，不要加引号。"
)


class QueryRewriter:
    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.0)
        self._cache = get_cache()

    def rewrite(self, question: str) -> str:
        # 改写是纯函数：按问题哈希缓存，命中省一次 LLM 调用（约 1s + token）
        key = Cache.key("rewrite", question)
        hit = self._cache.get_json(key)
        if hit and hit.get("q"):
            return hit["q"]
        try:
            resp = self._llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]
            )
            out = (resp.content or "").strip() or question
            self._cache.set_json(key, {"q": out})
            return out
        except Exception:
            # 改写失败降级为原句，检索链路不因 LLM 故障中断（失败不写缓存，下轮可重试）
            return question

    async def arewrite(self, question: str) -> str:
        """异步版 rewrite（v0.5）：LLM 换 ainvoke，等待期间事件循环可处理其他请求。"""
        key = Cache.key("rewrite", question)
        hit = self._cache.get_json(key)
        if hit and hit.get("q"):
            CACHE_HITS.labels(kind="rewrite").inc()
            return hit["q"]
        CACHE_MISSES.labels(kind="rewrite").inc()
        try:
            t0 = llm_start()
            resp = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]
            )
            llm_done(t0)
            out = (resp.content or "").strip() or question
            self._cache.set_json(key, {"q": out})
            return out
        except Exception:
            return question
