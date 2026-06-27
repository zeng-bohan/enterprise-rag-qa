"""生成层：引用溯源 + 无答案拒答 + 幻觉抑制 + 语义缓存。

设计要点与实现位置：
- 引用溯源：参考资料按 [n] 编号注入 prompt，回答强制标注编号；
- 无答案拒答：检索层阈值过滤后为空，直接返回固定话术，不调用 LLM（省 token）；
- 幻觉抑制：System Prompt 约束「只用参考资料作答、不得编造」，
  并配合第 2 周 RAGAs（Faithfulness）持续量化验证；
- 语义缓存：键 = 问题 + 命中文档内容哈希集。检索集变了键就变，旧缓存自动失效，
  避免「知识库已更新但返回旧答案」的脏缓存（详见 docs/DESIGN.md）。
"""
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.cache import Cache, get_cache
from app.core.llm import get_llm
from app.core.metrics import CACHE_HITS, CACHE_MISSES, llm_done, llm_start
from app.rag.vector_store import _doc_id

SYSTEM_PROMPT = """你是企业知识库智能助手。请严格依据【参考资料】回答用户问题，规则如下：
1. 只使用参考资料中的内容作答，不要使用外部知识，也不要编造；
2. 使用简体中文，简洁、分点、直击要点；
3. 引用资料处标注编号，如 [1]、[2]，编号与参考资料序号一致；
4. 如果参考资料中没有能够回答问题的内容，只回复：根据公司现有资料，无法回答该问题。"""

REFUSAL_ANSWER = "根据公司现有资料，无法回答该问题。"


def build_context(docs_with_scores: List[Tuple[Document, float]]) -> Tuple[str, List[dict]]:
    parts, citations = [], []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "未知")
        parts.append(f"[{i}] 来源《{source}》：\n{doc.page_content}")
        citations.append(
            {
                "index": i,
                "source": source,
                "score": round(float(score), 4),
                "snippet": doc.page_content[:80],
            }
        )
    return "\n\n".join(parts), citations


def generate(
    question: str,
    docs_with_scores: List[Tuple[Document, float]],
    cache: Optional[Cache] = None,
) -> dict:
    if not docs_with_scores:
        return {"answer": REFUSAL_ANSWER, "citations": [], "grounded": False, "cached": False}

    context, citations = build_context(docs_with_scores)

    # 语义缓存：键含命中内容哈希集（排序保证确定性）。引用不缓存，实时从本次检索组装。
    cache = cache or get_cache()
    doc_sig = ",".join(sorted(_doc_id(d) for d, _ in docs_with_scores))
    key = Cache.key("answer", question, doc_sig)
    hit = cache.get_json(key)
    if hit and hit.get("a"):
        return {"answer": hit["a"], "citations": citations, "grounded": True, "cached": True}

    user = f"【用户问题】\n{question}\n\n【参考资料】\n{context}"
    llm = get_llm()
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    answer = resp.content or ""
    cache.set_json(key, {"a": answer})
    return {"answer": answer, "citations": citations, "grounded": True, "cached": False}


async def agenerate(
    question: str,
    docs_with_scores: List[Tuple[Document, float]],
    cache: Optional[Cache] = None,
) -> dict:
    """异步版 generate（v0.5）：生成 LLM 用 ainvoke，等待不占线程。

    缓存读写仍是同步 Redis（单次约 1ms），不值得为它做事件循环往返。
    """
    if not docs_with_scores:
        return {"answer": REFUSAL_ANSWER, "citations": [], "grounded": False, "cached": False}

    context, citations = build_context(docs_with_scores)

    cache = cache or get_cache()
    doc_sig = ",".join(sorted(_doc_id(d) for d, _ in docs_with_scores))
    key = Cache.key("answer", question, doc_sig)
    hit = cache.get_json(key)
    if hit and hit.get("a"):
        CACHE_HITS.labels(kind="answer").inc()
        return {"answer": hit["a"], "citations": citations, "grounded": True, "cached": True}
    CACHE_MISSES.labels(kind="answer").inc()

    user = f"【用户问题】\n{question}\n\n【参考资料】\n{context}"
    llm = get_llm()
    t0 = llm_start()
    resp = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    llm_done(t0)
    answer = resp.content or ""
    cache.set_json(key, {"a": answer})
    return {"answer": answer, "citations": citations, "grounded": True, "cached": False}
