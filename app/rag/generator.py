"""生成层：引用溯源 + 无答案拒答 + 幻觉抑制 + 语义缓存 + 多轮 + 流式。

设计要点与实现位置：
- 引用溯源：参考资料按 [n] 编号注入 prompt，回答强制标注编号；
- 无答案拒答：检索层阈值过滤后为空，直接返回固定话术，不调用 LLM（省 token）；
- 幻觉抑制：System Prompt 约束「只用参考资料作答、不得编造」，
  并配合 RAGAs（Faithfulness）持续量化验证；
- 语义缓存：键 = 问题 + 命中文档内容哈希集。检索集变了键就变，旧缓存自动失效，
  避免「知识库已更新但返回旧答案」的脏缓存（详见 docs/DESIGN.md）；
- 多轮（v0.6）：带 history 时先做 condense（把追问改写成独立问题）再检索，
  生成 prompt 注入历史轮次；多轮请求不走语义缓存（同问题不同上文语义不同，
  缓存键不含历史，宁可不缓存也不返回脏答案）；
- 流式（v0.6）：astream_answer 以「事件流」产出（citations → token* → done），
  API 层原样转成 SSE；拒答与缓存命中也走同一事件协议，前端零特判。
"""
from typing import AsyncIterator, Dict, List, Optional, Tuple

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

CONDENSE_PROMPT = """请根据对话历史，把用户的最新问题改写成一个不依赖上下文、
可以独立检索知识库的完整问题。保留专有名词与数字，只输出改写后的问题本身。"""

# 生成 prompt 里最多带的历史轮数（控制 token 预算）
MAX_HISTORY_TURNS = 6


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


def _history_block(history: Optional[List[Dict[str, str]]]) -> str:
    turns = history[-MAX_HISTORY_TURNS:] if history else []
    lines = [f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')[:1000]}" for m in turns]
    return "\n".join(lines)


def condense_question(question: str, history: List[Dict[str, str]]) -> str:
    """把「追问」改写成独立问题（多轮检索的前置步骤）。失败降级为原问题。"""
    if not history:
        return question
    try:
        llm = get_llm(temperature=0.0)
        turns = _history_block(history)
        resp = llm.invoke(
            [
                SystemMessage(content=CONDENSE_PROMPT),
                HumanMessage(content=f"【对话历史】\n{turns}\n\n【最新问题】\n{question}"),
            ]
        )
        return (resp.content or "").strip() or question
    except Exception:
        return question


async def acondense_question(question: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return question
    try:
        llm = get_llm(temperature=0.0)
        turns = _history_block(history)
        resp = await llm.ainvoke(
            [
                SystemMessage(content=CONDENSE_PROMPT),
                HumanMessage(content=f"【对话历史】\n{turns}\n\n【最新问题】\n{question}"),
            ]
        )
        return (resp.content or "").strip() or question
    except Exception:
        return question


def _user_prompt(question: str, context: str, history: Optional[List[Dict[str, str]]]) -> str:
    user = f"【用户问题】\n{question}\n\n【参考资料】\n{context}"
    if history:
        user = f"【对话历史】\n{_history_block(history)}\n\n{user}"
    return user


def _cache_lookup(cache: Cache, question: str, docs_with_scores, history) -> Optional[dict]:
    # 多轮请求不走语义缓存：缓存键不含历史，命中会返回与上下文无关的脏答案
    if history:
        return None
    doc_sig = ",".join(sorted(_doc_id(d) for d, _ in docs_with_scores))
    return cache.get_json(Cache.key("answer", question, doc_sig))


def _cache_write(cache: Cache, question: str, docs_with_scores, answer: str) -> None:
    doc_sig = ",".join(sorted(_doc_id(d) for d, _ in docs_with_scores))
    cache.set_json(Cache.key("answer", question, doc_sig), {"a": answer})


def generate(
    question: str,
    docs_with_scores: List[Tuple[Document, float]],
    cache: Optional[Cache] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    if not docs_with_scores:
        return {"answer": REFUSAL_ANSWER, "citations": [], "grounded": False, "cached": False}

    context, citations = build_context(docs_with_scores)

    cache = cache or get_cache()
    hit = _cache_lookup(cache, question, docs_with_scores, history)
    if hit and hit.get("a"):
        return {"answer": hit["a"], "citations": citations, "grounded": True, "cached": True}

    llm = get_llm()
    resp = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_user_prompt(question, context, history))]
    )
    answer = resp.content or ""
    if not history:
        _cache_write(cache, question, docs_with_scores, answer)
    return {"answer": answer, "citations": citations, "grounded": True, "cached": False}


async def agenerate(
    question: str,
    docs_with_scores: List[Tuple[Document, float]],
    cache: Optional[Cache] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """异步版 generate（v0.5）：生成 LLM 用 ainvoke，等待不占线程。

    缓存读写仍是同步 Redis（单次约 1ms），不值得为它做事件循环往返。
    """
    if not docs_with_scores:
        return {"answer": REFUSAL_ANSWER, "citations": [], "grounded": False, "cached": False}

    context, citations = build_context(docs_with_scores)

    cache = cache or get_cache()
    hit = _cache_lookup(cache, question, docs_with_scores, history)
    if hit and hit.get("a"):
        CACHE_HITS.labels(kind="answer").inc()
        return {"answer": hit["a"], "citations": citations, "grounded": True, "cached": True}
    CACHE_MISSES.labels(kind="answer").inc()

    llm = get_llm()
    t0 = llm_start()
    resp = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_user_prompt(question, context, history))]
    )
    llm_done(t0)
    answer = resp.content or ""
    if not history:
        _cache_write(cache, question, docs_with_scores, answer)
    return {"answer": answer, "citations": citations, "grounded": True, "cached": False}


async def astream_answer(
    question: str,
    docs_with_scores: List[Tuple[Document, float]],
    cache: Optional[Cache] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> AsyncIterator[Dict]:
    """流式生成：产出统一事件流（dict），API 层负责转 SSE。

    事件协议：
      {"event": "citations", "data": {"citations": [...]}}
      {"event": "token",     "data": {"t": "增量文本"}}
      {"event": "done",      "data": {"grounded": bool, "cached": bool}}
    """
    _, citations = build_context(docs_with_scores) if docs_with_scores else ("", [])
    yield {"event": "citations", "data": {"citations": citations}}

    if not docs_with_scores:
        yield {"event": "token", "data": {"t": REFUSAL_ANSWER}}
        yield {"event": "done", "data": {"grounded": False, "cached": False}}
        return

    cache = cache or get_cache()
    hit = _cache_lookup(cache, question, docs_with_scores, history)
    if hit and hit.get("a"):
        CACHE_HITS.labels(kind="answer").inc()
        yield {"event": "token", "data": {"t": hit["a"]}}
        yield {"event": "done", "data": {"grounded": True, "cached": True}}
        return
    CACHE_MISSES.labels(kind="answer").inc()

    context, _ = build_context(docs_with_scores)
    llm = get_llm()
    t0 = llm_start()
    parts: List[str] = []
    try:
        async for chunk in llm.astream(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_user_prompt(question, context, history))]
        ):
            token = chunk.content or ""
            if token:
                parts.append(token)
                yield {"event": "token", "data": {"t": token}}
    finally:
        llm_done(t0)

    answer = "".join(parts)
    if not history:
        _cache_write(cache, question, docs_with_scores, answer)
    yield {"event": "done", "data": {"grounded": True, "cached": False}}
