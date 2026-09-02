"""多轮对话：condense 改写进入检索、历史进入生成、多轮不走语义缓存。"""
import asyncio
from unittest import mock

from app.rag import pipeline as pipeline_mod
from app.rag.generator import condense_question
from tests.conftest import FakeCache, FakeLLM, RealRAGPipeline, mkdoc


def test_condense_question_uses_llm_output():
    llm = FakeLLM("年假天数 是多少")
    with mock.patch("app.rag.generator.get_llm", lambda temperature=0.0: llm):
        out = condense_question("那能休几天？", [{"role": "user", "content": "公司有年假吗"}, {"role": "assistant", "content": "有"}])
    assert out == "年假天数 是多少"


def test_condense_question_without_history_is_identity():
    assert condense_question("年假几天", []) == "年假几天"


def test_condense_question_llm_failure_degrades_to_original():
    class Boom(FakeLLM):
        def invoke(self, messages):
            raise RuntimeError("超时")

    with mock.patch("app.rag.generator.get_llm", lambda temperature=0.0: Boom()):
        assert condense_question("那能休几天？", [{"role": "user", "content": "有年假吗"}]) == "那能休几天？"


def test_aask_with_history_condenses_then_keeps_history_in_generation(monkeypatch):
    """aask：检索用改写后的问题，生成拿到原始问题 + 完整 history。"""
    pipe = RealRAGPipeline.__new__(RealRAGPipeline)

    captured = {}

    async def fake_acondense(question, history):
        captured["condense_q"] = question
        captured["condense_history"] = history
        return "改写后的独立问题"

    async def fake_aretrieve(query, kb_id=None, top_k=None, threshold=None):
        captured["retrieval_q"] = query
        return [(mkdoc("年假 15 天"), 0.9)]

    async def fake_agenerate(question, docs, cache=None, history=None):
        captured["gen_q"] = question
        captured["gen_history"] = history
        return {"answer": "答", "citations": [], "grounded": True, "cached": False}

    monkeypatch.setattr(pipeline_mod, "acondense_question", fake_acondense)
    monkeypatch.setattr(pipe, "retriever", mock.MagicMock(), raising=False)
    monkeypatch.setattr(pipe.retriever, "aretrieve", fake_aretrieve)
    monkeypatch.setattr(pipe, "cache", FakeCache(), raising=False)
    monkeypatch.setattr(pipeline_mod, "agenerate", fake_agenerate)

    history = [{"role": "user", "content": "公司有年假吗"}]
    result = asyncio.run(pipe.aask("那能休几天？", history=history))

    assert captured["condense_q"] == "那能休几天？"
    assert captured["retrieval_q"] == "改写后的独立问题"  # 检索用 condense 结果
    assert captured["gen_q"] == "那能休几天？"  # 生成保留原始问题
    assert captured["gen_history"] == history  # 生成带上历史
    assert result["answer"] == "答"


def test_multi_turn_skips_semantic_cache(monkeypatch):
    """带 history 的生成不读写语义缓存（缓存键不含历史，宁不缓存不返回脏答案）。"""
    from app.rag.generator import agenerate
    from tests.conftest import FakeCache

    cache = FakeCache()
    llm = FakeLLM("多轮答案 [1]")
    monkeypatch.setattr("app.rag.generator.get_llm", lambda: llm)
    docs = [(mkdoc("年假 15 天"), 0.9)]
    history = [{"role": "user", "content": "有年假吗"}]

    first = asyncio.run(agenerate("那几天？", docs, cache=cache, history=history))
    second = asyncio.run(agenerate("那几天？", docs, cache=cache, history=history))
    assert first["cached"] is False and second["cached"] is False
    assert llm.async_calls == 2  # 两次都真实生成
    assert cache._store == {}  # 没有写缓存


def test_astream_answer_event_protocol(monkeypatch):
    from app.rag.generator import astream_answer
    from tests.conftest import FakeCache

    class StreamingLLM:
        async def astream(self, messages):
            for t in ["年假", "15 天 [1]"]:
                yield mock.MagicMock(content=t)

    monkeypatch.setattr("app.rag.generator.get_llm", lambda: StreamingLLM())
    docs = [(mkdoc("年假 15 天"), 0.9)]

    async def collect():
        return [e async for e in astream_answer("年假几天？", docs, cache=FakeCache())]

    events = asyncio.run(collect())
    assert [e["event"] for e in events] == ["citations", "token", "token", "done"]
    assert events[0]["data"]["citations"][0]["source"] == "员工手册.md"
    assert events[-1]["data"]["grounded"] is True
