"""生成层：引用溯源、无答案拒答与语义缓存键设计。"""
import asyncio

import pytest
from langchain_core.documents import Document

from app.rag.generator import REFUSAL_ANSWER, agenerate, build_context, generate
from tests.conftest import FakeCache, FakeLLM, mkdoc


def test_build_context_numbers_sources_and_truncates_snippet():
    long_text = "长" * 200
    docs = [
        (mkdoc(long_text, source="员工手册.md"), 0.91234),
        (mkdoc("报销一个月内提交", source="财务.md"), 0.8),
    ]
    context, citations = build_context(docs)
    assert context.startswith("[1] 来源《员工手册.md》")
    assert "[2] 来源《财务.md》" in context
    assert len(citations) == 2
    assert citations[0]["index"] == 1
    assert citations[0]["score"] == 0.9123  # 四位小数
    assert len(citations[0]["snippet"]) == 80
    assert citations[1]["snippet"] == "报销一个月内提交"


def test_generate_refuses_when_no_docs():
    result = generate("年假几天", [], cache=FakeCache())
    assert result == {
        "answer": REFUSAL_ANSWER,
        "citations": [],
        "grounded": False,
        "cached": False,
    }


def test_generate_happy_path_and_semantic_cache(monkeypatch):
    llm = FakeLLM("年假 15 天 [1]")
    monkeypatch.setattr("app.rag.generator.get_llm", lambda: llm)
    cache = FakeCache()
    docs = [(mkdoc("员工年假 15 天", source="员工手册.md"), 0.9)]

    first = generate("年假几天", docs, cache=cache)
    assert first["answer"] == "年假 15 天 [1]"
    assert first["grounded"] is True
    assert first["cached"] is False
    assert first["citations"][0]["source"] == "员工手册.md"

    second = generate("年假几天", docs, cache=cache)
    assert second["cached"] is True
    assert second["answer"] == "年假 15 天 [1]"
    assert llm.sync_calls == 1  # 命中语义缓存，不再调 LLM
    assert second["citations"][0]["index"] == 1  # 引用实时组装，不随缓存复用


def test_generate_cache_key_changes_with_hit_doc_set(monkeypatch):
    llm = FakeLLM("答")
    monkeypatch.setattr("app.rag.generator.get_llm", lambda: llm)
    cache = FakeCache()
    d1 = (mkdoc("甲内容"), 0.9)
    d2 = (mkdoc("乙内容"), 0.8)

    generate("问题", [d1], cache=cache)
    generate("问题", [d2], cache=cache)  # 检索集变了 → 键变了 → 必须重新生成
    assert llm.sync_calls == 2


def test_generate_unrelated_question_refusal_keeps_grounded_false(monkeypatch):
    llm = FakeLLM("不应该被调用")
    monkeypatch.setattr("app.rag.generator.get_llm", lambda: llm)
    result = generate("无法回答的问题", [], cache=FakeCache())
    assert result["grounded"] is False
    assert llm.sync_calls == 0  # 拒答不烧 token


def test_agenerate_refusal_and_cache_roundtrip(monkeypatch):
    llm = FakeLLM("异步答案 [1]")
    monkeypatch.setattr("app.rag.generator.get_llm", lambda: llm)
    cache = FakeCache()
    docs = [(mkdoc("考勤规定", source="考勤.md"), 0.7)]

    refusal = asyncio.run(agenerate("问题", [], cache=cache))
    assert refusal["grounded"] is False

    first = asyncio.run(agenerate("考勤怎么规定", docs, cache=cache))
    assert first["answer"] == "异步答案 [1]"
    second = asyncio.run(agenerate("考勤怎么规定", docs, cache=cache))
    assert second["cached"] is True
    assert llm.async_calls == 1


def test_build_context_handles_missing_source_metadata():
    doc = Document(page_content="无来源内容", metadata={})
    _, citations = build_context([(doc, 0.5)])
    assert citations[0]["source"] == "未知"
