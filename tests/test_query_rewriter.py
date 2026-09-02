"""查询改写：缓存复用、空输出兜底与故障降级。"""
import asyncio

import pytest

from app.rag.query_rewriter import QueryRewriter
from tests.conftest import FakeCache, FakeLLM


@pytest.fixture()
def cache(monkeypatch):
    c = FakeCache()
    monkeypatch.setattr("app.rag.query_rewriter.get_cache", lambda: c)
    return c


def test_rewrite_uses_llm_and_caches_result(monkeypatch, cache):
    llm = FakeLLM("请年假 提前 申请")
    monkeypatch.setattr("app.rag.query_rewriter.get_llm", lambda temperature=0.0: llm)

    rw = QueryRewriter()
    assert rw.rewrite("年假要提前多久说啊") == "请年假 提前 申请"
    assert rw.rewrite("年假要提前多久说啊") == "请年假 提前 申请"
    assert llm.sync_calls == 1  # 第二次命中缓存，不再调用 LLM


def test_rewrite_empty_llm_output_falls_back_to_question(monkeypatch, cache):
    llm = FakeLLM("  ")
    monkeypatch.setattr("app.rag.query_rewriter.get_llm", lambda temperature=0.0: llm)
    rw = QueryRewriter()
    assert rw.rewrite("年假政策") == "年假政策"


def test_rewrite_llm_failure_degrades_to_question_without_caching(monkeypatch, cache):
    class BoomLLM(FakeLLM):
        def invoke(self, messages):
            self.sync_calls += 1
            raise RuntimeError("限流")

    llm = BoomLLM()
    monkeypatch.setattr("app.rag.query_rewriter.get_llm", lambda temperature=0.0: llm)
    rw = QueryRewriter()
    assert rw.rewrite("年假政策") == "年假政策"
    assert cache._store == {}  # 失败不写缓存，下轮可重试
    assert rw.rewrite("年假政策") == "年假政策"
    assert llm.sync_calls == 2


def test_rewrite_different_questions_get_different_cache_keys(monkeypatch, cache):
    llm = FakeLLM("改写")
    monkeypatch.setattr("app.rag.query_rewriter.get_llm", lambda temperature=0.0: llm)
    rw = QueryRewriter()
    rw.rewrite("问题甲")
    rw.rewrite("问题乙")
    assert len(cache._store) == 2
    assert llm.sync_calls == 2


def test_arewrite_hits_cache_and_skips_llm(monkeypatch, cache):
    llm = FakeLLM("请年假 提前 申请")
    monkeypatch.setattr("app.rag.query_rewriter.get_llm", lambda temperature=0.0: llm)
    rw = QueryRewriter()
    assert asyncio.run(rw.arewrite("年假要提前多久说啊")) == "请年假 提前 申请"
    assert asyncio.run(rw.arewrite("年假要提前多久说啊")) == "请年假 提前 申请"
    assert llm.async_calls == 1
