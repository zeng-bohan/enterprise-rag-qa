"""混合检索：RRF 融合排序、两级拒答与截断预算。"""
import asyncio
from unittest import mock

from langchain_core.documents import Document

from app.rag.retriever import RERANK_K, HybridRetriever, rrf_fuse
from tests.conftest import FakeReranker, FakeStore, mkdoc


def _make_retriever(store_docs, reranker):
    """绕过 __init__（不构造真实 QueryRewriter / CrossEncoder），只装后处理所需部件。"""
    r = object.__new__(HybridRetriever)
    r._reranker = reranker
    return r


# ---------------------------------------------------------------------------
# rrf_fuse：纯函数
# ---------------------------------------------------------------------------
def test_rrf_double_route_beats_single_route():
    a, b, c = mkdoc("A"), mkdoc("B"), mkdoc("C")
    fused = rrf_fuse([(a, 10.0), (b, 5.0)], [(a, 0.9), (c, 0.8)])
    assert fused[0][0] is a  # 两路都命中，融合分最高
    assert len(fused) == 3


def test_rrf_scores_follow_rank_not_raw_score():
    # 原始分大小不影响融合分，只有名次影响
    hi, lo = mkdoc("高 raw 分"), mkdoc("低 raw 分")
    fused = rrf_fuse([(hi, 99.0)], [])
    fused2 = rrf_fuse([(lo, 0.01)], [])
    assert math_close(fused[0][1], fused2[0][1])


def math_close(x, y):
    return abs(x - y) < 1e-12


def test_rrf_merges_same_content_doc_from_both_routes():
    # 两路返回同一 content 的文档（id 为空时按 content 去重），只输出一条
    d = mkdoc("年假须提前申请")
    fused = rrf_fuse([(d, 1.0)], [(Document(page_content=d.page_content, metadata={}), 0.9)])
    assert len(fused) == 1


def test_rrf_empty_inputs():
    assert rrf_fuse([], []) == []


# ---------------------------------------------------------------------------
# _finalize：拒答与截断
# ---------------------------------------------------------------------------
def test_prefuse_refusal_when_no_bm25_and_vec_below_floor():
    r = _make_retriever(None, FakeReranker([]))
    out = r._finalize("q", [], [(mkdoc("弱相关"), 0.10)], k=5, vec_floor=0.35)
    assert out == []


def test_prefuse_refusal_when_both_routes_empty():
    r = _make_retriever(None, FakeReranker([]))
    assert r._finalize("q", [], [], k=5, vec_floor=0.35) == []


def test_rerank_refusal_when_all_below_floor():
    r = _make_retriever(None, FakeReranker([0.2, 0.49]))
    out = r._finalize("q", [(mkdoc("甲"), 1.0)], [(mkdoc("乙"), 0.8)], k=5, vec_floor=0.35)
    assert out == []


def test_rerank_floor_filters_low_scores_and_sorts_desc():
    r = _make_retriever(None, FakeReranker([0.9, 0.2, 0.8]))
    out = r._finalize(
        "q",
        [(mkdoc("甲"), 1.0), (mkdoc("乙"), 0.5), (mkdoc("丙"), 0.4)],
        [],
        k=5,
        vec_floor=0.35,
    )
    assert [d.page_content for d, _ in out] == ["甲", "丙"]
    assert [s for _, s in out] == [0.9, 0.8]


def test_rerank_budget_capped_at_rerank_k():
    reranker = FakeReranker([1.0] * 20)
    r = _make_retriever(None, reranker)
    docs = [(mkdoc(f"文档{i}"), 1.0) for i in range(20)]
    r._finalize("q", docs, [], k=5, vec_floor=0.35)
    assert reranker.seen_candidates == RERANK_K


# ---------------------------------------------------------------------------
# retrieve / aretrieve：离线全链路（真实 BM25 + 桩改写/重排/向量库）
# ---------------------------------------------------------------------------
class StubRewriter:
    def rewrite(self, question: str) -> str:
        return question

    async def arewrite(self, question: str) -> str:
        return question


def _build_integration_retriever(monkeypatch, rerank_scores):
    d1 = mkdoc("员工请年假须提前三个工作日在 OA 申请", source="考勤.md")
    d2 = mkdoc("差旅报销须在费用发生后一个月内提交发票", source="财务.md")
    d3 = mkdoc("考勤：工作日上下班需打卡，迟到按制度处理", source="考勤.md")
    monkeypatch.setattr("app.rag.retriever.QueryRewriter", StubRewriter)
    monkeypatch.setattr(
        "app.rag.retriever.CrossEncoderReranker", lambda: FakeReranker(rerank_scores)
    )
    store = FakeStore([d1, d2, d3], hits=[(d1, 0.9), (d3, 0.6)])
    return HybridRetriever(store), d1, d3


def test_retrieve_full_pipeline_ranks_grounded_docs(monkeypatch):
    # BM25 只召回 D1，向量召回 [D1, D3] → 融合候选 [D1, D3]，
    # 重排分 [0.9, 0.6] 均过 0.5 地板 → 两条都保留
    retriever, d1, d3 = _build_integration_retriever(monkeypatch, [0.9, 0.6])
    hits = retriever.retrieve("年假 提前 申请")
    assert [d.page_content for d, _ in hits] == [d1.page_content, d3.page_content]
    assert hits[0][1] == 0.9


def test_aretrieve_matches_sync_behavior(monkeypatch):
    retriever, d1, d3 = _build_integration_retriever(monkeypatch, [0.9, 0.6])
    hits = asyncio.run(retriever.aretrieve("年假 提前 申请"))
    assert [d.page_content for d, _ in hits] == [d1.page_content, d3.page_content]


def test_retrieve_custom_threshold_triggers_prefuse_refusal(monkeypatch):
    retriever, *_ = _build_integration_retriever(monkeypatch, [0.9, 0.6])
    # 向量最高 0.9 < 0.95，BM25 对「量子」无命中 → 前置拒答
    assert retriever.retrieve("量子计算 芯片", threshold=0.95) == []


def test_degraded_reranker_keeps_results_instead_of_empty(monkeypatch):
    """重排器降级（模型加载失败）时应退化为不重排，而不是被 0.5 地板清空结果。"""
    class UnavailableReranker(FakeReranker):
        def __init__(self):
            super().__init__([])
            self.seen_candidates = 0

        def rerank(self, query, docs):
            self.seen_candidates = len(docs)
            return [(d, 0.0) for d in docs]

        @property
        def available(self):
            return False

    r = object.__new__(HybridRetriever)
    r._reranker = UnavailableReranker()
    docs = [mkdoc(f"文档{i}", source="s.md") for i in range(3)]
    out = r._finalize(
        "年假", [(docs[0], 1.0)], [(docs[1], 0.8), (docs[2], 0.7)], k=5, vec_floor=0.35
    )
    assert [d.page_content for d, _ in out] == ["文档0", "文档1", "文档2"]  # RRF 序保留
    assert r._reranker.available is False
