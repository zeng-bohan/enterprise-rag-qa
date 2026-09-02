"""Cross-Encoder 重排：sigmoid 归一、降级路径与排序截断。"""
import math
from unittest import mock

from tests.conftest import mkdoc


def test_sigmoid_known_values():
    from app.rag.reranker import sigmoid

    assert sigmoid(0) == 0.5
    assert math.isclose(sigmoid(100), 1.0, rel_tol=1e-6)
    assert math.isclose(sigmoid(-100), 0.0, abs_tol=1e-6)
    assert sigmoid(1) > sigmoid(0) > sigmoid(-1)  # 单调


def _make_reranker(model):
    """绕过 __init__（不触发模型加载），直接注入内部模型桩。"""
    from app.rag.reranker import CrossEncoderReranker

    r = object.__new__(CrossEncoderReranker)
    r._model = model
    return r


class FakeCrossEncoder:
    def __init__(self, logits):
        self.logits = logits
        self.pairs = None

    def rerank_pairs(self, pairs):
        self.pairs = list(pairs)
        return iter(self.logits)


def test_rerank_orders_by_score_desc():
    model = FakeCrossEncoder([-1.0, 3.0, 0.0])
    r = _make_reranker(model)
    docs = [mkdoc("甲"), mkdoc("乙"), mkdoc("丙")]
    scored = r.rerank("年假政策", docs)
    assert [d.page_content for d, _ in scored] == ["乙", "丙", "甲"]
    assert scored[0][1] > 0.9  # sigmoid(3)
    # 查询-文档对按原顺序拼装，长文档截断到 800 字符
    assert model.pairs[0] == ("年假政策", "甲")


def test_rerank_truncates_long_documents():
    model = FakeCrossEncoder([0.0])
    r = _make_reranker(model)
    long_doc = mkdoc("长" * 2000)
    r.rerank("q", [long_doc])
    assert len(model.pairs[0][1]) == 800


def test_unavailable_model_preserves_order_with_zero_scores():
    from app.rag.reranker import CrossEncoderReranker

    r = _make_reranker(None)
    docs = [mkdoc("甲"), mkdoc("乙")]
    scored = r.rerank("q", docs)
    assert [d.page_content for d, _ in scored] == ["甲", "乙"]
    assert all(s == 0.0 for _, s in scored)
    assert not CrossEncoderReranker.available.fget(r)


def test_init_failure_degrades_to_unavailable(monkeypatch):
    # 模型下载/加载失败时降级为不重排，链路可用（__init__ 内吞异常）
    monkeypatch.setattr(
        "app.rag.reranker.TextCrossEncoder",
        mock.MagicMock(side_effect=RuntimeError("download failed")),
    )
    from app.rag.reranker import CrossEncoderReranker

    r = CrossEncoderReranker()
    assert not r.available
    docs = [mkdoc("甲")]
    assert r.rerank("q", docs) == [(docs[0], 0.0)]


def test_empty_docs_short_circuits():
    r = _make_reranker(FakeCrossEncoder([]))
    assert r.rerank("q", []) == []
