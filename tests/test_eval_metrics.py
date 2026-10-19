"""评测脚本的纯函数：召回汇总口径与 RAGAS 抽样方式。

这些逻辑决定对外公布的指标代表什么，比一次单点分数更值得有回归测试：
- 行级 recall 会被「一个 chunk 生成了几个问题」加权；
- head 抽样会让 n=100 落进单一源文档。
"""
from scripts.eval_ragas import composition, select_sample
from scripts.eval_recall import summarize


def _qa(chunk_id: str, source: str = "a.md", question: str = "") -> dict:
    return {"chunk_id": chunk_id, "source": source, "question": question or f"q-{chunk_id}"}


def test_recall_is_hit_fraction_over_rows():
    qa_set = [_qa("c1"), _qa("c1"), _qa("c2"), _qa("c3")]
    # 第 4 题未召回
    report = summarize(qa_set, [1, 3, 5, None])
    assert report["recall@1"] == 0.25
    assert report["recall@3"] == 0.5
    assert report["recall@5"] == 0.75


def test_chunk_macro_removes_duplicate_question_weighting():
    """c1 有两题（一题命中第 1、一题未召回），c2 一题命中第 1：
    行级 recall@1 = 2/3；chunk 级 = mean(0.5, 1.0) = 0.75。
    两个口径不同，报告必须同时给出。"""
    qa_set = [_qa("c1"), _qa("c1"), _qa("c2")]
    report = summarize(qa_set, [1, None, 1])
    assert report["recall@1"] == round(2 / 3, 4)
    assert report["recall@1_chunk_macro"] == 0.75


def test_composition_reports_unique_chunks_and_sources():
    qa_set = [_qa("c1", "a.md"), _qa("c1", "a.md"), _qa("c2", "b.md")]
    comp = summarize(qa_set, [1, 1, 1])["composition"]
    assert comp["rows"] == 3
    assert comp["unique_gold_chunks"] == 2
    assert comp["sources"] == {"a.md": 2, "b.md": 1}
    assert comp["rows_per_chunk_mean"] == 1.5


def test_head_sampling_reproduces_historical_slice():
    qa_set = [_qa(f"doc1-{i}", "doc1.md") for i in range(5)] + [
        _qa(f"doc2-{i}", "doc2.md") for i in range(5)
    ]
    sample = select_sample(qa_set, 3, "head")
    assert [s["chunk_id"] for s in sample] == ["doc1-0", "doc1-1", "doc1-2"]
    assert composition(sample)["sources"] == {"doc1.md": 3}


def test_stratified_sampling_covers_all_sources():
    qa_set = [_qa(f"doc1-{i}", "doc1.md") for i in range(5)] + [
        _qa(f"doc2-{i}", "doc2.md") for i in range(5)
    ]
    sample = select_sample(qa_set, 4, "stratified")
    comp = composition(sample)
    assert comp["sources"] == {"doc1.md": 2, "doc2.md": 2}
    assert [s["chunk_id"] for s in sample] == ["doc1-0", "doc2-0", "doc1-1", "doc2-1"]


def test_stratified_sampling_is_deterministic_and_bounded():
    qa_set = [_qa(f"c{i}", f"doc{i % 3}.md") for i in range(9)]
    first = select_sample(qa_set, 5, "stratified")
    second = select_sample(qa_set, 5, "stratified")
    assert first == second
    assert len(first) == 5
    assert len(select_sample(qa_set, 50, "stratified")) == len(qa_set)
