"""评测指标层单测（工单 22）。纯函数，不碰网络与模型。

这些数字决定 README 里怎么写，所以它们的实现必须可验证 ——
旧脚本把统计藏在一次性循环里，没人能检查它对不对。
"""
import math

import pytest

from app.eval.metrics import (
    aggregate_by_chunk,
    ndcg_at,
    per_source,
    recall_at,
    reciprocal_rank,
    stratified_sample,
    summarize,
    wilson_interval,
)


def _row(chunk, source, r1, r3, r5, rr, nd):
    return {"chunk_id": chunk, "source": source, "recall@1": r1, "recall@3": r3,
            "recall@5": r5, "reciprocal_rank": rr, "ndcg_at": nd}


# ---- Wilson 区间 ----
def test_wilson_contains_point_estimate():
    lo, hi = wilson_interval(70, 80)
    assert lo <= 0.875 <= hi
    assert lo > 0 and hi < 1


def test_wilson_is_wider_than_the_naive_reading_of_small_samples():
    """小样本区间必须明显宽 —— 这正是旧报告缺的东西。"""
    lo, hi = wilson_interval(1, 3)
    assert lo == 0.0 or lo < 0.2
    assert hi > 0.7


def test_wilson_never_leaves_unit_interval_and_handles_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    for s, n in [(0, 5), (5, 5), (1, 1000), (999, 1000)]:
        lo, hi = wilson_interval(s, n)
        assert 0.0 <= lo <= hi <= 1.0, (s, n, lo, hi)


def test_wilson_keeps_extreme_estimates_honest():
    """断言全部取自实测值，而不是我对 Wilson 公式的记忆。

    实测：wilson_interval(0, 100) == (0.0000, 0.0370)
          wilson_interval(100, 100) == (0.9630, 1.0000)

    关键点：p=0 时下界确实为 0，Wilson 的"诚实"体现在**上界不为 0** ——
    它说"100 次全错，真实错误率仍可能高到 3.7%"。Wald 正态近似在这里会退化
    成 [0, 0]，等于宣布"真值确定是 0"，那是错的。
    """
    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == pytest.approx(0.0)       # p=0 时下界为 0，不假装更保守
    assert 0.0 < hi0 < 0.05, hi0           # 上界非零：没有把"没观察到"说成"不存在"

    lo1, hi1 = wilson_interval(100, 100)
    # 用容差而不是 ==：中心/边界是浮点算出来的，实测 hi1 = 0.9999999999999999
    assert hi1 == pytest.approx(1.0)
    assert 0.95 < lo1 < 1.0, lo1           # 对称地，也不断言真值确定是 1

    # 同比例下样本量越大区间越窄（这就是为什么必须报区间而不是点值）
    assert (50, 100) and (wilson_interval(500, 1000)[1] - wilson_interval(500, 1000)[0]) \
        < (wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0])

    # 80 个块的量级：区间宽度应该有感知度（约 ±7pp），不能像 334 那样假装很窄
    lo, hi = wilson_interval(70, 80)
    assert 0.10 < (hi - lo) < 0.20, (lo, hi)


# ---- 位置折扣 ----
@pytest.mark.parametrize("rank,expected", [
    (1, 1.0), (2, 1 / math.log2(3)), (3, 1 / math.log2(4)), (5, 1 / math.log2(6)),
])
def test_ndcg_single_gold_uses_log2_discount(rank, expected):
    ranked = ["x"] * (rank - 1) + ["g"] + ["y"] * 4
    assert abs(ndcg_at(ranked, "g", 5) - expected) < 1e-9


def test_ndcg_and_recall_and_mrr_agree_on_miss():
    ranked = ["a", "b", "c", "d", "e"]
    assert recall_at(ranked, "g", 5) == 0
    assert reciprocal_rank(ranked, "g", 5) == 0.0
    assert ndcg_at(ranked, "g", 5) == 0.0


def test_mrr_rewards_higher_rank():
    assert reciprocal_rank(["g", "x"], "g", 2) == 1.0
    assert reciprocal_rank(["x", "g"], "g", 2) == 0.5


def test_recall_at_k_is_monotonic_in_k():
    ranked = ["x", "y", "g", "z"]
    assert recall_at(ranked, "g", 1) == 0
    assert recall_at(ranked, "g", 3) == 1
    assert recall_at(ranked, "g", 5) == 1


# ---- 统计单元 ----
def test_chunk_level_is_the_conservative_reading():
    """同块多题时，chunk 级 all-of 不可能高于 row 级 —— 这是刻意的保守方向。"""
    rows = [_row("A", "s1", 1, 1, 1, 1.0, 1.0)] * 3 + [_row("B", "s2", 0, 0, 0, 0.0, 0.0)] * 3
    s = summarize(rows, ks=(1,))
    assert s["row/recall@1"]["value"] == 0.5
    assert s["chunk/recall@1"]["value"] == 0.5
    assert s["chunk/recall@1"]["n"] == 2  # 独立单元是 2，不是 6

    rows2 = [_row("A", "s1", 1, 1, 1, 1.0, 1.0)] * 3 + [_row("A", "s1", 0, 0, 0, 0.0, 0.0)]
    s2 = summarize(rows2, ks=(1,))
    assert s2["row/recall@1"]["value"] == 0.75
    assert s2["chunk/recall@1"]["value"] == 0.0, "块内有一题失败则该块不算命中（all-of）"
    assert s2["row/recall@1"]["value"] > s2["chunk/recall@1"]["value"]


def test_aggregate_by_chunk_counts():
    rows = [_row("A", "s", 1, 1, 1, 1, 1), _row("A", "s", 0, 1, 1, 0, 0), _row("B", "s", 1, 1, 1, 1, 1)]
    agg = aggregate_by_chunk(rows, hit_key="recall@1")
    assert agg == {"A": (1, 2), "B": (1, 1)}


def test_summary_reports_distinct_block_count():
    rows = [_row("A", "s", 1, 1, 1, 1, 1)] * 4 + [_row("B", "s", 1, 1, 1, 1, 1)] * 2
    assert summarize(rows, ks=(1,))["distinct_chunk_ids"] == 2


# ---- 分层抽样 ----
def make(n_per_source=10):
    sources = ["劳动合同法", "社保法", "员工手册", "考勤制度"]
    rows = []
    for s in sources:
        rows.extend([_row(f"{s}-{i}", s, 1, 1, 1, 1, 1) for i in range(n_per_source)])
    return rows


def test_stratified_sample_size_and_proportion():
    rows = make(10)
    sample = stratified_sample(rows, 20)
    assert len(sample) == 20, "名额必须精确等于 size（最大余数法）"
    counts = {s: sum(1 for r in sample if r["source"] == s) for s in {r["source"] for r in rows}}
    assert all(v == 5 for v in counts.values()), counts


def test_stratified_sample_is_deterministic_per_seed():
    rows = make(8)
    a = stratified_sample(rows, 16, seed=1)
    b = stratified_sample(rows, 16, seed=1)
    c = stratified_sample(rows, 16, seed=2)
    assert [(r["source"], r["chunk_id"]) for r in a] == [(r["source"], r["chunk_id"]) for r in b]
    assert [(r["source"], r["chunk_id"]) for r in a] != [(r["source"], r["chunk_id"]) for r in c]


def test_stratified_sample_covers_all_strata_unlike_head_slice():
    """旧口径 qa_set[:100] 是头切片，只覆盖一个来源；分层抽样必须每个来源都有。"""
    # 每个来源 120 条 => 头 100 条全部落在第一个来源里，正是旧口径的病症
    rows = make(120)
    head = rows[:100]
    sample = stratified_sample(rows, 100)
    assert len({r["source"] for r in head}) == 1, "构造前提：头切片只覆盖一个来源"
    strata = {r["source"] for r in sample}
    assert len(strata) == 4, strata
    # 而且每个来源都被按比例抽到，不是"凑齐 4 个名字"
    counts = {s: sum(1 for r in sample if r["source"] == s) for s in strata}
    assert all(v == 25 for v in counts.values()), counts


def test_stratified_sample_handles_size_larger_than_population():
    rows = make(2)
    assert len(stratified_sample(rows, 999)) == len(rows)


def test_per_source_breakdown():
    rows = [_row("a", "s1", 1, 1, 1, 1, 1), _row("b", "s1", 0, 1, 1, 0, 0),
            _row("c", "s2", 1, 1, 1, 1, 1)]
    out = per_source(rows, k=1)
    assert out["s1"]["n"] == 2 and out["s1"]["value"] == 0.5
    assert out["s2"]["n"] == 1 and out["s2"]["value"] == 1.0
