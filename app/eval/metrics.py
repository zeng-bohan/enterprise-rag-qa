"""评测指标层（工单 22 / 24 / 25）：纯函数，可单测，不碰网络与模型。

为什么这些函数值得单独存在
--------------------------
旧评测脚本直接在 `scripts/eval_recall.py` 里累加计数然后除以 n。三个后果：

1. **n=334 被当成 334 次独立试验**，而那 334 行只覆盖 80 个不同 gold chunk
   （平均 4.2 问/块）。同一块上的 4 个问题高度相关，把它们当独立样本会**低估
   置信区间**，于是 0.1 个百分点的差异看起来像改进。这里把统计单元显式改成
   chunk，并给出 Wilson 区间。
2. **没有区间**：点值 87.4% 与 87.9% 无法区分。
3. **只有 Recall@k**：看不出"正确块排在第 2 还是第 5"，也看不出前几名里
   有多少无关块。补 MRR 与 nDCG。

分层抽样（`stratified_sample`）解决另一个问题：旧 RAGAS 用 `qa_set[:100]`，
那 100 条实测 100% 来自单一文档，等于只评了八分之一的语料。
"""
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """比例的 Wilson 95% 区间。

    不用正态近似（Wald）：成功数接近 0 或 1、或样本很小时，Wald 会给出越界
    甚至负数的区间。80 个块这个量级正是它会失手的场景。
    """
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4.0 * trials * trials))
    return (max(0.0, center - margin), min(1.0, center + margin))


def recall_at(ranked_ids: Sequence[str], gold_id: str, k: int) -> int:
    return int(gold_id in list(ranked_ids)[:k])


def reciprocal_rank(ranked_ids: Sequence[str], gold_id: str, k: int) -> float:
    """gold 在前 k 名内的 1/rank，不在则为 0。"""
    for rank, cid in enumerate(list(ranked_ids)[:k], start=1):
        if cid == gold_id:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked_ids: Sequence[str], gold_id: str, k: int) -> float:
    """单相关文档的 nDCG@k。

    增益用 log2(rank+1) 折扣；理想情形（gold 在第 1 位）分母为 1，
    所以返回值直接就是折扣后的增益，落在 [0,1]。
    """
    for rank, cid in enumerate(list(ranked_ids)[:k], start=1):
        if cid == gold_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def aggregate_by_chunk(
    rows: Iterable[Dict], hit_key: str = "hit", group_key: str = "chunk_id"
) -> Dict[str, Tuple[int, int]]:
    """按 chunk 归并：{chunk_id: (命中数, 问题数)}。

    统计单元的选择是这套指标的核心修正。报告时同时给两个口径：
    row 级与旧数字可比，chunk 级才是可信的独立样本数。
    """
    buckets: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        buckets[row[group_key]].append(int(row[hit_key]))
    return {cid: (sum(hits), len(hits)) for cid, hits in buckets.items()}


def summarize(
    rows: List[Dict], ks: Tuple[int, ...] = (1, 3, 5), group_key: str = "chunk_id"
) -> Dict:
    """把逐题结果汇总成 row 级与 chunk 级两套指标 + Wilson 区间。

    chunk 级的命中判定用"该块的所有问题是否全部命中"（all-of），而不是
    "至少一个命中"（any-of）：any-of 会因为只要有一题命中就算块命中，
    从而把同一块上其它题的失败抹平，指标只会更乐观。取 all-of 是保守方向。
    """
    out: Dict = {"n_rows": len(rows)}
    for k in ks:
        hits = [r[f"recall@{k}"] for r in rows]
        n = len(hits)
        s = sum(hits)
        lo, hi = wilson_interval(s, n) if n else (0.0, 0.0)
        out[f"row/recall@{k}"] = {"value": round(s / n, 4) if n else 0.0,
                                  "n": n, "ci95": [round(lo, 4), round(hi, 4)]}

        by_chunk = aggregate_by_chunk(rows, hit_key=f"recall@{k}", group_key=group_key)
        blocks = len(by_chunk)
        all_hit = sum(1 for h, t in by_chunk.values() if t and h == t)
        clo, chi = wilson_interval(all_hit, blocks) if blocks else (0.0, 0.0)
        out[f"chunk/recall@{k}"] = {"value": round(all_hit / blocks, 4) if blocks else 0.0,
                                    "n": blocks, "ci95": [round(clo, 4), round(chi, 4)]}

    for name, fn in (("mrr@5", "reciprocal_rank"), ("ndcg@5", "ndcg_at")):
        vals = [r[fn] for r in rows if fn in r]
        out[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["distinct_" + group_key + "s"] = len({r[group_key] for r in rows})
    return out


def per_source(rows: List[Dict], k: int = 1, source_key: str = "source") -> Dict[str, Dict]:
    """按来源文档拆分。整份指标可能被一个"容易"的文档主导，拆开才看得见。"""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        buckets[r.get(source_key, "未知")].append(r[f"recall@{k}"])
    return {
        src: {"n": len(h), "value": round(sum(h) / len(h), 4) if h else 0.0}
        for src, h in sorted(buckets.items())
    }


def stratified_sample(rows: List[Dict], size: int, strata_key: str = "source",
                      seed: int = 20260922) -> List[Dict]:
    """按来源分层随机抽样（工单 21 对 `qa_set[:100]` 的修正）。

    固定 seed：抽样结果必须可复现，否则两次跑出的指标差异分不清是模型差异
    还是抽样噪声。最大余数法分配名额，保证总数精确等于 size。
    """
    rng = random.Random(seed)
    by_strata: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_strata[r.get(strata_key, "未知")].append(r)

    total = len(rows)
    if total == 0 or size >= total:
        out = [dict(r) for r in rows]
        rng.shuffle(out)
        return out

    # 先取整数部分，再把剩余名额按小数部分从大到小分配
    quotas: Dict[str, float] = {s: len(v) / total * size for s, v in by_strata.items()}
    chosen = {s: int(q) for s, q in quotas.items()}
    leftover = size - sum(chosen.values())
    for s in sorted(quotas, key=lambda x: -(quotas[x] - int(quotas[x])))[:max(0, leftover)]:
        chosen[s] += 1

    sample: List[Dict] = []
    for stratum, items in by_strata.items():
        take = min(chosen.get(stratum, 0), len(items))
        sample.extend(rng.sample(items, take))
    rng.shuffle(sample)
    return sample
