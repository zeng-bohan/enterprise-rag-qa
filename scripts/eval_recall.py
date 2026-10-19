# -*- coding: utf-8 -*-
"""Recall@k 评测（v0.4）：纯向量基线 vs 混合检索。

指标口径：Recall@k = 评测集问题中，ground-truth chunk 出现在检索结果 top-k 内的比例。

评测集是「每个 chunk 生成 3~5 个问题」，所以行级 recall 会被问题多的 chunk 加权。
报告同时给出 chunk 级宏平均（每个 gold chunk 先算自身命中率，再对 chunk 取平均），
两者一起看才能区分「检索变好」与「评测集采样偏向某几段文本」。

用法：
  .venv/Scripts/python scripts/eval_recall.py             # 混合检索
  .venv/Scripts/python scripts/eval_recall.py --baseline  # 纯向量基线（v0.1 形态）
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.retriever import HybridRetriever
from app.rag.vector_store import _doc_id, get_store

QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
REPORT = ROOT / "data" / "qa_set" / "recall_report.json"
K_VALUES = (1, 3, 5)


def summarize(qa_set: Sequence[dict], positions: Sequence[Optional[int]]) -> dict:
    """把每题命中位置汇总成报告字段（纯函数，便于离线测试）。

    positions[i] 是第 i 题的 gold chunk 在检索结果中的 1-based 排名，未召回为 None。
    """
    n = len(qa_set)
    row_metrics = {
        f"recall@{k}": round(sum(1 for p in positions if p is not None and p <= k) / n, 4)
        for k in K_VALUES
    }

    per_chunk: Dict[str, List[Optional[int]]] = defaultdict(list)
    for qa, pos in zip(qa_set, positions):
        per_chunk[qa["chunk_id"]].append(pos)
    chunk_metrics = {}
    for k in K_VALUES:
        rates = [
            sum(1 for p in ps if p is not None and p <= k) / len(ps)
            for ps in per_chunk.values()
        ]
        chunk_metrics[f"recall@{k}_chunk_macro"] = round(sum(rates) / len(rates), 4)

    composition = {
        "rows": n,
        "unique_gold_chunks": len(per_chunk),
        "unique_questions": len({qa["question"] for qa in qa_set}),
        "sources": dict(Counter(qa.get("source", "") for qa in qa_set).most_common()),
        "rows_per_chunk_mean": round(n / len(per_chunk), 2),
    }
    return {**row_metrics, **chunk_metrics, "composition": composition}


def run(qa_set, retrieve_fn, label, k=5) -> dict:
    positions: List[Optional[int]] = []
    n = len(qa_set)
    for i, qa in enumerate(qa_set, 1):
        docs = retrieve_fn(qa["question"], top_k=k)
        ids = [_doc_id(d) for d, _ in docs]
        try:
            pos = ids.index(qa["chunk_id"]) + 1  # 1-based
        except ValueError:
            pos = None
        positions.append(pos)
        if i % 50 == 0:
            print(f"  [{label}] {i}/{n}...")
    return summarize(qa_set, positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="纯向量基线（不启用混合检索）")
    args = parser.parse_args()

    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    print(f"[eval] 评测集 {len(qa_set)} 行，{len({q['chunk_id'] for q in qa_set})} 个唯一 gold chunk，top-k = 5")

    store = get_store()
    if args.baseline:
        print("[eval] 模式：纯向量基线（v0.1 形态）")
        metrics = run(qa_set, store.search, "baseline")
        result = {"mode": "baseline", "n": len(qa_set), **metrics}
    else:
        print("[eval] 模式：混合检索（改写+BM25+向量+RRF+重排）")
        retriever = HybridRetriever(store)
        metrics = run(qa_set, retriever.retrieve, "hybrid")
        result = {"mode": "hybrid", "n": len(qa_set), **metrics}

    result["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n===== 结果 =====")
    for k_, v in result.items():
        print(f"  {k_}: {v}")
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 报告已存：{REPORT}")


if __name__ == "__main__":
    main()
