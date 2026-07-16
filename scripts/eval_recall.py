# -*- coding: utf-8 -*-
"""Recall@k 评测（v0.4）：纯向量基线 vs 混合检索。

指标口径：Recall@k = 评测集问题中，ground-truth chunk 出现在检索结果 top-k 内的比例。

用法：
  .venv/Scripts/python scripts/eval_recall.py             # 混合检索
  .venv/Scripts/python scripts/eval_recall.py --baseline  # 纯向量基线（v0.1 形态）
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.retriever import HybridRetriever
from app.rag.vector_store import _doc_id, get_store

QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
REPORT = ROOT / "data" / "qa_set" / "recall_report.json"


def run(qa_set, retrieve_fn, label, k=5):
    at = {1: 0, 3: 0, 5: 0}
    n = len(qa_set)
    for i, qa in enumerate(qa_set, 1):
        docs = retrieve_fn(qa["question"], top_k=k)
        ids = [_doc_id(d) for d, _ in docs]
        try:
            pos = ids.index(qa["chunk_id"]) + 1  # 1-based
        except ValueError:
            pos = None
        for key in at:
            if pos is not None and pos <= key:
                at[key] += 1
        if i % 50 == 0:
            print(f"  [{label}] {i}/{n}...")
    return {f"recall@{key}": round(at[key] / n, 4) for key in at}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="纯向量基线（不启用混合检索）")
    args = parser.parse_args()

    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    print(f"[eval] 评测集 {len(qa_set)} 条，top-k = 5")

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

    print("\n===== 结果 =====")
    for k_, v in result.items():
        print(f"  {k_}: {v}")
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 报告已存：{REPORT}")


if __name__ == "__main__":
    main()
