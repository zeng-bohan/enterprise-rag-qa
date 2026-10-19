# -*- coding: utf-8 -*-
"""拒答评测：知识库外的问题应当被拒答，知识库内的问题不应被误拒。

口径：
- 负例集 `data/qa_set/refusal_set.json` 是 20 个语料答不上的问题（年会、宿舍、
  期权等），检索器返回空即记为正确拒答；
- 正例从 `qa_300.json` 按文档分层抽样，检索器返回空即记为误拒；
- 拒答判定完全走 `HybridRetriever.retrieve`，与线上链路一致：前置分数下限与
  重排下限都会参与，生成模型不参与（拒答的定义就是"不调用 LLM"）。

这个评测回答的是"闸门有没有用"，不评价生成的答案质量。拒答阈值是全局常数，
换语料后应重跑本脚本再调阈值。

用法：
  .venv/Scripts/python scripts/eval_refusal.py --positives 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RERANK_FLOOR
from scripts.eval_ragas import select_sample

NEG_PATH = ROOT / "data" / "qa_set" / "refusal_set.json"
QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
REPORT = ROOT / "data" / "qa_set" / "refusal_report.json"


def summarize(neg_total: int, neg_refused: int, pos_total: int, pos_refused: int) -> dict:
    """纯函数：把计数换成比率，便于离线测试。"""
    return {
        "negatives": neg_total,
        "negatives_refused": neg_refused,
        "refusal_rate": round(neg_refused / neg_total, 4) if neg_total else None,
        "positives": pos_total,
        "positives_refused": pos_refused,
        "false_refusal_rate": round(pos_refused / pos_total, 4) if pos_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="拒答率与误拒率评测")
    parser.add_argument("--positives", type=int, default=50, help="正例抽样条数")
    args = parser.parse_args()

    negatives = json.loads(NEG_PATH.read_text(encoding="utf-8"))
    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    positives = select_sample(qa_set, args.positives, "stratified")

    pipeline = RAGPipeline()
    retriever = pipeline.retriever

    neg_refused = 0
    for item in negatives:
        hits = retriever.retrieve(item["question"])
        refused = not hits
        neg_refused += int(refused)
        print(f"  [neg] {'REFUSED' if refused else 'ANSWERED'}  {item['question']}")

    pos_refused = 0
    for item in positives:
        hits = retriever.retrieve(item["question"])
        refused = not hits
        pos_refused += int(refused)
        if refused:
            print(f"  [pos] FALSE-REFUSAL  {item['question']}")

    summary = summarize(len(negatives), neg_refused, len(positives), pos_refused)
    summary["threshold"] = {
        "score_threshold": settings.score_threshold,
        "rerank_floor": RERANK_FLOOR,
    }
    summary["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n===== 结果 =====")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 报告已存：{REPORT}")


if __name__ == "__main__":
    main()
