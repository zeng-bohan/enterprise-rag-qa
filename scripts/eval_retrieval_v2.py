"""检索评测 v2（工单 22 / 24 / 25）：三条基线对照 + chunk 级置信区间 + 拒答率。

与旧 `eval_recall.py` 的差别，逐条对应评审发现
--------------------------------------------
- **报两个统计单元**：row 级（与旧数字可比）与 chunk 级（可信的独立样本数，
  带 Wilson 95% 区间）。334 行只覆盖 80 个 gold chunk，旧口径把 4.17 个相关问题
  当成 4.17 次独立试验，区间因此偏窄。
- **补 MRR@5 / nDCG@5**：只有 Recall@k 看不出正确块排第 2 还是第 5。
- **三条模式全部落盘**：hybrid / 纯向量 / 纯 BM25。旧脚本支持 --baseline
  但从没把对照行提交过，于是"混合检索更好"这个说法在仓库里没有证据。
- **per-source 明细**：整份指标可能被一个最容易的文档主导。
- **负样本集**：拒答此前零量化，误拒率与误答率都不知道。

前置：先把语料灌进向量库
    .venv/Scripts/python scripts/ingest_docs.py

用法
    .venv/Scripts/python scripts/eval_retrieval_v2.py
产物：data/qa_set/retrieval_report_v2.json
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.eval.metrics import (  # noqa: E402
    ndcg_at,
    per_source,
    recall_at,
    reciprocal_rank,
    summarize,
)
from app.rag.bm25_index import BM25Index  # noqa: E402
from app.rag.registry import get_registry  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from app.rag.vector_store import _doc_id, get_store  # noqa: E402

QA = ROOT / "data" / "qa_set" / "qa_300.json"
NEG = ROOT / "data" / "qa_set" / "negative_set.json"
REPORT = ROOT / "data" / "qa_set" / "retrieval_report_v2.json"
KS = (1, 3, 5)


def run_mode(retriever, store, bm25, mode: str, query: str, kb_id: str):
    """返回按相关性降序的 chunk 内容哈希列表。

    模式间的不对称是**有意**的：hybrid 走完整生产路径（含重排与拒答地板），
    两条基线走各自的裸检索 —— 要比较的正是"生产路径"与"更简单的替代方案"。
    """
    if mode == "hybrid":
        return [_doc_id(d) for d, _ in retriever.retrieve(query, kb_id=kb_id, top_k=5)]
    if mode == "vector":
        return [_doc_id(d) for d, _ in store.search(query, 5, kb_id=kb_id)]
    if mode == "bm25":
        return [_doc_id(d) for d, _ in bm25.search(query, 5)]
    raise ValueError(f"未知模式：{mode}")


def eval_rows(retriever, store, bm25, mode, qa, kb_id):
    rows, latencies = [], []
    for i, item in enumerate(qa, 1):
        gold = item["chunk_id"]
        t0 = time.perf_counter()
        ranked = run_mode(retriever, store, bm25, mode, item["question"], kb_id)
        latencies.append(time.perf_counter() - t0)
        row = {
            "question": item["question"],
            "source": item.get("source", "未知"),
            "chunk_id": gold,
            "ranked": ranked,
        }
        for k in KS:
            row[f"recall@{k}"] = recall_at(ranked, gold, k)
        row["reciprocal_rank"] = reciprocal_rank(ranked, gold, 5)
        row["ndcg_at"] = ndcg_at(ranked, gold, 5)
        rows.append(row)
        if i % 50 == 0:
            print(f"  [{mode}] {i}/{len(qa)}")
    stats = summarize(rows, ks=KS)
    stats["per_source_recall@1"] = per_source(rows, k=1)
    stats["latency_ms"] = {
        "p50": round(statistics.median(latencies) * 1000, 1),
        "p95": round(statistics.quantiles(latencies, n=20)[18] * 1000, 1) if len(latencies) >= 20 else None,
    }
    # 空结果率 = 触发拒答的比例。召回与拒答共用地板，所以这个数字必须一起看：
    # 否则"阈值调高 → recall 掉了"会被读成"检索变差了"
    stats["empty_rate"] = round(sum(1 for r in rows if not r["ranked"]) / len(rows), 4)
    return stats, rows


def eval_refusal(retriever, kb_id, neg_items):
    """拒答行为：应当拒答的是否拒了（漏拒率），应当回答的是否被误拒（误拒率）。"""
    out = {}
    for category in ("irrelevant", "near_miss", "in_scope"):
        subset = [n for n in neg_items if n["category"] == category]
        if not subset:
            continue
        refused = 0
        for n in subset:
            hits = retriever.retrieve(n["question"], kb_id=kb_id, top_k=5)
            is_refusal = len(hits) == 0
            if is_refusal:
                refused += 1
        out[category] = {
            "n": len(subset),
            "refused": refused,
            "refusal_rate": round(refused / len(subset), 4),
        }
    should_refuse = sum(v["refused"] for k, v in out.items() if k != "in_scope")
    should_refuse_n = sum(v["n"] for k, v in out.items() if k != "in_scope")
    answerable_refused = out.get("in_scope", {}).get("refused", 0)
    answerable_n = out.get("in_scope", {}).get("n", 0)
    return {
        "by_category": out,
        # 负样本被放行 = 会拿着无关资料硬答 = 幻觉风险
        "false_accept_rate": round(1 - should_refuse / should_refuse_n, 4) if should_refuse_n else None,
        # 正样本被拒 = 白白损失可用性
        "false_refusal_rate": round(answerable_refused / answerable_n, 4) if answerable_n else None,
    }


def main() -> int:
    if not QA.exists():
        print(f"找不到评测集：{QA}")
        return 1
    qa = json.loads(QA.read_text(encoding="utf-8"))
    neg = json.loads(NEG.read_text(encoding="utf-8"))["items"]

    store = get_store()
    registry = get_registry()
    kb_name = settings.default_kb
    kb_id = registry.kb_id_by_name(kb_name)
    if kb_id is None:
        print(f"默认知识库「{kb_name}」不存在或为空，请先运行 scripts/ingest_docs.py")
        return 1
    if store.count(kb_id=kb_id) == 0:
        print("知识库内没有 chunk，请先运行 scripts/ingest_docs.py")
        return 1

    retriever = HybridRetriever(store)
    bm25 = BM25Index(store.get_all_documents(kb_id))

    print(f"[eval] backend={settings.vector_backend} chunks={store.count(kb_id=kb_id)} "
          f"questions={len(qa)} distinct_chunks={len({q['chunk_id'] for q in qa})}")
    print(f"[eval] reranker_available={retriever._reranker.available}")  # noqa: SLF001

    modes = {}
    detail = {}
    for mode in ("hybrid", "vector", "bm25"):
        print(f"[eval] 模式 {mode} …")
        stats, rows = eval_rows(retriever, store, bm25, mode, qa, kb_id)
        modes[mode] = {k: v for k, v in stats.items()}
        # 落盘逐题命中位次，便于事后复核（不存 ranked 全文，避免报告膨胀）
        detail[mode] = [
            {kk: r[kk] for kk in ("question", "source", "chunk_id", "recall@1", "recall@3",
                                  "recall@5", "reciprocal_rank", "ndcg_at")}
            for r in rows
        ]

    print("[eval] 拒答行为 …")
    refusal = eval_refusal(retriever, kb_id, neg)

    report = {
        "conditions": {
            "backend": settings.vector_backend,
            "kb_id": kb_id,
            "chunks_in_kb": store.count(kb_id=kb_id),
            "n_questions": len(qa),
            "distinct_gold_chunks": len({q["chunk_id"] for q in qa}),
            "reranker": "bge-reranker-base" if retriever._reranker.available else "UNAVAILABLE(降级)",  # noqa: SLF001
            "embed_model": settings.embed_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "score_threshold": settings.score_threshold,
            "llm_generation_model": settings.deepseek_model,
            "llm_base_url": settings.deepseek_base_url,
            "note": (
                "hybrid 行含重排与拒答地板，vector/bm25 行为裸检索。"
                "chunk 级口径用 all-of（同块所有问题都命中才算命中），是保守方向。"
            ),
        },
        "modes": modes,
        "refusal": refusal,
        "question_source_distribution": dict(Counter(q.get("source", "未知") for q in qa)),
        "per_question": detail,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'mode':<8}{'row R@1':>10}{'chunk R@1 (95% CI)':>26}{'MRR@5':>8}{'nDCG@5':>8}{'empty':>8}")
    for mode, stats in modes.items():
        r1 = stats["row/recall@1"]["value"]
        c1 = stats["chunk/recall@1"]
        print(f"{mode:<8}{r1*100:>9.1f}%"
              f"{c1['value']*100:>13.1f}% [{c1['ci95'][0]*100:.1f}, {c1['ci95'][1]*100:.1f}] n={c1['n']:<4}"
              f"{stats['mrr@5']:>8.3f}{stats['ndcg@5']:>8.3f}{stats['empty_rate']*100:>7.1f}%")
    print(f"\n[refusal] false_accept={refusal['false_accept_rate']} "
          f"false_refusal={refusal['false_refusal_rate']} by_cat={ {k: v['refusal_rate'] for k, v in refusal['by_category'].items()} }")
    print(f"[eval] 报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
