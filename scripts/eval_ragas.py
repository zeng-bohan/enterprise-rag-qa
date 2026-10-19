# -*- coding: utf-8 -*-
"""RAGAs 评测流水线（v0.4）：Faithfulness（忠实度，1-幻觉率）+ AnswerRelevancy（答案相关性）。

口径：
- Faithfulness：答案里的每个论断是否都能从「检索到的上下文」找到依据——衡量幻觉；
- AnswerRelevancy：答案是否切题（用问题-答案语义相似度 + 反向问题生成）；
- LLM-as-Judge 用 DeepSeek，Embedding 用本地 BGE——整套评测零外部依赖、可复现。

用法：
  .venv/Scripts/python scripts/eval_ragas.py          # 抽样 100 条（控制成本）
  .venv/Scripts/python scripts/eval_ragas.py --full   # 全量
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 限流：本地 ONNX 推理（BGE/重排器）只使用 2 个 CPU 线程，评测期间不影响日常使用
os.environ["OMP_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.rag.pipeline import RAGPipeline

QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
OUT = ROOT / "data" / "qa_set" / "ragas_report.json"


def _ragas():
    """按需导入 ragas / openai。

    评测依赖（ragas、openai）单独放在 requirements-eval.txt，不进运行时依赖：
    只想跑离线测试或启动服务的人不该为此多装一套 LLM 评测库。把它放在函数里，
    本模块的纯函数（抽样、样本统计）才能在离线测试里被导入。
    """
    from openai import AsyncOpenAI
    from ragas import SingleTurnSample
    from ragas.embeddings import BaseRagasEmbedding
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    return SimpleTurnSample, BaseRagasEmbedding, llm_factory, AnswerRelevancy, Faithfulness, AsyncOpenAI


def make_ragas_embeddings():
    """本地 BGE 的 ragas 现代 embedding 接口适配。"""
    _, BaseRagasEmbedding, *_ = _ragas()

    class BGEEmbeddingsRagas(BaseRagasEmbedding):
        def __init__(self) -> None:
            self._emb = BGEEmbeddings()

        def embed_text(self, text: str) -> list[float]:
            return self._emb.embed_query(text)

        async def aembed_text(self, text: str) -> list[float]:
            return await asyncio.to_thread(self._emb.embed_query, text)

    return BGEEmbeddingsRagas()


def score(make_metric, samples: list, keys: list[str], batch: int = 20) -> list[float]:
    """ragas 0.4.3 的 collections 指标不兼容 evaluate()，直接调用其批量打分接口。

    注意：abatch_score 会把字典的每个键都作为关键字参数传给 ascore()，
    因此必须按指标的参数签名裁剪输入字段（Faithfulness 不需要 reference 等字段）；
    指标内部一律走异步调用，客户端需为 AsyncOpenAI。
    按 batch 分批 gather：把 DeepSeek 并发压在每批 20 路以内，避免 429 拖慢评测；
    每个小批次新建 client/llm，避免 AsyncOpenAI 跨事件循环复用报错。
    """
    _, _, llm_factory, _, _, AsyncOpenAI = _ragas()
    inputs = [{k: getattr(s, k) for k in keys} for s in samples]
    values: list[float] = []
    for start in range(0, len(inputs), batch):
        client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
        # ragas 默认 max_tokens=1024：Faithfulness 一次输出全部语句判定 JSON，
        # 长答案会被截断抛 IncompleteOutputException；deepseek-chat 输出上限 8192
        llm = llm_factory(settings.deepseek_model, client=client, max_tokens=8192)
        chunk = inputs[start:start + batch]
        results = asyncio.run(make_metric(llm).abatch_score(chunk))
        values.extend(float(getattr(r, "value", 0.0)) for r in results)
        print(f"  [score] {min(start + batch, len(inputs))}/{len(inputs)}")
    return values


def build_samples(qa_set, pipeline) -> list:
    SingleTurnSample, *_ = _ragas()
    # 预热：BGE/重排器是惰性加载，先跑通一条把模型初始化掉，避免多线程首次使用竞争
    _ = pipeline.ask(qa_set[0]["question"])

    def build_one(item):
        i, qa = item
        result = pipeline.ask(qa["question"])
        # 上下文用完整 chunk（引用里的 snippet 只有前 80 字，不足以支撑 Faithfulness 判定）
        hits = pipeline.retriever.retrieve(qa["question"], top_k=5)
        contexts = [d.page_content for d, _ in hits]
        if i % 20 == 0:
            print(f"  [build] {i}/{len(qa_set)}...")
        return SingleTurnSample(
            user_input=qa["question"],
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=qa["answer"],
        )

    # 100 条串行 ask 要十几分钟；ask 的瓶颈在 LLM/网络 IO，8 线程并发压到 ~1 分钟
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(build_one, enumerate(qa_set, 1)))


def select_sample(qa_set: list[dict], n: int, mode: str) -> list[dict]:
    """从评测集抽取 n 条（纯函数，便于离线测试）。

    head：取前 n 条。历史报告（Faithfulness 94.1%）用的就是这种抽法，而 QA 集按
    `data/docs` 的文件顺序生成，前 100 条全部来自《劳动合同法》，只覆盖 24 个
    gold chunk —— 样本对语料的代表性很窄。

    stratified：按来源文档轮流取，尽量让每个文档、更多独立 chunk 进入样本。
    这是推荐口径；两种抽法都只能减少偏差，不能消除「问题由 chunk 文本生成」这一
    结构性泄漏，因此报告里必须同时写明抽样方式。
    """
    if mode == "head":
        return list(qa_set[:n])
    if mode != "stratified":
        raise ValueError(f"未知抽样方式：{mode}")
    buckets: dict[str, list[dict]] = {}
    for item in qa_set:
        buckets.setdefault(item.get("source", ""), []).append(item)
    ordered = [buckets[key] for key in sorted(buckets)]
    picked: list[dict] = []
    depth = 0
    while len(picked) < n and any(depth < len(b) for b in ordered):
        for bucket in ordered:
            if depth < len(bucket) and len(picked) < n:
                picked.append(bucket[depth])
        depth += 1
    return picked


def composition(sample: list[dict]) -> dict:
    return {
        "rows": len(sample),
        "unique_gold_chunks": len({item["chunk_id"] for item in sample}),
        "sources": dict(Counter(item.get("source", "") for item in sample).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="全量评测（成本较高）")
    parser.add_argument(
        "--sample",
        choices=["head", "stratified"],
        default="head",
        help="抽样方式：head（历史口径，前 n 条）/ stratified（按文档分层，推荐）",
    )
    args = parser.parse_args()

    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if not args.full:
        qa_set = select_sample(qa_set, 100, args.sample)
        print(f"[ragas] 抽样 {len(qa_set)} 条（--sample {args.sample}，--full 跑全量）")
    sample_info = composition(qa_set)
    print(f"[ragas] 样本组成：{sample_info}")

    pipeline = RAGPipeline()
    samples = build_samples(qa_set, pipeline)

    # ragas 0.4.x：LLM 用 InstructorLLM（OpenAI 兼容客户端指向 DeepSeek），Embedding 用本地 BGE 适配；
    # client/llm 已挪进 score() 按小批次新建，避免 AsyncOpenAI 跨事件循环复用
    _, _, _, AnswerRelevancy, Faithfulness, _AsyncOpenAI = _ragas()
    embeddings = make_ragas_embeddings()
    print("[ragas] 打分中（Faithfulness + AnswerRelevancy）...")
    faithfulness = score(
        lambda llm: Faithfulness(llm=llm), samples,
        ["user_input", "response", "retrieved_contexts"],
    )
    relevancy = score(
        lambda llm: AnswerRelevancy(llm=llm, embeddings=embeddings), samples,
        ["user_input", "response"],
    )

    n = len(qa_set)
    mean_f = round(sum(faithfulness) / n, 4)
    mean_r = round(sum(relevancy) / n, 4)
    summary = {
        "n": n,
        "faithfulness": mean_f,
        "answer_relevancy": mean_r,
        "hallucination_rate": round(1 - mean_f, 4),
        # 口径元数据：分数离开它们无法解释，评审模型与被评模型同族（DeepSeek），
        # 存在自评偏差；样本组成决定这个分数代表哪部分语料。
        "judge_model": settings.deepseek_model,
        "generator_model": settings.deepseek_model,
        "embedding_model": settings.embed_model,
        "sampling": "full" if args.full else args.sample,
        "sample_composition": sample_info,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("\n===== RAGAs 结果 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_path = ROOT / "data" / "qa_set" / "ragas_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "faithfulness", "answer_relevancy"])
        for qa, fv, rv in zip(qa_set, faithfulness, relevancy):
            w.writerow([qa["question"], qa["answer"], round(fv, 4), round(rv, 4)])
    print(f"[ragas] 报告已存：{OUT} / {detail_path}")


if __name__ == "__main__":
    main()
