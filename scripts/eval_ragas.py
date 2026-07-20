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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 限流：本地 ONNX 推理（BGE/重排器）只使用 2 个 CPU 线程，评测期间不影响日常使用
os.environ["OMP_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.rag.pipeline import RAGPipeline

QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
OUT = ROOT / "data" / "qa_set" / "ragas_report.json"


class BGEEmbeddingsRagas(BaseRagasEmbedding):
    """本地 BGE 的 ragas 现代 embedding 接口适配。"""

    def __init__(self) -> None:
        self._emb = BGEEmbeddings()

    def embed_text(self, text: str) -> list[float]:
        return self._emb.embed_query(text)

    async def aembed_text(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._emb.embed_query, text)


def score(make_metric, samples: list[SingleTurnSample], keys: list[str], batch: int = 20) -> list[float]:
    """ragas 0.4.3 的 collections 指标不兼容 evaluate()，直接调用其批量打分接口。

    注意：abatch_score 会把字典的每个键都作为关键字参数传给 ascore()，
    因此必须按指标的参数签名裁剪输入字段（Faithfulness 不需要 reference 等字段）；
    指标内部一律走异步调用，客户端需为 AsyncOpenAI。
    按 batch 分批 gather：把 DeepSeek 并发压在每批 20 路以内，避免 429 拖慢评测；
    每个小批次新建 client/llm，避免 AsyncOpenAI 跨事件循环复用报错。
    """
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


def build_samples(qa_set, pipeline) -> list[SingleTurnSample]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="全量评测（成本较高）")
    args = parser.parse_args()

    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    if not args.full:
        qa_set = qa_set[:100]
        print(f"[ragas] 抽样 {len(qa_set)} 条（--full 跑全量）")

    pipeline = RAGPipeline()
    samples = build_samples(qa_set, pipeline)

    # ragas 0.4.x：LLM 用 InstructorLLM（OpenAI 兼容客户端指向 DeepSeek），Embedding 用本地 BGE 适配；
    # client/llm 已挪进 score() 按小批次新建，避免 AsyncOpenAI 跨事件循环复用
    embeddings = BGEEmbeddingsRagas()
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
