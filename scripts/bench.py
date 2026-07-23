# -*- coding: utf-8 -*-
"""自研 HTTP 压测脚本。

并发模型：起 N 个 asyncio 协程（N = 并发数），每个协程串行打自己分到的请求，
即任意时刻恒有 ≤ N 路请求在飞。统计全部请求的客户端耗时 → QPS + P50/P95/P99。

用法：
  .venv/Scripts/python scripts/bench.py -c 8 -n 100

数据源：data/qa_set/qa_300.json 的前 n 条。每个问题只打一次——
问答有语义缓存，同一问题重复打会命中缓存跳过 LLM，把延迟测虚。
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "data" / "qa_set" / "qa_300.json"
BASE_URL = "http://127.0.0.1:8000"


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


async def worker(client: httpx.AsyncClient, questions: list[str],
                 client_ms: list[float], server_ms: list[float]) -> None:
    for q in questions:
        t0 = time.perf_counter()
        r = await client.post("/v1/chat", json={"question": q})
        r.raise_for_status()
        client_ms.append(time.perf_counter() - t0)
        server_ms.append(r.json().get("latency_ms", 0))


async def run(concurrency: int, total: int, offset: int = 0) -> None:
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))[offset:offset + total]
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        # 预热：触发服务端 BGE/重排器的惰性模型加载，不纳入统计
        await client.post("/v1/chat", json={"question": qa[0]["question"]})

        client_ms, server_ms = [], []
        chunks = [qa[i::concurrency] for i in range(concurrency)]
        t0 = time.perf_counter()
        await asyncio.gather(*(
            worker(client, [q["question"] for q in chunk], client_ms, server_ms)
            for chunk in chunks
        ))
        wall = time.perf_counter() - t0

    n = len(client_ms)
    print(f"并发 {concurrency} × {n} 请求（每个问题仅 1 次，语义缓存不生效）")
    print(f"  QPS    : {n / wall:.1f}（wall {wall:.1f}s）")
    print(f"  客户端延迟: P50 {pct(client_ms, .5) * 1000:6.0f} ms  "
          f"P95 {pct(client_ms, .95) * 1000:6.0f} ms  "
          f"P99 {pct(client_ms, .99) * 1000:6.0f} ms")
    print(f"  服务端耗时: 均值 {sum(server_ms) / n:6.0f} ms  "
          f"P95 {pct(server_ms, .95):6.0f} ms（响应体 latency_ms）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 服务压测")
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="并发协程数")
    parser.add_argument("-n", "--total", type=int, default=100, help="总请求数（= 使用的问题条数）")
    parser.add_argument("--offset", type=int, default=0, help="QA 集起始下标（避开已缓存的问题）")
    args = parser.parse_args()
    asyncio.run(run(args.concurrency, args.total, args.offset))
