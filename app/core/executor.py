"""CPU 密集任务专用线程池。

背景（v0.5 压测结论）：ONNX 推理（BGE 向量化 / Cross-Encoder 重排）与 BM25 分词是
纯 CPU 计算，没有网络等待可挂起，只能交给线程池。若与 FastAPI 默认线程池共享且
不加并发上限，并发请求会让线程超卖、互相抢核——实测 8 并发缓存命中请求 P50 从
~1s 恶化到 ~20s（详见 docs/DESIGN.md）。

方案：CPU 密集任务走独立的、有界（max_workers）的线程池，超出部分排队，
换取稳定的单请求延迟与可控的总吞吐。

用法（async 函数内）：
    hits = await run_cpu(self._bm25.search, kw_query, RECALL_K)
"""
import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# i5-14400F（6P+4E，16 逻辑核）+ ONNX_THREADS=4：单条推理约 4 线程，
# 3 个并发即 12 线程，实测该配比单请求延迟与吞吐综合最佳（扫描表见 docs/DESIGN.md）；
# 可用环境变量 RAG_CPU_WORKERS 覆盖（压测调参用）。
CPU_WORKERS = int(os.environ.get("RAG_CPU_WORKERS", "3"))

_executor = ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix="rag-cpu")


async def run_cpu(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把同步的 CPU 密集函数丢进专用线程池，await 其结果（不阻塞事件循环）。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, functools.partial(fn, *args, **kwargs))
