"""CPU 密集任务专用线程池（按用途分池）。

背景（v0.5 压测结论）：ONNX 推理（BGE 向量化 / Cross-Encoder 重排）与 BM25 分词是
纯 CPU 计算，没有网络等待可挂起，只能交给线程池。若与 FastAPI 默认线程池共享且
不加并发上限，并发请求会让线程超卖、互相抢核——实测 8 并发缓存命中请求 P50 从
~1s 恶化到 ~20s（详见 docs/DESIGN.md）。

为什么是"分池"而不是"一个有界池"（工单 14）
------------------------------------------
单一有界池解决了抢核，却引入了另一个故障模式：**互相等**。原先这个池同时承接
在线检索的推理、注册中心的元数据 CRUD、以及摄取任务的批量 embed。后果是：

- 一次大文档摄取要在池里占满多个批次，在线请求排在它后面 —— 一个上传就能
  把全站查询的 P95 拖爆（head-of-line blocking）；
- 元数据查询（SQLite/PG 的小 SELECT）本身微秒级，却可能在一个 1.1GB 模型的
  重排任务后面排队；
- 工单 15 之后，单个检索请求会**并发**投两个任务进查询池（关键词路与向量路），
  共享池下这会让"能同时处理几个请求"直接减半。

现在拆成两个池，配额独立、互不排队：

- `_query`：推理与检索计算（含 rerank / BM25 / 向量查询）。上限按物理核配比定，
  见下面的实测注释。
- `_meta`：注册中心与元数据读写。这类调用不等推理，也不该被推理挡住。

摄取任务不在这里 —— 它的正确落点是独立 worker 进程（工单 12），因为任何进程内的
线程配额都挡不住"一个任务占满池"这种模式。
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
# 元数据操作是短平快的读写，不占 CPU，配额给得宽一点也不会抢核
META_WORKERS = int(os.environ.get("RAG_META_WORKERS", "4"))
# 摄取（解析→切片→批量 embed）单独一池，且刻意给得很小。
# 这是工单 12（arq 独立 worker 进程）之前的过渡措施：它不能解决"一个大文件要等多久"，
# 但能阻断原来那个更糟的模式 —— 一次上传把在线检索全部排死。
INGEST_WORKERS = int(os.environ.get("RAG_INGEST_WORKERS", "2"))

_query_executor = ThreadPoolExecutor(max_workers=CPU_WORKERS, thread_name_prefix="rag-cpu")
_meta_executor = ThreadPoolExecutor(max_workers=META_WORKERS, thread_name_prefix="rag-meta")
_ingest_executor = ThreadPoolExecutor(max_workers=INGEST_WORKERS, thread_name_prefix="rag-ingest")


async def _submit(executor: ThreadPoolExecutor, fn, args, kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


async def run_cpu(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把推理/检索这类 CPU 密集函数丢进查询池并 await（不阻塞事件循环）。"""
    return await _submit(_query_executor, fn, args, kwargs)


async def run_meta(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把注册中心 / 元数据读写丢进元数据池：不许排在推理任务后面。"""
    return await _submit(_meta_executor, fn, args, kwargs)


async def run_ingest(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把文档摄取丢进摄取池：一个 200 页 PDF 不该让在线查询排队。"""
    return await _submit(_ingest_executor, fn, args, kwargs)
