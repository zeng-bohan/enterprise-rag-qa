"""Prometheus 指标（v0.5）：压测与运行期可观测性。

指标清单（均带标签，便于按维度聚合）：
- rag_requests_total{status}            请求计数（ok / error）
- rag_request_latency_seconds{stage}    请求耗时直方图（total / retrieval）
- rag_inflight_requests                 在途请求数（Gauge）
- rag_cache_hits_total{kind}            缓存命中计数（kind = rewrite / answer / embed）
- rag_cache_misses_total{kind}          缓存未命中计数（同上）
- rag_llm_calls_total                   LLM 调用次数（改写 + 生成）
- rag_llm_latency_seconds               LLM 调用耗时直方图

暴露方式：app/main.py 的 GET /metrics（prometheus_client 文本格式，
可直接被 Prometheus 抓取、Grafana 展示）。
"""
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Gauge, Histogram

# 延迟直方图分桶（秒）：覆盖缓存命中（~1s）到 LLM 慢响应（30s+）的区间
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)

REQUESTS = Counter("rag_requests_total", "HTTP 请求计数", ["status"])
LATENCY = Histogram("rag_request_latency_seconds", "请求耗时", ["stage"], buckets=_LATENCY_BUCKETS)
INFLIGHT = Gauge("rag_inflight_requests", "在途请求数")
CACHE_HITS = Counter("rag_cache_hits_total", "缓存命中计数", ["kind"])
CACHE_MISSES = Counter("rag_cache_misses_total", "缓存未命中计数", ["kind"])
LLM_CALLS = Counter("rag_llm_calls_total", "LLM 调用次数（改写 + 生成）")
LLM_LATENCY = Histogram("rag_llm_latency_seconds", "LLM 调用耗时", buckets=_LATENCY_BUCKETS)
# 流式问答真正决定体感的不是总耗时，而是"多久看到第一个字"。只看 LLM_LATENCY
# 会把「2s 出首 token、共 10s」和「10s 才出首 token」记成同一个数。
FIRST_TOKEN = Histogram(
    "rag_llm_first_token_seconds",
    "流式生成：请求到首个 token 的耗时",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0),
)
RATE_LIMITED = Counter("rag_rate_limited_total", "被限流拒绝的请求数")


def llm_start() -> float:
    """记录一次 LLM 调用开始，返回起点时间；配合 llm_done 记录耗时。"""
    LLM_CALLS.inc()
    return time.perf_counter()


def llm_done(t0: float) -> None:
    LLM_LATENCY.observe(time.perf_counter() - t0)


@contextmanager
def llm_timer() -> Iterator[None]:
    """一次 LLM 调用的计数 + 耗时，同步 / 异步两条路径共用同一个口径。

    用 try/finally 而非顺序调用：LLM 抛错（超时 / 限流）时耗时仍要落进直方图，
    否则面板上看到的是「LLM 调用次数变少」而不是「LLM 在失败」。
    """
    t0 = llm_start()
    try:
        yield
    finally:
        llm_done(t0)


@contextmanager
def counted() -> Iterator[None]:
    """端点级请求计数：正常结束记 ok，异常记 error 后原样抛出。

    收敛前先补一句历史：原本只有 /v1/chat 计数，/v1/retrieval-test 完全不进
    rag_requests_total，两个端点的 QPS 在同一块面板上不可比。
    """
    try:
        yield
    except Exception:
        REQUESTS.labels(status="error").inc()
        raise
    REQUESTS.labels(status="ok").inc()


def cache_hit(kind: str) -> None:
    CACHE_HITS.labels(kind=kind).inc()


def cache_miss(kind: str) -> None:
    CACHE_MISSES.labels(kind=kind).inc()
