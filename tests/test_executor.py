"""CPU 专用线程池：结果透传与线程隔离。"""
import asyncio

from app.core.executor import run_cpu


def test_run_cpu_returns_function_result():
    def add(a, b, *, c=0):
        return a + b + c

    assert asyncio.run(run_cpu(add, 1, 2, c=3)) == 6


def test_run_cpu_executes_on_dedicated_pool_thread():
    import threading

    seen = {}

    def where():
        seen["thread"] = threading.current_thread().name
        return True

    assert asyncio.run(run_cpu(where)) is True
    assert seen["thread"].startswith("rag-cpu")
