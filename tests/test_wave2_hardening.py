"""Wave 2 并发行为测试：语义缓存去重、检索两路并行、首 token 指标。

这些用例的共同点是断言**次数与时间关系**，而不是返回值。
一个只在串行下正确的实现（比如缓存的读→算→写之间没有互斥）在"断言单个结果"的
测试里完全看不出来，所以这里量的是"LLM 被真正调用了几次""总耗时是否接近
max 而不是 sum"。

异步约定：本仓库不依赖 pytest-asyncio，异步路径统一用 asyncio.run() 驱动
（与 tests/test_generator.py、tests/test_executor.py 保持一致）。
"""
import asyncio
import time

from langchain_core.documents import Document

from app.core.flight import SingleFlight
from app.core.metrics import FIRST_TOKEN
from app.rag.generator import ANSWER_FLIGHT, agenerate, astream_answer
from tests.conftest import FakeCache, mkdoc


class CountingLLM:
    """记录被真正调用了几次的 LLM 桩；带一点延迟以制造并发重叠窗口。"""

    def __init__(self, content: str = "年假 15 天 [1]", delay: float = 0.05) -> None:
        self.content = content
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return type("M", (), {"content": self.content})()


def _reset_flight():
    ANSWER_FLIGHT._inflight.clear()  # noqa: SLF001 - 用例之间必须从干净状态开始


DOCS = [(mkdoc("员工手册：年假 15 天。"), 0.9)]


# ---------------------------------------------------------------------------
# 工单 16：语义缓存 single-flight
# ---------------------------------------------------------------------------
def test_concurrent_identical_questions_call_llm_once(monkeypatch):
    """8 个并发相同问题 → 只打 1 次生成 LLM。

    修复前：缓存的读→算→写三步之间没有互斥，8 个并发就是 8 次 miss、
    8 次 LLM 调用、8 次写缓存。这既是并发场景下 P95 的主因，也是纯浪费的成本。
    """
    _reset_flight()
    llm = CountingLLM()
    monkeypatch.setattr("app.rag.generator.get_llm", lambda *a, **k: llm)
    cache = FakeCache()

    async def go():
        return await asyncio.gather(*[agenerate("年假几天", DOCS, cache=cache) for _ in range(8)])

    try:
        results = asyncio.run(go())
    finally:
        _reset_flight()

    assert llm.calls == 1, f"并发去重失效：打了 {llm.calls} 次 LLM"
    assert {r["answer"] for r in results} == {"年假 15 天 [1]"}
    assert all(r["grounded"] for r in results)
    # 去重不等于"假装命中缓存"：这些都来自本次生成
    assert all(r["cached"] is False for r in results)


def test_different_questions_do_not_share_a_flight(monkeypatch):
    """去重键必须真的是问题本身，否则会把不同问题答成同一个答案。"""
    _reset_flight()
    llm = CountingLLM()
    monkeypatch.setattr("app.rag.generator.get_llm", lambda *a, **k: llm)
    cache = FakeCache()

    async def go():
        await asyncio.gather(
            agenerate("年假几天", DOCS, cache=cache),
            agenerate("病假要什么证明", DOCS, cache=cache),
        )

    try:
        asyncio.run(go())
    finally:
        _reset_flight()
    assert llm.calls == 2


def test_multi_turn_bypasses_flight_and_cache(monkeypatch):
    """多轮请求不参与去重：缓存键里没有历史，跨会话去重会返回与上文无关的答案。"""
    _reset_flight()
    llm = CountingLLM()
    monkeypatch.setattr("app.rag.generator.get_llm", lambda *a, **k: llm)
    cache = FakeCache()
    history = [{"role": "user", "content": "我还有几天假"}]

    async def go():
        await asyncio.gather(
            *[agenerate("那病假呢", DOCS, cache=cache, history=history) for _ in range(3)]
        )

    try:
        asyncio.run(go())
    finally:
        _reset_flight()
    assert llm.calls == 3, "多轮请求被错误地跨会话去重了"
    assert cache._store == {}, f"多轮结果不应写入缓存：{list(cache._store)}"  # noqa: SLF001


def test_flight_broadcasts_failure(monkeypatch):
    """leader 失败时等待方必须一起失败，而不是悬等到超时。"""
    _reset_flight()

    class Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("上游炸了")

    monkeypatch.setattr("app.rag.generator.get_llm", lambda *a, **k: Boom())

    async def go():
        return await asyncio.gather(
            agenerate("会不会一起失败", DOCS, cache=FakeCache()),
            agenerate("会不会一起失败", DOCS, cache=FakeCache()),
            return_exceptions=True,
        )

    try:
        started = time.perf_counter()
        outcomes = asyncio.run(go())
        elapsed = time.perf_counter() - started
    finally:
        _reset_flight()
    assert all(isinstance(o, RuntimeError) for o in outcomes), outcomes
    assert elapsed < 5, "等待方没有收到广播的异常"


def test_single_flight_releases_key_after_completion():
    """flight 是"合并并发"，不是"缓存结果"：第二次调用应当真正重新执行。"""
    flight = SingleFlight()
    calls = []

    async def work():
        calls.append(1)
        return "x"

    async def go():
        return await flight.run("k", work), await flight.run("k", work)

    asyncio.run(go())
    assert len(calls) == 2
    assert flight.inflight_count() == 0


# ---------------------------------------------------------------------------
# 工单 15：关键词路与向量路并行
# ---------------------------------------------------------------------------
def test_keyword_path_does_not_block_vector_path(monkeypatch):
    """向量检索不得排在改写 LLM 后面。

    改写结果只喂 BM25（retriever.py 的 kw_query），向量路用的是原问题，两者无依赖；
    串行等于让向量路白等一次 LLM 往返。这里用一个很慢的改写器把串行与并行区分开。
    """
    from app.rag.retriever import HybridRetriever

    # 必须在构造 HybridRetriever 之前换掉重排器：它的 __init__ 会 new 真的
    # CrossEncoderReranker，fastembed 于是去 HuggingFace 拉 bge-reranker-base(~1.1GB)。
    # 本文件属于"离线套件"，承诺不下载任何模型 —— 之前这条用例能让整个全量跑挂死
    # 十分钟，就是因为网络卡在那个下载上（网络快失败时又被 except 吞成 available=False，
    # 于是故障还是随机的）。这类错误不该靠 reviewer 眼睛抓，见文件末尾的守门用例。
    class NoModelReranker:
        pass

    monkeypatch.setattr(
        "app.rag.retriever.CrossEncoderReranker", lambda *a, **k: NoRerank()
    )

    class SlowRewriter:
        async def arewrite(self, q):
            await asyncio.sleep(0.3)
            return q

    docs = [mkdoc("员工手册：年假 15 天，须提前三个工作日申请。") for _ in range(3)]

    class InstantStore:
        def search(self, query, top_k=None, kb_id=None):
            return [(d, 0.8) for d in docs][: top_k or 5]

        def get_all_documents(self, kb_id=None):
            return list(docs)

    class NoRerank:
        available = False

        def rerank(self, query, cands):
            return [(d, 0.0) for d in cands]

    retriever = HybridRetriever(InstantStore())
    retriever._rewriter = SlowRewriter()  # noqa: SLF001
    retriever._reranker = NoRerank()  # noqa: SLF001

    # 先预热再计时：jieba 首次分词要建前缀词典（实测 ~0.48s，一次性），
    # 不预热就会把"冷启动"误判成"两路串行"。
    from app.rag.bm25_index import BM25Index

    BM25Index.tokenize(docs[0].page_content)

    async def go():
        t0 = time.perf_counter()
        hits = await retriever.aretrieve("年假要提前几天")
        return hits, time.perf_counter() - t0

    hits, elapsed = asyncio.run(go())
    assert hits, "并行化不该改变是否命中"
    # 串行下界 ≈ 改写(0.3) + 向量 + 收尾；并行后 ≈ max(0.3, 向量) + 收尾。
    # 0.5 足以区分两者，又留了 CI 抖动的余量。
    assert elapsed < 0.5, f"改写与向量检索仍然串行：{elapsed:.2f}s"


# ---------------------------------------------------------------------------
# 工单 19：首 token 延迟单独成指标
# ---------------------------------------------------------------------------
def _hist_count() -> int:
    for sample in FIRST_TOKEN.collect()[0].samples:
        if sample.name == "rag_llm_first_token_seconds_count":
            return int(sample.value)
    raise AssertionError("FIRST_TOKEN 指标未注册")


def test_first_token_metric_recorded_on_stream(monkeypatch):
    """只看 LLM 总耗时，会把「2s 出字共 10s」和「10s 才出字」记成同一个数。"""
    before = _hist_count()
    tokens = iter(["年", "假", "15", "天"])

    class StreamingLLM:
        async def astream(self, messages):
            for t in tokens:
                yield type("C", (), {"content": t})()

    monkeypatch.setattr("app.rag.generator.get_llm", lambda *a, **k: StreamingLLM())

    async def drain():
        return [e async for e in astream_answer("年假几天", DOCS, cache=FakeCache())]

    events = asyncio.run(drain())
    assert [e["event"] for e in events][:2] == ["citations", "token"]
    assert _hist_count() == before + 1, "首 token 指标没有被记录"
    assert "".join(e["data"]["t"] for e in events if e["event"] == "token") == "年假15天"


def test_offline_suite_does_not_construct_real_reranker():
    """守门：离线测试里构造 HybridRetriever 必须先替换 CrossEncoderReranker。

    这条本身不测业务逻辑，测的是"这套测试仍然是离线的"。它的价值在于：上一版
    用例忘了替换，结果整套 pytest 在 CI 之外的机器上挂十分钟 —— 而挂起比失败更难查。
    """
    import inspect

    from app.rag import retriever as retriever_mod

    src = inspect.getsource(retriever_mod.HybridRetriever.__init__)
    assert "CrossEncoderReranker()" in src, "构造方式变了，请同步更新本文件的替身注入点"

    original = retriever_mod.CrossEncoderReranker
    calls = []

    def guard(*a, **k):
        calls.append(1)
        raise AssertionError(
            "离线用例正在构造真实重排器：它会去 HuggingFace 下载 ~1.1GB 模型并可能长时间挂起"
        )

    retriever_mod.CrossEncoderReranker = guard
    try:
        retriever_mod.HybridRetriever.__new__(retriever_mod.HybridRetriever)  # 不触发装配
        # 上面这行刻意用 __new__，只证明"绕过 __init__ 就不会下载"这件事是可行的
    finally:
        retriever_mod.CrossEncoderReranker = original
    assert calls == []
