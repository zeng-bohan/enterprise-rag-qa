"""全局测试夹具：整套测试离线运行——不连 PostgreSQL / Redis，不下载模型，不出网络。

约定：
- LLM / Cross-Encoder / 向量库 / 缓存一律用桩替换，测的是分层逻辑与编排，不是模型；
- 需要外部资源的构造路径（如 app.api.chat 的模块级 RAGPipeline 单例）在 import 前
  统一打桩，端点行为由用例自行注入 fake。
"""
from typing import Dict, List, Optional, Tuple
from unittest import mock

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# app.api.chat 在 import 时构造真实 RAGPipeline（连 PG / 加载模型），
# 测试进程统一替换为惰性桩；端点测试用 monkeypatch 注入 aask 行为。
# ---------------------------------------------------------------------------
from app.rag import pipeline as _pipeline_mod

# 打桩前保留真实类引用：个别测试需要构造真 RAGPipeline（__new__ 跳过依赖装配）
RealRAGPipeline = _pipeline_mod.RAGPipeline


class _StubPipeline:
    """不持任何资源的空壳：真实依赖只能由用例显式注入，误用即 AttributeError。"""

    def __init__(self, *args, **kwargs) -> None:
        self.store = None
        self.cache = None
        self.retriever = None


mock.patch.object(_pipeline_mod, "RAGPipeline", _StubPipeline).start()


# ---------------------------------------------------------------------------
# 通用 fake
# ---------------------------------------------------------------------------
class FakeMsg:
    """模拟 langchain 消息：QueryRewriter / generator 只读 .content。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """记录调用次数的最小 LLM 桩：invoke / ainvoke 返回预设内容。"""

    def __init__(self, content: str = "改写结果") -> None:
        self.content = content
        self.sync_calls = 0
        self.async_calls = 0

    def invoke(self, messages) -> FakeMsg:
        self.sync_calls += 1
        return FakeMsg(self.content)

    async def ainvoke(self, messages) -> FakeMsg:
        self.async_calls += 1
        return FakeMsg(self.content)


class FakeCache:
    """与 app.core.cache.Cache 同接口的进程内字典桩（available=True 语义）。"""

    def __init__(self) -> None:
        self._store: Dict[str, dict] = {}

    @staticmethod
    def key(*parts: str) -> str:
        from app.core.cache import Cache

        return Cache.key(*parts)

    @property
    def available(self) -> bool:
        return True

    def get_json(self, key: str) -> Optional[dict]:
        return self._store.get(key)

    def set_json(self, key: str, value: dict, ttl: Optional[int] = None) -> None:
        self._store[key] = value


class FakeReranker:
    """按预设分数返回的重排桩：记录收到的候选数，供截断断言用。"""

    def __init__(self, scores: List[float]) -> None:
        self.scores = scores
        self.seen_candidates = 0

    @property
    def available(self) -> bool:
        return True

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        self.seen_candidates = len(docs)
        scored = list(zip(docs, self.scores))
        scored.sort(key=lambda x: -x[1])
        return scored


class FakeStore:
    """向量库桩：search 固定返回预设命中，get_all_documents 供 BM25 建索引。"""

    def __init__(self, docs: List[Document], hits: Optional[List[Tuple[Document, float]]] = None):
        self._docs = docs
        self._hits = hits or []

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Document]:
        return list(self._docs)

    def search(self, query: str, top_k: Optional[int] = None, kb_id: Optional[str] = None) -> List[Tuple[Document, float]]:
        return list(self._hits)[: top_k or 5]


def mkdoc(text: str, source: str = "员工手册.md") -> Document:
    return Document(page_content=text, metadata={"source": source})
