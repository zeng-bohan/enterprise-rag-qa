"""BGE 本地向量化封装（fastembed / ONNX Runtime，无需 torch、无需 GPU）。

实现 LangChain Embeddings 的鸭子类型接口（embed_documents / embed_query），
因此可直接传给 LangChain 的 VectorStore。

选型说明：
- BGE（智源）是中文检索的主流开源 embedding 系列；
- fastembed 用 ONNX Runtime 跑 bge-small-zh-v1.5，CPU 单条查询约 20-50ms，
  零外部 API 依赖、数据不出本机；
- 压测阶段如需提速，可换 sentence-transformers + GPU（RTX 4060）。

缓存（v0.3）：按文本内容哈希缓存向量（Redis），全量重建索引时相同 chunk
免重复计算；Redis 不可用时自动直算，不阻塞链路。
"""
from typing import List

from fastembed import TextEmbedding

from app.config import settings
from app.core.cache import Cache, get_cache
from app.core.metrics import CACHE_HITS, CACHE_MISSES

EMB_TTL = 7 * 24 * 3600  # Embedding 结果稳定，缓存 7 天


class BGEEmbeddings:
    def __init__(self) -> None:
        # fastembed 首次调用会自动下载模型到本地缓存（~95MB）；
        # threads 显式限制单条推理线程数，避免并发时吃满全核互相阻塞（压测结论）
        self._model = TextEmbedding(model_name=settings.embed_model, threads=settings.onnx_threads)
        self._cache = get_cache()

    def _lookup(self, key: str):
        hit = self._cache.get_json(key)
        if not hit:
            CACHE_MISSES.labels(kind="embed").inc()
            return None
        CACHE_HITS.labels(kind="embed").inc()
        return hit["v"] if hit else None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing = []
        for i, t in enumerate(texts):
            key = Cache.key("emb", t)
            vec = self._lookup(key)
            if vec is not None:
                out[i] = vec
            else:
                missing.append((i, t))
        if missing:
            for vec, (i, t) in zip(
                self._model.embed([t for _, t in missing]), missing
            ):
                v = vec.tolist()
                out[i] = v
                self._cache.set_json(Cache.key("emb", t), {"v": v}, ttl=EMB_TTL)
        return out

    def embed_query(self, text: str) -> List[float]:
        # query_embed 返回 Iterable[ndarray]，取第一个元素避免与 LangChain 的包裹层重复嵌套
        key = Cache.key("embq", text)
        vec = self._lookup(key)
        if vec is not None:
            return vec
        v = next(iter(self._model.query_embed(text))).tolist()
        self._cache.set_json(key, {"v": v}, ttl=EMB_TTL)
        return v

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]
