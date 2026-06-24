"""检索层 v2：查询改写 → 双路召回（BM25 + 向量）→ RRF 融合 → Cross-Encoder 重排。

拒答判定（两条路径）：
1. 前置拒答：BM25 无命中 且 向量最高相关度低于阈值 —— 直接返回空，不调用生成 LLM；
2. 重排拒答：融合重排后所有候选的 Cross-Encoder 分数都低于 0.5（sigmoid 归一的
   「相关/无关」分界）—— 同样返回空触发拒答话术。

演进记录：
- v1 纯向量检索：对关键词类问题排序弱（如「年假提前几天」正确 chunk 排第 2）；
- v2 引入 BM25 混合 + 重排：字面匹配与语义匹配互补，实测排序修正（见 git 历史对比）。
"""
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.core.executor import run_cpu
from app.rag.bm25_index import BM25Index
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import CrossEncoderReranker
from app.rag.vector_store import ChromaStore, PGvectorStore

RRF_K = 60        # RRF 常数：对排名做对数级折扣，避免单路高分垄断
RECALL_K = 20     # 单路召回数量（召回宁多勿漏，靠重排提精度）
RERANK_K = 8      # 参与 Cross-Encoder 重排的候选数（重排慢，控制预算）
RERANK_FLOOR = 0.5  # sigmoid 归一后「相关 / 无关」分界


def rrf_fuse(
    bm25_hits: List[Tuple[Document, float]],
    vec_hits: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion：按两路排名融合，输出 (doc, 融合分) 降序。"""
    fused: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}
    for hits in (bm25_hits, vec_hits):
        for rank, (doc, _) in enumerate(hits, start=1):
            key = doc.id or doc.page_content
            doc_by_id.setdefault(key, doc)
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return [(doc_by_id[key], score) for key, score in ranked]


class HybridRetriever:
    def __init__(self, store: "ChromaStore | PGvectorStore") -> None:
        self._store = store
        self._bm25 = BM25Index(store.get_all_documents())
        self._rewriter = QueryRewriter()
        self._reranker = CrossEncoderReranker()

    def _finalize(
        self,
        query: str,
        bm25_hits: List[Tuple[Document, float]],
        vec_hits: List[Tuple[Document, float]],
        k: int,
        vec_floor: float,
    ) -> List[Tuple[Document, float]]:
        """双路召回之后的公共后处理：前置拒答 → RRF 融合 → 重排 → 重排拒答 + 截断。"""
        # 2) 前置拒答：字面与语义都没有相关命中
        if not bm25_hits and (not vec_hits or vec_hits[0][1] < vec_floor):
            return []

        # 3) RRF 融合 → 4) Cross-Encoder 重排
        fused = rrf_fuse(bm25_hits, vec_hits)
        candidates = [doc for doc, _ in fused[:RERANK_K]]
        reranked = self._reranker.rerank(query, candidates)

        # 5) 重排拒答 + 截断
        result = [(doc, score) for doc, score in reranked if score >= RERANK_FLOOR]
        return result[:k]

    def retrieve(
        self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        k = top_k or settings.retrieval_top_k
        vec_floor = threshold if threshold is not None else settings.score_threshold

        # 1) 查询改写：原问题喂向量检索（语义），改写结果喂 BM25（字面）
        kw_query = self._rewriter.rewrite(query)
        bm25_hits = self._bm25.search(kw_query, RECALL_K)
        vec_hits = self._store.search(query, RECALL_K)
        return self._finalize(query, bm25_hits, vec_hits, k, vec_floor)

    async def aretrieve(
        self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """异步版 retrieve（v0.5）：改写 LLM 用 await；BM25/向量/重排是 CPU 计算，
        丢进专用有界线程池（app.core.executor），避免并发时线程超卖抢核。"""
        k = top_k or settings.retrieval_top_k
        vec_floor = threshold if threshold is not None else settings.score_threshold

        kw_query = await self._rewriter.arewrite(query)
        bm25_hits = await run_cpu(self._bm25.search, kw_query, RECALL_K)
        vec_hits = await run_cpu(self._store.search, query, RECALL_K)
        return await run_cpu(self._finalize, query, bm25_hits, vec_hits, k, vec_floor)
