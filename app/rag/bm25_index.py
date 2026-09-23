"""BM25 关键词检索索引。

为什么需要 BM25：
纯向量检索对「专有名词 / 编号 / 精确关键词」类问题召回弱——语义空间把字面匹配
稀释了；BM25 恰好补强字面匹配（TF-IDF 家族，含文档长度归一化），两者用 RRF 融合
取长补短，是工业界 RAG 混合检索的标准套路。

实现要点：
- 中文无天然词边界，用 jieba 精确模式分词；
- rank_bm25.BM25Okapi 默认参数（k1=1.5, b=0.75）对中文制度文档效果稳定；
- 索引由 HybridRetriever 按知识库惰性构建，文档增删后 invalidate(kb_id)，
  下次查询时全量重建（不是启动时一次性构建，也没有增量更新）。

已知规模瓶颈（工单 11 处理，这里先把话说明白，避免注释和实现继续不符）：
- 每次 invalidate 之后，第一个请求要在在线线程池里把该 KB 全部 chunk 拉回内存、
  逐条 jieba 分词、重建 BM25Okapi —— 十万 chunk 级 KB 上就是秒级到十秒级的 P99 尖刺；
- BM25Okapi.get_scores 是纯 Python 实现，每次查询线性遍历整个语料，本身即 O(N)。
落点是 PG 侧真倒排（tsvector + GIN，分词结果入库），届时「增量更新」才真正成立。
"""
from typing import List, Optional, Tuple

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, docs: Optional[List[Document]] = None) -> None:
        self._docs: List[Document] = []
        self._index: Optional[BM25Okapi] = None
        if docs:
            self.build(docs)

    def build(self, docs: List[Document]) -> None:
        self._docs = list(docs)
        tokenized = [self.tokenize(d.page_content) for d in self._docs]
        self._index = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t.strip() for t in jieba.lcut(text) if t.strip()]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """返回按 BM25 分数降序的 (doc, score)，分数为 0 的命中不返回。"""
        if not self._index:
            return []
        scores = self._index.get_scores(self.tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(self._docs[i], float(s)) for i, s in ranked[:top_k] if s > 0]
