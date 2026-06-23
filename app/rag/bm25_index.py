"""BM25 关键词检索索引。

为什么需要 BM25：
纯向量检索对「专有名词 / 编号 / 精确关键词」类问题召回弱——语义空间把字面匹配
稀释了；BM25 恰好补强字面匹配（TF-IDF 家族，含文档长度归一化），两者用 RRF 融合
取长补短，是工业界 RAG 混合检索的标准套路。

实现要点：
- 中文无天然词边界，用 jieba 精确模式分词；
- rank_bm25.BM25Okapi 默认参数（k1=1.5, b=0.75）对中文制度文档效果稳定；
- 索引在进程启动时从向量库全量构建，入库脚本执行后需重启服务才会刷新
  （v0.3 换 PGvector 时顺带支持增量更新）。
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
