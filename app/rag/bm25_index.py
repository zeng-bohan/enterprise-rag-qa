"""BM25 关键词检索索引。

为什么需要 BM25：
纯向量检索对「专有名词 / 编号 / 精确关键词」类问题召回弱，语义空间把字面匹配
稀释了；BM25 恰好补强字面匹配（TF-IDF 家族，含文档长度归一化），两者用 RRF 融合
取长补短。

这一版把 rank_bm25 的 BM25Okapi 换成了带倒排表的实现：
- 查询只遍历包含查询词的文档（postings），不再对全部语料逐条打分；
- 支持 add_documents / remove_documents 增量更新，不必每次重建整个索引；
- 分词结果按正文哈希缓存，同一段文本在重建时不会重复做 jieba 分词。

打分公式与 rank_bm25.BM25Okapi 保持一致（k1=1.5、b=0.75，负 IDF 用平均 IDF 的
0.25 倍替代），并用对照测试锁定排序与分数，避免换实现改变检索行为。

已知规模瓶颈（工单 11，这里先把话说明白）：
- HybridRetriever 在知识库版本变化后仍要重新拉取该库全部 chunk 并重建索引；
  跨进程写入（worker）没有增量通知，只能靠版本号触发重建；
- 全量重建的耗时主要来自 jieba 分词，本实现的哈希缓存能消掉未变更 chunk 的
  分词成本，但读取与建表仍是 O(N)；
- 十万 chunk 级知识库的落点仍是数据库侧倒排（PostgreSQL tsvector + GIN，
  分词结果入库），届时增量更新才真正跨进程成立。
"""
import hashlib
import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import jieba
from langchain_core.documents import Document

K1 = 1.5
B = 0.75
EPSILON = 0.25


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=100_000)
def _tokenize_cached(text: str) -> Tuple[str, ...]:
    """按正文缓存分词结果：重建索引时同一段文本不会重复分词。"""
    return tuple(t.strip() for t in jieba.lcut(text) if t.strip())


class BM25Index:
    def __init__(self, docs: Optional[List[Document]] = None) -> None:
        self._docs: Dict[str, Document] = {}
        self._order: Dict[str, int] = {}
        self._doc_keys: Dict[str, set] = {}
        self._postings: Dict[str, Dict[str, int]] = {}
        self._doc_len: Dict[str, int] = {}
        self._total_len = 0
        self._next_order = 0
        self._idf_cache: Optional[Dict[str, float]] = None
        self._eps_cache = 0.0
        if docs:
            self.build(docs)

    # ---------- 构建与增量更新 ----------

    def build(self, docs: Iterable[Document]) -> None:
        self._docs.clear()
        self._order.clear()
        self._doc_keys.clear()
        self._postings.clear()
        self._doc_len.clear()
        self._total_len = 0
        self._next_order = 0
        self._invalidate_idf()
        self.add_documents(docs)

    def add_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            key = self._key_for(doc)
            if key in self._docs:
                self.remove_documents([key])
            tokens = _tokenize_cached(doc.page_content)
            doc_id = str(doc.metadata.get("doc_id", ""))
            self._docs[key] = doc
            self._order[key] = self._next_order
            self._next_order += 1
            if doc_id:
                self._doc_keys.setdefault(doc_id, set()).add(key)
            self._doc_len[key] = len(tokens)
            self._total_len += len(tokens)
            for token in tokens:
                posting = self._postings.setdefault(token, {})
                posting[key] = posting.get(key, 0) + 1
        self._invalidate_idf()

    def remove_documents(self, keys: Iterable[str]) -> int:
        removed = 0
        for key in keys:
            doc = self._docs.pop(key, None)
            if doc is None:
                continue
            removed += 1
            tokens = _tokenize_cached(doc.page_content)
            self._total_len -= self._doc_len.pop(key, len(tokens))
            self._order.pop(key, None)
            doc_id = str(doc.metadata.get("doc_id", ""))
            if doc_id and doc_id in self._doc_keys:
                self._doc_keys[doc_id].discard(key)
                if not self._doc_keys[doc_id]:
                    self._doc_keys.pop(doc_id)
            for token in set(tokens):
                posting = self._postings.get(token)
                if posting is None:
                    continue
                posting.pop(key, None)
                if not posting:
                    self._postings.pop(token, None)
        if removed:
            self._invalidate_idf()
        return removed

    def remove_by_doc(self, doc_id: str) -> int:
        """删除一个文档登记的全部 chunk（同一 doc_id 可能对应多个 chunk）。"""
        return self.remove_documents(list(self._doc_keys.get(str(doc_id), ())))

    # ---------- 查询 ----------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return list(_tokenize_cached(text))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """返回按 BM25 分数降序的 (doc, score)，分数为 0 的命中不返回。"""
        if not self._docs:
            return []
        terms = [t for t in self.tokenize(query) if t in self._postings]
        if not terms:
            return []
        idf, eps = self._idf()
        n = len(self._docs)
        avgdl = self._total_len / n if n else 0.0
        scores: Dict[str, float] = {}
        for term in terms:
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            if term_idf < 0:
                term_idf = eps
            for key, tf in self._postings[term].items():
                dl = self._doc_len[key]
                denom = tf + K1 * (1 - B + B * dl / avgdl) if avgdl else tf + K1
                scores[key] = scores.get(key, 0.0) + term_idf * tf * (K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._order[kv[0]]))
        return [(self._docs[key], float(score)) for key, score in ranked[:top_k] if score > 0]

    # ---------- 内部 ----------

    def _key_for(self, doc: Document) -> str:
        doc_id = str(doc.metadata.get("doc_id", ""))
        base = f"{doc_id}:{content_hash(doc.page_content)}" if doc_id else content_hash(doc.page_content)
        if base not in self._docs:
            return base
        suffix = 1
        while f"{base}#{suffix}" in self._docs:
            suffix += 1
        return f"{base}#{suffix}"

    def _invalidate_idf(self) -> None:
        self._idf_cache = None

    def _idf(self) -> Tuple[Dict[str, float], float]:
        if self._idf_cache is not None:
            return self._idf_cache, self._eps_cache
        n = len(self._docs)
        idf: Dict[str, float] = {}
        for term, posting in self._postings.items():
            df = len(posting)
            idf[term] = math.log(n - df + 0.5) - math.log(df + 0.5)
        average = sum(idf.values()) / len(idf) if idf else 0.0
        self._idf_cache = idf
        self._eps_cache = EPSILON * average
        return idf, self._eps_cache
