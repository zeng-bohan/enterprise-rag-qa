"""Cross-Encoder 重排。

为什么需要重排：
BM25 / 向量检索都是「单条文档独立打分」的双塔模式，模型看不到「查询-文档」对之间
的细粒度交互；Cross-Encoder 把查询和文档拼成一个序列输入模型做交叉注意力，相关性
排序显著更准，代价是推理慢，所以只对融合后的 top-8 候选重排。

实现要点：
- fastembed 0.8 的 TextCrossEncoder 在 fastembed.rerank.cross_encoder 下，
  推理接口为 rerank_pairs()，输出原始 logits；
- logits 经 sigmoid 归一化到 (0,1)，0.5 作为「相关 / 无关」分界（可解释、可调）；
- 模型下载失败时降级为不重排（保持原顺序），保证链路可用。
"""
import math
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings

RERANK_MODEL = "BAAI/bge-reranker-base"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model: Optional[TextCrossEncoder] = None
        try:
            # threads 限单条推理线程数：默认全核会让并发重排互相抢核（压测结论）
            self._model = TextCrossEncoder(
                model_name=RERANK_MODEL, threads=settings.onnx_threads
            )
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not self._model or not docs:
            return [(d, 0.0) for d in docs]
        # 截断到 800 字符：重排看的是「与问题是否相关」，长文档尾部噪声大且拖慢推理
        pairs = [(query, d.page_content[:800]) for d in docs]
        raw = list(self._model.rerank_pairs(pairs))
        scored = [(d, sigmoid(float(s))) for d, s in zip(docs, raw)]
        scored.sort(key=lambda x: -x[1])
        return scored
