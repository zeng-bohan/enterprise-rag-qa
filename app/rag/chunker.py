"""语义切片：按中文语义边界递归切分 + 重叠窗口保持上下文完整。

切分粒度是 RAG 效果的第一道开关：
- 切太碎：语义断裂，召回片段答非所问；
- 切太长：单 chunk 噪声大，相关度被稀释；
- 重叠窗口（overlap）保证跨 chunk 边界的句子（如「前文所述」）不丢失。
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文语义边界优先级：段落 > 换行 > 句末标点 > 句中标点
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def split_documents(
    docs: List[Document], chunk_size: int = 500, overlap: int = 60
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=CHINESE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        keep_separator=False,
    )
    return splitter.split_documents(docs)
