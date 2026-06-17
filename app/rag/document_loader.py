"""文档解析：PDF / Markdown / TXT 统一入口。

职责边界：
- 只做「文件 → 带元数据的 Document 列表」；
- metadata.source 统一记录文件名，供下游引用溯源（回答中的 [1][2] 标注）。
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def load_document(path: Path) -> List[Document]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def load_directory(dir_path: Path) -> List[Document]:
    docs: List[Document] = []
    for f in sorted(Path(dir_path).iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(load_document(f))
    return docs
