"""解析 data/docs 下全部 PDF / Markdown / TXT，切分后全量重建向量索引。

用法：.venv/Scripts/python scripts/ingest_docs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.chunker import split_documents
from app.rag.document_loader import load_directory
from app.rag.vector_store import get_store


def main() -> None:
    docs_dir = ROOT / "data" / "docs"
    docs = load_directory(docs_dir)
    if not docs:
        print(f"[ingest] {docs_dir} 下没有文档，请先放入 PDF / Markdown 文件")
        return

    chunks = split_documents(docs)
    store = get_store()
    store.clear()  # 全量重建，保证脚本可重复执行
    n = store.add_documents(chunks)
    backend = type(store).__name__
    print(f"[ingest] 解析文档 {len(docs)} 篇 → 切分 {len(chunks)} 个 chunk → 入库 {n} 条（后端：{backend}）")


if __name__ == "__main__":
    main()
