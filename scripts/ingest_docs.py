"""解析 data/docs 下全部 PDF / Markdown / TXT，按文档登记后切分入库（v0.6 多知识库语义）。

每个源文件登记为一条 document 记录（doc_id），chunk 携带 kb_id / doc_id 血缘，
支持按文档 / 按知识库删除。默认写入 DEFAULT_KB（缺省 default）。

用法：.venv/Scripts/python scripts/ingest_docs.py            # 写入默认知识库
      .venv/Scripts/python scripts/ingest_docs.py --kb 帮助中心
      .venv/Scripts/python scripts/ingest_docs.py --rebuild # 先清空该知识库再全量重建
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.chunker import split_documents
from app.rag.document_loader import load_document
from app.rag.pipeline import RAGPipeline
from app.rag.registry import KBError


def main() -> None:
    parser = argparse.ArgumentParser(description="把 data/docs 下的文档登记入库")
    parser.add_argument("--kb", default=None, help="目标知识库名（缺省 DEFAULT_KB）")
    parser.add_argument("--rebuild", action="store_true", help="入库前清空目标知识库")
    args = parser.parse_args()

    docs_dir = ROOT / "data" / "docs"
    files = sorted(f for f in docs_dir.iterdir() if f.is_file()) if docs_dir.exists() else []
    if not files:
        print(f"[ingest] {docs_dir} 下没有文档，请先放入 PDF / Markdown 文件")
        return

    pipe = RAGPipeline()
    kb_name = args.kb or settings.default_kb
    try:
        kb_id = pipe.registry.kb_id_by_name(kb_name) or pipe.registry.create_kb(kb_name, "脚本入库")["kb_id"]
    except KBError as exc:
        print(f"[ingest] 知识库创建失败: {exc}")
        return

    if args.rebuild:
        for doc in pipe.registry.list_documents(kb_id):
            pipe.store.delete_by_doc(doc["doc_id"])
        pipe.registry.delete_documents_of_kb(kb_id)
        pipe.retriever.invalidate(kb_id)
        print(f"[ingest] 已清空知识库「{kb_name}」")

    total_chunks = 0
    for f in files:
        try:
            chunks = split_documents(load_document(f))
            record = pipe.registry.add_document(kb_id, f.name, chunk_count=len(chunks))
            pipe.store.add_documents(chunks, kb_id=kb_id, doc_id=record["doc_id"])
            total_chunks += len(chunks)
            print(f"[ingest] ✓ {f.name}: {len(chunks)} chunks (doc_id={record['doc_id'][:8]}…)")
        except Exception as exc:
            pipe.registry.add_document(kb_id, f.name, chunk_count=0, status="failed", error=str(exc)[:500])
            print(f"[ingest] ✗ {f.name}: {exc}")
    pipe.retriever.invalidate(kb_id)
    print(
        f"[ingest] 完成：知识库「{kb_name}」共 {total_chunks} chunks，"
        f"后端 {type(pipe.store).__name__} + {type(pipe.registry).__name__}"
    )


if __name__ == "__main__":
    main()
