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
from app.core.kb_version import bump_kb_version
from app.rag.ingest import ingest_document
from app.rag.pipeline import RAGPipeline
from app.rag.registry import KBError


def ingest_file(pipe, kb_id: str, path: Path) -> dict:
    """登记并索引单个文件，失败时只留下一条 failed 记录。

    旧写法先按 chunk 数登记、再写向量；写入失败后又 add_document 一次，
    于是同一次失败在文档列表里出现两条记录，其中一条还声称自己有 chunk。
    这里改成与 API / worker 相同的一次性流程：先登记 queued，再由
    ingest_document 推进到 indexed 或 failed。
    """
    record = pipe.registry.add_document(kb_id, path.name, chunk_count=0, status="queued")
    doc_id = record["doc_id"]
    try:
        return ingest_document(pipe.store, pipe.registry, kb_id, doc_id, path)
    except Exception as exc:
        pipe.registry.update_document(
            kb_id, doc_id, status="failed", chunk_count=0, error=str(exc)[:500]
        )
        raise


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
            pipe.store.delete_by_doc(doc["doc_id"], kb_id=kb_id)
        pipe.registry.delete_documents_of_kb(kb_id)
        pipe.retriever.invalidate(kb_id)
        bump_kb_version(kb_id)  # 其他进程（API）也要重建各自的 BM25 索引
        print(f"[ingest] 已清空知识库「{kb_name}」")

    total_chunks = 0
    for f in files:
        try:
            record = ingest_file(pipe, kb_id, f)
        except Exception as exc:
            print(f"[ingest] FAIL {f.name}: {exc}")
            continue
        total_chunks += record["chunk_count"]
        print(f"[ingest] OK   {f.name}: {record['chunk_count']} chunks (doc_id={record['doc_id'][:8]}...)")
    pipe.retriever.invalidate(kb_id)
    print(
        f"[ingest] 完成：知识库「{kb_name}」共 {total_chunks} chunks，"
        f"后端 {type(pipe.store).__name__} + {type(pipe.registry).__name__}"
    )


if __name__ == "__main__":
    main()
