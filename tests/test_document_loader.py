"""文档解析：后缀分发、来源元数据与目录过滤。"""
from pathlib import Path

from langchain_core.documents import Document

from app.rag import document_loader as dl


def _write(tmp_path: Path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def test_markdown_and_txt_get_source_metadata(tmp_path):
    f = _write(tmp_path, "员工手册.md", "# 手册\n年假 15 天")
    docs = dl.load_document(f)
    assert len(docs) == 1
    assert "年假" in docs[0].page_content
    assert docs[0].metadata["source"] == "员工手册.md"

    f2 = _write(tmp_path, "说明.txt", "报销一个月内提交")
    docs2 = dl.load_document(f2)
    assert docs2[0].metadata["source"] == "说明.txt"


def test_load_directory_filters_unsupported_suffixes(tmp_path):
    _write(tmp_path, "a.md", "内容甲")
    _write(tmp_path, "ignore.py", "print(1)")
    _write(tmp_path, "b.txt", "内容乙")
    docs = dl.load_directory(tmp_path)
    sources = [d.metadata["source"] for d in docs]
    assert sources == ["a.md", "b.txt"]  # sorted 顺序，.py 被过滤


def test_pdf_routes_to_pypdf_loader(tmp_path, monkeypatch):
    recorded = {}

    class StubLoader:
        def __init__(self, path):
            recorded["path"] = path

        def load(self):
            return [Document(page_content="pdf 内容", metadata={})]

    monkeypatch.setattr(dl, "PyPDFLoader", StubLoader)
    f = _write(tmp_path, "考勤.pdf", "占位")
    docs = dl.load_document(f)
    assert recorded["path"] == str(f)
    assert docs[0].metadata["source"] == "考勤.pdf"


def test_non_listed_text_suffix_falls_back_to_text_loader(tmp_path):
    # 后缀白名单只在 load_directory 生效；单文件入口对任意文本文件宽容处理
    f = _write(tmp_path, "data.json", "{}")
    docs = dl.load_document(f)
    assert docs[0].page_content == "{}"
    assert docs[0].metadata["source"] == "data.json"
