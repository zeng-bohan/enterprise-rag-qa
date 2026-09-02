"""语义切片：粒度上限、重叠窗口与元数据透传。"""
from langchain_core.documents import Document

from app.rag.chunker import split_documents


def _long_doc(paragraphs: int = 12) -> Document:
    body = "\n\n".join(
        f"第{i}条 员工应当遵守考勤与假期管理的各项规定，年假申请须提前在系统提交。" for i in range(paragraphs)
    )
    return Document(page_content=body, metadata={"source": "考勤制度.md"})


def test_chunks_respect_chunk_size():
    chunks = split_documents([_long_doc()], chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_metadata_is_preserved_on_chunks():
    chunks = split_documents([_long_doc(6)], chunk_size=200, overlap=40)
    assert all(c.metadata.get("source") == "考勤制度.md" for c in chunks)


def test_short_doc_yields_single_chunk():
    doc = Document(page_content="年假须提前申请。", metadata={"source": "a.md"})
    chunks = split_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "年假须提前申请"  # keep_separator=False 剥掉句号


def test_chinese_sentence_boundaries_preferred():
    # 句号是次级分隔符：切片应在句子边界附近断开，而不是把句子拦腰截断
    body = "。".join(["员工请年假须提前三个工作日提交申请"] * 20) + "。"
    doc = Document(page_content=body, metadata={"source": "b.md"})
    chunks = split_documents([doc], chunk_size=120, overlap=20)
    assert all(len(c.page_content) <= 120 for c in chunks)
    # 每个切片都应从句首开始（keep_separator=False 去掉句号，但不应残留句中碎片）
    assert all(not c.page_content.startswith("作日") for c in chunks)


def test_empty_input_yields_no_chunks():
    doc = Document(page_content="", metadata={"source": "c.md"})
    assert split_documents([doc]) == []
