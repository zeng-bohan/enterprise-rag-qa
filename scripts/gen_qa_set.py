# -*- coding: utf-8 -*-
"""生成 QA 评测集（v0.4）。

方法（评测集怎么来的、可信度如何保证）：
1. 用与入库完全相同的解析+切片流程处理 data/docs 全部文档；
2. 逐 chunk 让 LLM 生成「答案必须且只能来自该 chunk」的问答对，
   该 chunk 的内容哈希即 ground truth（召回评测的判定依据）；
3. LLM 生成存在偏差风险 → 生成后人工抽检 10%（QA 条目带 source，可溯源复核）。

用法：.venv/Scripts/python scripts/gen_qa_set.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.rag.chunker import split_documents
from app.rag.document_loader import load_directory
from app.rag.vector_store import _doc_id

OUT = ROOT / "data" / "qa_set" / "qa_300.json"
TARGET = 300

SYSTEM_PROMPT = (
    "你是企业知识库评测集构造助手。给定一段制度/法规文本，构造可用这段文本直接回答的问答对。"
    "要求：\n"
    "1. 问题写成真实员工会问的自然语言，类型多样化（数字类、流程类、条件类、定义类）；\n"
    "2. 答案必须且只能来自给定文本，忠于原文，不添加外部知识；\n"
    "3. 输出 JSON 对象：{\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}, ...]}；\n"
    "4. 构造 3~5 个问答对；文本内容不足以支撑 3 个时允许少于 3 个。"
)


def gen_for_chunk(llm, chunk, n_target: int) -> list[dict]:
    prompt = (
        f"文本（来源《{chunk.metadata.get('source', '未知')}》）：\n{chunk.page_content}\n\n"
        f"请构造约 {n_target} 个问答对。只输出 JSON。"
    )
    for attempt in range(2):
        try:
            resp = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            data = json.loads(resp.content)
            pairs = data.get("pairs", data if isinstance(data, list) else [])
            return [p for p in pairs if p.get("question") and p.get("answer")]
        except Exception:
            time.sleep(2)  # 解析失败重试一次
    return []


def main() -> None:
    llm = get_llm(temperature=0.2)  # 轻微随机性让问题措辞多样，答案仍忠于原文
    docs = load_directory(ROOT / "data" / "docs")
    chunks = split_documents(docs)
    print(f"[gen] 文档 {len(docs)} 篇 → 切片 {len(chunks)} 个 chunk")

    n_per = max(3, min(5, TARGET // max(len(chunks), 1) + 1))
    qa_set = []
    for i, chunk in enumerate(chunks, 1):
        pairs = gen_for_chunk(llm, chunk, n_per)
        for p in pairs:
            qa_set.append(
                {
                    "question": p["question"],
                    "answer": p["answer"],
                    "chunk_id": _doc_id(chunk),  # 内容哈希 = 检索命中的判定依据
                    "source": chunk.metadata.get("source", ""),
                }
            )
        if i % 10 == 0:
            print(f"[gen] {i}/{len(chunks)} chunks, 已生成 {len(qa_set)} 条")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(qa_set, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] 完成：{len(qa_set)} 条 QA → {OUT}")


if __name__ == "__main__":
    main()
