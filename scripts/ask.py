"""命令行快速验证问答效果。

用法：
    .venv/Scripts/python scripts/ask.py "员工请年假需要提前几天申请？"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.pipeline import RAGPipeline


def main() -> None:
    question = " ".join(sys.argv[1:]) or "员工请年假需要提前几天申请？"
    result = RAGPipeline().ask(question)
    print(f"问：{question}\n")
    print(f"答：{result['answer']}\n")
    if result["citations"]:
        print("引用：")
        for c in result["citations"]:
            print(f"  [{c['index']}] {c['source']}  (相关度 {c['score']:.3f})")
    if not result.get("grounded"):
        status = "拒答（未调用生成）"
    elif result.get("cached"):
        status = "缓存命中"
    else:
        status = "真实生成"
    print(f"\n检索 {result['retrieval_ms']}ms / 总耗时 {result['latency_ms']}ms / {status}")


if __name__ == "__main__":
    main()
