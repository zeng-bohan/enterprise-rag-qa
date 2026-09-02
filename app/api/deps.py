"""API 依赖单例：pipeline 只构造一次（Embedding / 向量库加载是主要启动开销）。

chat 与 manage 两个路由共用同一实例；测试在 import 前 stub RAGPipeline 类即可整体替换。
"""
from app.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
