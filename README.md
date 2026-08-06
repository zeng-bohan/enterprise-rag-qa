# 企业知识库 RAG 问答系统

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![PGvector](https://img.shields.io/badge/PGvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![RAGAs](https://img.shields.io/badge/RAGAs-4B8BBE?style=flat-square)

![Recall@1](https://img.shields.io/badge/Recall%401-87.4%25-4C9F70?style=flat-square)
![Faithfulness](https://img.shields.io/badge/Faithfulness-94.1%25-4C9F70?style=flat-square)
![P95](https://img.shields.io/badge/P95_32s%20%E2%86%92%2010s-FF6F00?style=flat-square)

基于大模型（DeepSeek）的企业级知识库问答系统：文档解析 → 语义切片 → 向量化入库 → 混合检索 → 引用溯源生成，支持无答案拒答与评测流水线。

> 当前版本：v0.5（异步并发 + Prometheus 监控）。Roadmap 见文末。

## 架构

```
用户问题
   │
   ▼
FastAPI (/v1/chat)
   │
   ├─ 1) 查询改写（DeepSeek，口语 → 关键词查询；失败降级原句）
   ▼
混合检索 Retriever
   ├─ 2) 双路召回：BM25（jieba 分词，字面匹配） + 向量检索（BGE，语义匹配）各 top-20
   ├─ 3) RRF 融合（rank 折扣融合，避免单路高分垄断）
   └─ 4) Cross-Encoder 重排（bge-reranker，top-8 精排，sigmoid 0.5 分界过滤无关项）
   │         └─ 前置拒答：BM25 无命中且向量相关度低于阈值 → 不调用生成 LLM
   ▼
Generator ──► DeepSeek 生成（System Prompt 约束 + [n] 引用溯源）
               无检索命中 → 拒答话术（不调用 LLM）
   ▼
回答 + 引用 + 耗时统计
```

- **LLM**：DeepSeek（OpenAI 兼容协议，`langchain-openai` 接入）
- **Embedding**：BGE-small-zh（`fastembed` 本地 ONNX 推理，无 GPU / 无外部 API 依赖）
- **向量库**：PostgreSQL 16 + pgvector（docker compose 一键起；`VECTOR_BACKEND=chroma` 可切回本地 Chroma）
- **缓存**：Redis 7（查询改写 / 问答结果语义缓存 / Embedding 缓存，宕机自动降级）
- **文档解析**：PDF（pypdf）/ Markdown / TXT 统一入口
- **检索**：BM25 + 向量混合（RRF）+ 查询改写 + Cross-Encoder 重排
- **服务**：FastAPI 异步接口（LLM 全链路 ainvoke）+ CPU 密集任务专用线程池 + Prometheus 指标

## 快速开始

```bash
# 1. 环境
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2. 配置：复制 .env.example 为 .env，填入 DeepSeek API Key

# 3. 起基础设施（PostgreSQL+pgvector、Redis）
docker compose up -d

# 4. 入库（解析 data/docs 全部文档，全量重建索引）
.venv/Scripts/python scripts/ingest_docs.py

# 5. 命令行快速验证
.venv/Scripts/python scripts/ask.py "员工请年假需要提前几天申请？"

# 6. 启动服务
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Swagger 文档：http://127.0.0.1:8000/docs
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | `{"question": "..."}` → 回答 + 引用溯源 + 分阶段耗时 |
| GET | `/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标（请求 / 延迟直方图 / 缓存命中率 / LLM 调用） |

## 目录结构

```
app/
├── main.py            # FastAPI 入口
├── config.py          # .env 配置（pydantic-settings）
├── schemas.py         # 出入参模型
├── api/chat.py        # /v1/chat
├── core/llm.py        # DeepSeek 客户端
├── core/embeddings.py # BGE 向量化（fastembed）
└── rag/
    ├── document_loader.py  # PDF/MD/TXT 解析
    ├── chunker.py          # 中文语义切片（重叠窗口）
    ├── vector_store.py     # 向量库抽象层（PGvector / Chroma 双实现）
    ├── retriever.py        # 检索层（阈值过滤 + 拒答依据）
    ├── generator.py        # 生成层（引用溯源 / 幻觉抑制）
    └── pipeline.py         # 主流程编排 + 耗时统计
```

## Roadmap

- [x] v0.1 全链路：解析 → 切片 → 向量化 → 检索 → 生成 → 拒答
- [x] v0.2 混合检索：BM25 + 向量（RRF 融合）+ 查询改写 + Cross-Encoder 重排
- [x] v0.3 存储升级：PGvector + Redis 语义缓存 + Docker Compose 一键部署
- [x] v0.4 评测：334 条 QA 评测集 + Recall@k 实测（Recall@1 66.2% → 87.4%）+ RAGAs 实测（100 条抽样：Faithfulness 94.1% / Answer Relevance 88.7%，幻觉率 5.9%）
- [x] v0.5 性能：异步并发（LLM 全链路 ainvoke + CPU 专用有界线程池）+ Prometheus 指标；压测 P95 32s → 10s（详见 docs/DESIGN.md 第 12 节）
