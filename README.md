# Enterprise Knowledge-base RAG Q&A

An enterprise RAG service for document-grounded question answering. It combines document parsing, hybrid retrieval, reranking, citation-grounded generation, answer refusal, and observability.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Recall@1](https://img.shields.io/badge/Recall%401-87.4%25-4C9F70?style=flat-square)
![Faithfulness](https://img.shields.io/badge/Faithfulness-94.1%25-4C9F70?style=flat-square)
![P95](https://img.shields.io/badge/P95-32s%20to%2010s-FF6F00?style=flat-square)
[![CI](https://github.com/zengbohan1/enterprise-rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/zengbohan1/enterprise-rag-qa/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/Tests-60%20passed-4C9F70?style=flat-square)

## Capabilities

- Parse PDF, Markdown, and TXT files into a knowledge base.
- Retrieve with BM25 plus vector search, fused by reciprocal rank fusion.
- Rerank candidates with a Cross-Encoder before generation.
- Generate answers with numbered citations, or refuse when evidence is insufficient.
- Run asynchronously with Prometheus metrics and Redis-backed caches that degrade gracefully when unavailable.
- Use PostgreSQL with pgvector by default, or set `VECTOR_BACKEND=chroma` for local Chroma storage.

Measured on the project evaluation set: Recall@1 **87.4%**, Faithfulness **94.1%**, and P95 latency improved from **32s** to **10s** after the asynchronous pipeline work.

## Architecture

```text
Question
  -> query rewrite
  -> BM25 + vector retrieval
  -> reciprocal rank fusion
  -> Cross-Encoder reranking
  -> evidence threshold / refusal
  -> DeepSeek generation with citations
  -> answer, citations, and timings
```

## Quick start

### 1. Create the environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env

# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set your `DEEPSEEK_API_KEY` in `.env`.

### 2. Start PostgreSQL and Redis

```bash
docker compose up -d
```

### 3. Build the index and run the service

```bash
# Windows
.venv\Scripts\python scripts/ingest_docs.py
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# macOS / Linux
.venv/bin/python scripts/ingest_docs.py
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Try a CLI question before starting the API if needed:

```bash
# Windows
.venv\Scripts\python scripts/ask.py "员工请年假需要提前几天申请？"

# macOS / Linux
.venv/bin/python scripts/ask.py "员工请年假需要提前几天申请？"
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/v1/chat` | Submit `{"question": "..."}` and receive an answer, citations, and stage timings. |
| GET | `/health` | Health check. |
| GET | `/metrics` | Prometheus request, latency, cache, and LLM metrics. |

## Project structure

```text
app/
├── main.py            # FastAPI application
├── config.py          # environment configuration (.env, pydantic-settings)
├── schemas.py         # request/response models
├── api/chat.py        # /v1/chat endpoint
├── core/
│   ├── llm.py         # DeepSeek client (OpenAI-compatible)
│   ├── embeddings.py  # local BGE embeddings (fastembed/ONNX) with caching
│   ├── cache.py       # Redis cache with graceful degradation
│   ├── executor.py    # bounded thread pool for CPU-bound work
│   └── metrics.py     # Prometheus metrics
└── rag/
    ├── document_loader.py  # PDF / Markdown / TXT parsing
    ├── chunker.py          # Chinese-aware semantic chunking
    ├── query_rewriter.py   # LLM query rewrite for BM25 recall
    ├── bm25_index.py       # jieba + BM25Okapi keyword index
    ├── vector_store.py     # PGvector / Chroma backends behind one interface
    ├── retriever.py        # hybrid recall + RRF fusion + reranking + refusal
    ├── reranker.py         # Cross-Encoder reranking (fastembed)
    ├── generator.py        # citation-grounded generation, refusal, semantic cache
    └── pipeline.py         # orchestration + per-stage timings
scripts/
├── ingest_docs.py      # build the index from data/docs
├── ask.py              # CLI question
├── fetch_corpus.py     # fetch source documents
├── gen_qa_set.py       # generate the chunk-grounded QA eval set
├── eval_recall.py      # Recall@1/3/5 (hybrid vs vector-only baseline)
├── eval_ragas.py       # RAGAS faithfulness / relevancy
└── bench.py            # latency benchmark
tests/                  # 60 offline tests (no PG / Redis / model downloads)
data/qa_set/            # QA eval set and metric reports
docker-compose.yml
```

## Tests & evaluation

**Unit / integration tests — fully offline.** The suite stubs the LLM, Cross-Encoder, vector store, and Redis, so it runs without PostgreSQL, Redis, model downloads, or network access:

```bash
pip install -r requirements-dev.txt
pytest tests -q        # 60 passed
```

**Retrieval evaluation.** `scripts/gen_qa_set.py` builds a chunk-grounded QA set (each question is answerable only from a single chunk; 10% human spot-check), and `scripts/eval_recall.py` measures Recall@k for the hybrid retriever against a vector-only baseline:

- Recall@1 **87.4%** / Recall@3 **97.9%** / Recall@5 **98.2%** (n=334, hybrid) — full report in `data/qa_set/recall_report.json`

**Generation quality.** `scripts/eval_ragas.py` scores generated answers with RAGAS:

- Faithfulness **94.1%**, answer relevancy **88.7%**, hallucination rate **5.9%** (n=100) — full report in `data/qa_set/ragas_report.json`

## License

[MIT](LICENSE)
