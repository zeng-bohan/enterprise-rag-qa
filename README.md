# Enterprise Knowledge-base RAG Q&A

An enterprise RAG service for document-grounded question answering — multi-knowledge-base management, document lifecycle, hybrid retrieval, streaming answers, and multi-turn chat. Built to the feature bar of mainstream KB products (Dify / FastGPT / RAGFlow), with an evaluation suite instead of marketing numbers.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=Docker&logoColor=white)
![Recall@1](https://img.shields.io/badge/Recall%401-87.4%25-4C9F70?style=flat-square)
![Faithfulness](https://img.shields.io/badge/Faithfulness-94.1%25-4C9F70?style=flat-square)
![P95](https://img.shields.io/badge/P95-32s%20to%2010s-FF6F00?style=flat-square)
[![CI](https://github.com/zengbohan1/enterprise-rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/zengbohan1/enterprise-rag-qa/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/Tests-89%20passed-4C9F70?style=flat-square)

## Features

**Knowledge base management**
- Multiple knowledge bases; documents are uploaded per KB and indexed synchronously (parse → chunk → embed → link).
- Document lifecycle with full chunk lineage (`kb_id` / `doc_id` on every chunk): list, delete a document (its chunks go with it), delete a KB (everything goes with it).
- Two storage profiles behind one interface: **PostgreSQL + pgvector** (production, docker compose) or **Chroma + SQLite registry** (zero-dependency local). Switch with `VECTOR_BACKEND`.

**Retrieval**
- Hybrid recall: BM25 (jieba) + vector search, fused by reciprocal rank fusion, then Cross-Encoder reranking.
- Per-KB BM25 index, lazily built and invalidated on document changes.
- Answer refusal on two paths (pre-retrieval score floor + rerank floor) — no LLM tokens are burned on out-of-KB questions.
- `POST /v1/retrieval-test` — hit testing without generation (the "retrieval testing" page of Dify / FastGPT).

**Generation**
- Numbered citations, grounded prompts, and refusal fallback.
- Streaming answers over SSE: `citations → token* → done` event protocol.
- Multi-turn chat: follow-up questions are condensed into standalone queries before retrieval; conversation history is injected into generation.
- Semantic cache keyed by question + hit-document set (auto-invalidated when the KB changes); multi-turn requests bypass the cache by design.

**Operations**
- Prometheus metrics (`/metrics`), per-stage latency histograms, cache/LLM counters.
- Redis caches degrade gracefully when unavailable; local mode needs no Redis at all.
- Optional API-key auth (`X-API-Key`) on all `/v1` routes; health and metrics stay open.
- Measured on the project evaluation set: Recall@1 **87.4%**, Faithfulness **94.1%**, and P95 latency improved from **32s** to **10s** after the asynchronous pipeline work.

## Architecture

```text
                        ┌────────────────────────────────────────────┐
  upload (multipart) ──►│ Management API  /v1/kbs, /v1/kbs/{id}/docs │──► KBRegistry (PG tables / SQLite)
                        └───────────────┬────────────────────────────┘      │ metadata: KB, document, status
                                        │ chunks (kb_id, doc_id lineage)    ▼
                                        ▼                        Vector store (PGvector / Chroma)
  question ──► /v1/chat ──────► RAGPipeline
               /v1/chat/stream ──►   1. condense (multi-turn) → query rewrite (LLM)
                                     2. BM25 (per-KB) + vector recall, filtered by kb_id
                                     3. reciprocal rank fusion → Cross-Encoder rerank
                                     4. refusal floor / evidence threshold
                                     5. DeepSeek generation with citations
                                        ├─ JSON: answer + citations + timings
                                        └─ SSE:  citations → token* → done
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

### 2. Start PostgreSQL and Redis (production profile)

```bash
docker compose up -d
```

For local runs without Docker, set `VECTOR_BACKEND=chroma` in `.env` — the KB registry falls back to a local SQLite file and no Redis is required.

### 3. Ingest and run

```bash
# Windows
.venv\Scripts\python scripts/ingest_docs.py        # into the default KB; --kb name / --rebuild supported
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# macOS / Linux
.venv/bin/python scripts/ingest_docs.py
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

### 4. Try it

```bash
# Ask (non-streaming)
curl -X POST http://127.0.0.1:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"question": "员工请年假需要提前几天申请？"}'

# Ask (SSE streaming: citations → token* → done)
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream -H "Content-Type: application/json" \
  -d '{"question": "年假有多少天？"}'

# Create a KB, upload a document, hit-test retrieval
curl -X POST http://127.0.0.1:8000/v1/kbs -H "Content-Type: application/json" -d '{"name": "帮助中心"}'
curl -X POST http://127.0.0.1:8000/v1/kbs/<kb_id>/documents -F "file=@手册.pdf"
curl -X POST http://127.0.0.1:8000/v1/retrieval-test -H "Content-Type: application/json" \
  -d '{"query": "年假政策", "kb_id": "<kb_id>"}'
```

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/v1/chat` | Ask a question; supports `kb_id` and `history` (multi-turn). Returns answer, citations, timings. |
| POST | `/v1/chat/stream` | Same, as SSE: `citations → token* → done`. |
| POST | `/v1/kbs` | Create a knowledge base. |
| GET | `/v1/kbs` | List KBs with document / chunk counts. |
| DELETE | `/v1/kbs/{kb_id}` | Delete a KB and all of its chunks. |
| POST | `/v1/kbs/{kb_id}/documents` | Upload a file (multipart `file`, pdf/md/txt); parses, chunks, indexes. |
| GET | `/v1/kbs/{kb_id}/documents` | List documents with status and chunk counts. |
| DELETE | `/v1/kbs/{kb_id}/documents/{doc_id}` | Delete a document and its chunks. |
| POST | `/v1/retrieval-test` | Retrieval hit-testing without generation. |
| GET | `/health` | Health check. |
| GET | `/metrics` | Prometheus request, latency, cache, and LLM metrics. |

## Project structure

```text
app/
├── main.py            # FastAPI application (lifespan: ensure default KB)
├── config.py          # environment configuration (.env, pydantic-settings)
├── schemas.py         # request/response models
├── api/
│   ├── deps.py        # shared pipeline singleton
│   ├── chat.py        # /v1/chat (JSON + SSE streaming)
│   └── manage.py      # KB / document lifecycle, retrieval hit-testing
├── core/
│   ├── auth.py        # X-API-Key dependency (open when unset)
│   ├── llm.py         # DeepSeek client (OpenAI-compatible)
│   ├── embeddings.py  # local BGE embeddings (fastembed/ONNX) with caching
│   ├── cache.py       # Redis cache with graceful degradation
│   ├── executor.py    # bounded thread pool for CPU-bound work
│   └── metrics.py     # Prometheus metrics
└── rag/
    ├── registry.py         # KB / document registry (PG tables / SQLite)
    ├── document_loader.py  # PDF / Markdown / TXT parsing
    ├── chunker.py          # Chinese-aware semantic chunking
    ├── query_rewriter.py   # LLM query rewrite for BM25 recall
    ├── bm25_index.py       # jieba + BM25Okapi keyword index
    ├── vector_store.py     # PGvector / Chroma backends behind one interface
    ├── retriever.py        # per-KB hybrid recall + RRF fusion + rerank + refusal
    ├── reranker.py         # Cross-Encoder reranking (fastembed)
    ├── generator.py        # citations, refusal, semantic cache, multi-turn, streaming
    └── pipeline.py         # orchestration + per-stage timings
scripts/
├── ingest_docs.py      # ingest data/docs into a KB (per-document lineage)
├── ask.py              # CLI question
├── fetch_corpus.py     # fetch source documents
├── gen_qa_set.py       # generate the chunk-grounded QA eval set
├── eval_recall.py      # Recall@1/3/5 (hybrid vs vector-only baseline)
├── eval_ragas.py       # RAGAS faithfulness / relevancy
└── bench.py            # latency benchmark
tests/                  # 89 offline tests (no PG / Redis / model downloads)
data/qa_set/            # QA eval set and metric reports
docker-compose.yml
```

## Tests & evaluation

**Unit / integration tests — fully offline.** The suite stubs the LLM, Cross-Encoder, vector store, registry, and Redis, so it runs without PostgreSQL, Redis, model downloads, or network access:

```bash
pip install -r requirements-dev.txt
pytest tests -q        # 89 passed
```

**Retrieval evaluation.** `scripts/gen_qa_set.py` builds a chunk-grounded QA set (each question is answerable only from a single chunk; 10% human spot-check), and `scripts/eval_recall.py` measures Recall@k for the hybrid retriever against a vector-only baseline:

- Recall@1 **87.4%** / Recall@3 **97.9%** / Recall@5 **98.2%** (n=334, hybrid) — full report in `data/qa_set/recall_report.json`

**Generation quality.** `scripts/eval_ragas.py` scores generated answers with RAGAS:

- Faithfulness **94.1%**, answer relevancy **88.7%**, hallucination rate **5.9%** (n=100) — full report in `data/qa_set/ragas_report.json`

## Roadmap

Mainstream-KB capabilities deliberately left out of v0.6, in priority order: async ingestion for large files, parent-child (small-to-big) chunking, Office-format parsing (docx/xlsx/pptx), per-KB access control, connector sync (web/Confluence/飞书), and answer feedback loops.

## License

[MIT](LICENSE)
