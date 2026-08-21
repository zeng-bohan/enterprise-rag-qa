# Enterprise Knowledge-base RAG Q&A

An enterprise RAG service for document-grounded question answering. It combines document parsing, hybrid retrieval, reranking, citation-grounded generation, answer refusal, and observability.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Recall@1](https://img.shields.io/badge/Recall%401-87.4%25-4C9F70?style=flat-square)
![Faithfulness](https://img.shields.io/badge/Faithfulness-94.1%25-4C9F70?style=flat-square)
![P95](https://img.shields.io/badge/P95-32s%20to%2010s-FF6F00?style=flat-square)

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
├── config.py          # environment configuration
├── api/chat.py        # /v1/chat endpoint
├── core/
│   ├── llm.py         # DeepSeek client
│   └── embeddings.py  # local BGE embeddings
└── rag/
    ├── document_loader.py
    ├── chunker.py
    ├── vector_store.py
    ├── retriever.py
    ├── generator.py
    └── pipeline.py
scripts/
├── ingest_docs.py
└── ask.py
docker-compose.yml
```

## Validation status

The repository currently contains the runnable application and ingestion scripts. Its `tests/` directory is only a placeholder, so the documented validation path is the ingestion command, CLI query, and `/docs` API flow above rather than a claimed test suite.

## License

[MIT](LICENSE)
