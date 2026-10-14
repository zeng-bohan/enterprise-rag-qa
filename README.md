# 企业知识库 RAG 问答

基于 FastAPI 的文档问答服务，支持多知识库管理、文档索引、混合检索、引用回答和流式对话。项目提供 Chroma 本地模式与 PostgreSQL/pgvector 部署配置。

## 功能

- 管理多个知识库和文档；支持 PDF、Markdown、TXT 上传、删除和文档到切片的来源追踪。
- 使用 BM25 与向量检索召回，经过 RRF 融合和 Cross-Encoder 重排后生成回答。
- 支持多轮对话、引用、低相关度拒答、语义缓存和 SSE 流式输出。
- 支持 Chroma 与 pgvector 向量后端；元数据可使用 SQLite 或 PostgreSQL，Redis 用于缓存与可选的异步摄取。
- 可选 API Key 鉴权、请求限流和 Prometheus 指标。

## 技术栈

| 范围 | 技术 |
| --- | --- |
| 服务与 API | Python、FastAPI、Uvicorn、Pydantic Settings |
| RAG 与模型 | LangChain、DeepSeek 兼容 API、FastEmbed/ONNX、BM25、Cross-Encoder |
| 存储 | Chroma 或 PostgreSQL/pgvector；SQLite 元数据；可选 Redis |
| 运行与验证 | Docker Compose、pytest、GitHub Actions |

CI 使用 Python 3.12。向量化与重排模型在本机运行；首次使用时可能需要下载模型文件。

## 处理流程

~~~mermaid
flowchart LR
    D[PDF / Markdown / TXT] --> P[解析与切片]
    P --> V[向量索引与文档元数据]
    Q[问题] --> R[BM25 + 向量召回]
    V --> R
    R --> F[RRF 融合与重排]
    F --> L[LLM 生成]
    L --> A[回答与引用 / SSE]
~~~

## 项目结构

~~~text
app/
├── api/       # 对话、知识库与文档 API
├── core/      # 配置、LLM、缓存、鉴权、指标与执行器
└── rag/       # 文档解析、切片、索引、检索与生成
data/
├── docs/      # 随仓库提供的示例文档
└── qa_set/    # 评测数据与报告
scripts/       # 导入、提问与评测工具
tests/         # 离线测试与存储后端契约测试
docs/DESIGN.md # 主要设计决策
docker-compose.yml
.env.example
~~~

## 快速开始

### 本地模式

本地模式使用 Chroma 和 SQLite，不需要启动 PostgreSQL 或 Redis。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
~~~

在 `.env` 中设置：

~~~dotenv
VECTOR_BACKEND=chroma
DEEPSEEK_API_KEY=<你的模型 API Key>
~~~

模型 API Key 用于生成回答；没有可用 Key 时服务仍可启动，但无法完成真实 LLM 回答。随后导入仓库中的示例文档并启动 API：

~~~bash
python scripts/ingest_docs.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

打开 [API 文档](http://127.0.0.1:8000/docs)。Windows PowerShell 可将激活命令替换为 `.venv\Scripts\Activate.ps1`，并用 `Copy-Item .env.example .env` 复制配置文件。

### PostgreSQL 与 Redis

`docker compose up -d` 会启动 PostgreSQL、Redis 和摄取 worker；Compose 不包含 FastAPI 服务本身。使用 pgvector 时，在 `.env` 中设置 `VECTOR_BACKEND=pgvector`，并确认 `POSTGRES_DSN` 与 `REDIS_URL` 指向本机映射端口，然后在宿主机运行上面的 Uvicorn 命令。

同步摄取是默认模式。使用队列时，API 与 worker 两侧都要设置 `INGEST_MODE=queue`。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/v1/chat` | 普通问答，支持知识库和对话历史 |
| POST | `/v1/chat/stream` | SSE 流式问答 |
| POST | `/v1/kbs` | 创建知识库 |
| GET | `/v1/kbs` | 列出知识库 |
| DELETE | `/v1/kbs/{kb_id}` | 删除知识库及其内容 |
| POST | `/v1/kbs/{kb_id}/documents` | 上传文档 |
| GET | `/v1/kbs/{kb_id}/documents` | 列出知识库文档 |
| DELETE | `/v1/kbs/{kb_id}/documents/{doc_id}` | 删除文档及其切片 |
| POST | `/v1/retrieval-test` | 只检索、不生成，用于检查召回结果 |
| GET | `/health`、`/metrics` | 健康检查和 Prometheus 指标 |

启用 `API_KEYS` 后，`/v1` 业务接口需通过 `X-API-Key` 请求头鉴权；未配置时不启用鉴权。`/health` 与 `/metrics` 始终开放。完整交互文档可在服务启动后通过 `/docs` 查看。

示例问答：

~~~bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"员工请年假需要提前几天申请？"}'
~~~

## 配置

配置从仓库根目录的 `.env` 读取；完整字段和默认值见 [.env.example](.env.example)。

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` | LLM 服务凭证、模型和兼容 API 地址 |
| `VECTOR_BACKEND` | 选择 `chroma` 或 `pgvector` |
| `POSTGRES_DSN` | PostgreSQL 连接串，pgvector 模式使用 |
| `REDIS_URL` | 缓存与异步摄取队列连接 |
| `API_KEYS` | 逗号分隔的 API Key；留空关闭鉴权 |
| `INGEST_MODE` | `sync` 同步索引；`queue` 使用 arq worker |
| `MAX_UPLOAD_MB`、`RATE_LIMIT_RPS` | 上传大小与可选限流参数 |

知识库和文档元数据表由注册中心初始化；仓库没有独立的 Alembic/Flyway 迁移命令。启用队列摄取时，请同时启动 Compose worker。

## 开发命令

~~~bash
python -m pip install -r requirements-dev.txt
pytest tests -m "not contract" -q
~~~

后端契约测试需要 PostgreSQL/pgvector 与 Redis：

~~~bash
docker compose up -d
pytest tests -m contract -q
~~~

CI 分别运行离线测试与真实存储后端契约测试。RAG 设计说明见 [docs/DESIGN.md](docs/DESIGN.md)。

## 部署说明

现有 Compose 文件负责 PostgreSQL、Redis 和 worker，不负责构建或启动 API。可以在宿主机运行 API；若要容器化 API，需要为服务补充对应的镜像构建和运行配置。将服务暴露到可信网络前，请配置 `API_KEYS`，并按实际环境设置上传限制与限流。