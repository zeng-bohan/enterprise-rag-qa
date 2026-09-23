# 设计决策记录（Design Decisions）

> 每个决策记录：背景 → 候选方案 → 选择 → 理由 → 常见疑问点。

## 1. LLM 选型：DeepSeek（OpenAI 兼容协议接入）

- **背景**：RAG 系统需要生成模型，且要控制成本（学生项目 + 评测要跑数百次调用）。
- **候选**：OpenAI / 智谱 / 通义 / 本地 Ollama。
- **选择**：DeepSeek（`deepseek-chat`），经 `langchain-openai` 的 `ChatOpenAI` 接入。
- **理由**：价格最低档（约 ¥1-2/百万 token）、中文能力强、API 兼容 OpenAI 协议，LangChain 生态零改造成本；换任何 OpenAI 兼容模型只需改 `.env` 的 base_url 和 model。
- **常见疑问**：为什么不直接用 OpenAI SDK 而用 LangChain 封装？——LangChain 统一了流式、重试、消息模型，且后续换模型供应商代码不动。

## 2. Embedding 选型：本地 BGE（fastembed / ONNX Runtime）

- **候选**：API 类（OpenAI/智谱 embedding）/ 本地 sentence-transformers（torch）/ 本地 fastembed（ONNX）。
- **选择**：`bge-small-zh-v1.5` + fastembed。
- **理由**：DeepSeek 没有 embedding API，必须另选；BGE 是中文检索主流开源系列；fastembed 免 torch（环境轻、装得快），CPU 单条 20-50ms 够用；数据不出本机。
- **常见疑问**：压测阶段想提速怎么办？——换 sentence-transformers + RTX 4060 GPU，接口不变（都实现 embed_documents/embed_query）。
- **踩坑记录**：fastembed 0.8 的 `query_embed` 返回 `Iterable[ndarray]`，`list()` 再经 LangChain 包裹成 `[[array]]` 导致 chromadb 报错；修法 `next(iter(...)).tolist()`。

## 3. 向量库：双后端抽象（Chroma ↔ PGvector）

- **背景**：v0.1 要快速跑通（零依赖），生产形态要 PGvector（SQL 运维、混合检索同库、团队熟悉）。
- **选择**：`VectorStore` 接口 + `ChromaStore` / `PGvectorStore` 两个实现，`.env` 的 `VECTOR_BACKEND` 切换。
- **理由**：检索层只依赖接口，换后端零业务代码改动——「面向接口编程」在 RAG 工程里的落法。
- **实现细节**：PGvector 用余弦距离 `<=>` 算子，`score = 1 - distance`，与 Chroma relevance score 口径一致，检索阈值通用。
- **常见疑问**：为什么用连接池？——FastAPI 同步 def 跑在线程池，多线程共享单个 psycopg 连接会协议交错，必须 `psycopg_pool` 池化。

## 4. 切片：中文语义边界的递归切分

- **候选**：固定长度硬切 / 按标题切 / 递归字符切分（带中文标点边界）。
- **选择**：`RecursiveCharacterTextSplitter`，separators 按「段落>换行>句末标点>句中标点」优先级，chunk 500 字、重叠 60 字。
- **理由**：纯固定长度会把句子拦腰截断；标题切分对无结构化标题的文档失效；递归切分在中文标点处断句 + 重叠窗口兼顾语义完整。
- **常见疑问**：切太碎/太长各有什么问题？——碎则语义断裂答非所问，长则噪声稀释相关度。

## 5. 检索：为什么做混合检索 + RRF + 重排

- **背景（真实数据驱动）**：v0.1 纯向量检索时，「年假提前几天」的正确 chunk 只排第 2——语义空间稀释了字面匹配。
- **选择**：查询改写（LLM）→ BM25（jieba 分词）+ 向量双路 top-20 → RRF 融合 → Cross-Encoder（bge-reranker）top-8 重排。
- **为什么 RRF 而不是加权求和**：两路分数分布不同（BM25 无上界、余弦 ∈[-1,1]），加权要先归一化，权重还要调参；RRF 只依赖排名，免归一化免调参，实践中更稳。
- **为什么重排只取 top-8**：Cross-Encoder 是交叉注意力，精度高但慢；对融合后前 8 个精排，精度/成本平衡点。
- **为什么 sigmoid 0.5 分界**：fastembed 重排器输出原始 logits，sigmoid 归一化到 (0,1)，0.5 成为可解释的「相关/无关」分界。
- **结果**：年假问题正确 chunk 升至 #1（0.986）；无关文档被 0.5 阈值过滤。

## 6. 拒答：两条路径，缺一不可

- **路径 A（前置，省 token）**：BM25 无命中且向量最高分低于阈值 → 直接拒答，不调用生成 LLM。
- **路径 B（重排后）**：重排后所有候选 < 0.5 → 拒答。
- **路径 C（兜底）**：System Prompt 约束「只用参考资料作答，否则回复固定话术」——检索漏网时靠 prompt 兜住幻觉。
- **常见疑问**：为什么不只靠 prompt？——prompt 兜底消耗一次生成调用（钱+延迟），且大模型不总是听话；前置拦截是成本最优解。

## 7. 缓存：三个对象，三种键设计

| 对象 | 键 | 为什么 |
|---|---|---|
| 查询改写 | 问题哈希 | 改写是纯函数（temperature=0） |
| 问答结果 | 问题 + 命中文档内容哈希集 | 检索集变则失效，避免脏缓存 |
| Embedding | chunk 内容哈希，TTL 7 天 | 模型固定向量稳定；换模型 7 天后自动刷新 |

- **为什么问答缓存键用内容哈希而非文档 id**：id 由内容派生且幂等，知识库更新后 id 不变，按 id 做键缓存不会失效。
- **为什么不缓存 citations**：引用必须反映当前检索结果，每次实时组装。
- **容错**：Redis 不可用全部降级直查，缓存层永不阻塞主链路。

## 8. 生成参数：temperature=0

- **理由**：评测集需要确定性（同问题同答案，Recall/正确率可复现）；生产问答也不需要创造性。

## 9. PDF 解析：必须带 ToUnicode 映射

- **踩坑记录**：用 PyMuPDF 内置 CJK 字体生成的 PDF，pypdf 抽取全乱码——内置字体无 ToUnicode 映射；修法：生成 PDF 时嵌入真实字体（微软雅黑），抽取正常。判断一个 PDF 能否被可靠解析，先看字体是否嵌入。

## 10. 基础设施排障记录（真实踩坑）

- **WSL 内核过旧**：Docker Desktop 引擎启动失败，日志 `checking preconditions: WSL update required`。商店通道 `wsl --update` 卡 0%，改走 GitHub 官方 MSI（microsoft/WSL releases，2.7.11）安装解决。
- **Docker Hub 拉取失败**：`auth.docker.io` 认证超时（国内网络），配 `registry-mirrors`（daocloud / 1ms / xuanyuan / dockerpull）解决。
- **端口冲突**：宿主机 6379 被本地旧版 Redis（parking-env 开发环境）占用，客户端 `HELLO` 握手报 unknown command——实际没连到容器里的 Redis 7。排查思路：`netstat -ano` 找占用进程 → 确认归属 → 初期将容器映射到 6380 临时规避 → 确认旧 Redis 永久弃用并停用后，恢复标准 6379 映射。

## 11. 评测：RAGAs 指标口径与实测

- **为什么只选 Faithfulness + Answer Relevance**：Faithfulness 逐句检验「答案的每个论断能否在检索到的上下文找到依据」，是幻觉的直接度量（幻觉率 = 1 - Faithfulness）；Answer Relevance 检验答案是否切题。两个指标覆盖质量的两面，且都能用 DeepSeek 作 LLM-as-Judge、本地 BGE 作 Embedding——零外部依赖、可复现。
- **打分口径**：`retrieved_contexts` 用完整 chunk 而非引用 snippet（snippet 只有前 80 字，不足以支撑逐句判定）；抽样 100 条控制成本（`--full` 跑全量）。
- **实测结果（100 条抽样）**：Faithfulness 94.1%、Answer Relevance 88.7%、幻觉率 5.9%。
- **工程坑**：
  - ragas 0.4.3 的 collections 指标不兼容 `evaluate()`，直接调 `abatch_score`；它会把输入字典的每个键都作为关键字参数传给 `ascore()`，必须按指标签名裁剪字段，否则报未知参数。
  - 指标内部走异步，客户端必须用 `AsyncOpenAI`（同步客户端 `agenerate` 直接报错）。
  - 默认 `max_tokens=1024`：Faithfulness 一次性输出全部语句的判定 JSON，长答案被截断抛 `IncompleteOutputException`；`llm_factory(..., max_tokens=8192)`（deepseek-chat 输出上限）解决。
  - 性能：100 条 build 串行要十几分钟 → 8 线程并发（先预热一次避免 ONNX 模型多线程首次加载竞争）；打分分批 gather（每批 20）控制 DeepSeek 并发防 429，总时长约 10 分钟。

## 12. 性能（v0.5）：异步并发改造与压测结论

### 压测方法

- 自研脚本 `scripts/bench.py`：N 个 asyncio 协程（N = 并发数）各串行打请求，问题取自自建 QA 集、每个只打一次——问答有语义缓存，重复打同一问题会命中缓存跳过 LLM，把延迟测虚；
- 两条路径分开测：缓存命中路径（纯本地 CPU，暴露推理争抢）与未命中路径（含 2 次 LLM 调用，暴露外部依赖与线程模型）。

### 改造前 baseline（sync def + FastAPI 40 线程池 + ONNX 默认全核）

- 缓存命中 8 并发 × 24：P50 19.8s（单发仅 1.0s）——CPU 争抢把单请求拖慢约 20 倍；
- 未命中 8 并发 × 100：QPS 0.9、P95 32s——LLM 超时重试把线程池耗尽，请求排队雪崩。

### 三处改造

1. **endpoint 改 async def，LLM 全链路 `ainvoke`**：等待不占线程，LLM 重试不再拖垮线程池（P95 32s → 10s 的直接原因）；
2. **CPU 密集任务进专用有界线程池**（`app/core/executor.py`）：BM25/向量/重排超出排队，并发可控；
3. **ONNX 显式限线程**（`onnx_threads`）：默认全核让单条推理吃满 CPU、并发零加速。实测 2 并发重排耗时 1.7×、4 并发 2.6×——本机 ONNX 推理并行上限约 2×（i5-14400F，6 大核 + 4 小核，混合架构调度 + 65W 功耗墙）。

### 参数扫描（缓存命中路径，8 并发 × 24）

| onnx_threads | cpu_workers | QPS | P50 | 单发 |
|---|---|---|---|---|
| 2 | 6 | 0.88 | 20.1s | 2.7s |
| 2 | 4 | 0.81 | 19.3s | 2.7s |
| 3 | 4 | 0.84 | 18.2s | 2.0s |
| 4 | 3 | 0.86 | 17.0s | 1.6s |

### 改造后（onnx_threads=4、workers=3）

- 未命中 8 并发 × 100：QPS 0.9、P50 8.6s、**P95 32s → 10.0s**；
- Prometheus 指标定位瓶颈：LLM 调用 P98 < 2s（DeepSeek 不是瓶颈），
  retrieval 95/101 落在 5~10s——瓶颈在本机 CPU 推理。

### 结论与下一步

- 异步化的价值在高并发与稳定性（P95 3 倍改善）；纯 CPU 推理下 QPS 天花板约 1，
  受限于混合架构 CPU 的 ONNX 并行度（~2×）与 bge-reranker-base 单次 ~0.7s 的推理成本；
- 下一步（按收益排序）：RTX 4060 GPU 推理（sentence-transformers / onnxruntime-gpu）；
  换更轻量重排模型或缩减重排候选数（需重跑 Recall 评测验证质量不回退）。

## 13. 多知识库注册中心（v0.6）：元数据与内容分层

主流企业知识库产品（Dify / FastGPT / RAGFlow）的三层模型：知识库 → 文档 → chunk。
chunk 本体（向量 + 文本）在向量库，kb_id / doc_id 血缘**同时写列与 metadata**；
知识库与文档的元数据（名字、状态、chunk 数、错误信息）在注册中心。
注册中心与向量库共用 `VECTOR_BACKEND` 开关：pgvector 模式 = 两张 PG 表（与 chunks
同库同连接池）；chroma 模式 = 标准库 sqlite3 单文件（data/registry.db），本地零依赖
不引入新服务。文件记录 doc_id 用 uuid 而非内容哈希：同一份文件允许进多个知识库。

## 14. chunk 血缘与删除闭环（v0.6）

删除的完整性靠血缘，不靠遍历：删文档 = `registry.delete_document`（元数据）+
`store.delete_by_doc`（按 doc_id 一条 SQL / 一个 where 删掉全部 chunk）；删知识库同理。
BM25 是进程内按 kb 惰性构建的，任何增删后 `retriever.invalidate(kb_id)` 使对应
索引失效，下次检索自动重建。老库迁移用 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`，
幂等可重跑。

## 15. 流式事件协议（v0.6）

SSE 三类事件固定顺序：`citations`（检索完成，含引用与分数）→ `token`（增量文本，
可多条）→ `done`（grounded / cached / retrieval_ms / latency_ms）。拒答与缓存命中
复用同一协议（citations → 单条 token → done），前端零特判；内部错误发 `error` 事件。
done 里的分阶段耗时延续了 v0.5 的可观测口径，流式不丢指标。

## 16. 多轮对话：condense 改写 + 缓存策略（v0.6）

追问（「那报销比例呢」）无法直接检索——先让 LLM 把「历史 + 追问」改写成独立问题
（condense），改写结果喂检索，生成 prompt 注入最近 6 轮历史。两处降级：condense
失败用原问题；多轮请求**完全绕过语义缓存**——缓存键是「问题 + 命中文档集」，不含
历史，同一问题不同上文语义不同，宁可不缓存也不返回脏答案。

## 17. 向量索引：为什么 ANN_INDEX 默认是 none（工单 09，实测）

直觉结论是"给 `chunks.embedding` 加个 HNSW 索引，检索就会快"。实测之后这个结论被
推翻了，而且推翻的方向很重要。

测量条件（`scripts/bench_ann.py`，产物 `data/bench/ann_report.json`）：
50000 行 × 512 维合成单位向量，均分 8 个知识库，100 条查询，单机 Docker 内的
pgvector/pg16，默认 shared_buffers。ground truth 用 numpy 在进程内精确算，
不复用被待测方案污染的 SQL 路径。

| 模式 | ef_search | p50 | p95 | recall@10 |
| --- | --- | --- | --- | --- |
| 顺序扫描（全局精确） | — | 107.4ms | 116.9ms | **1.000** |
| 顺序扫描 + `kb_id` 过滤 | — | **17.1ms** | 18.5ms | **1.000** |
| HNSW（无过滤） | 100 | 3.5ms | 7.0ms | 0.177 |
| HNSW（无过滤） | 1000 | 16.7ms | 19.4ms | 0.764 |
| HNSW + `kb_id` 过滤 | 100 | 14.8ms | 17.6ms | 1.000 |

三个结论：

1. **本服务的检索永远带 `WHERE kb_id = %s`**，而 `kb_id` 上的 btree 会先把候选缩到
   1/8 再排序取 top-k。这条路 17ms 且**完全精确** —— 在这个量级上 HNSW 一点便宜都
   没占到。加索引的收益要等到单个知识库内部行数再涨一个量级才可能出现。
2. **HNSW 在 pgvector 默认 `ef_search=100` 下把全局召回打到 0.177。** 对一个以引用
   溯源为卖点的系统，"快 30 倍但静默丢掉八成正确 chunk"不是优化，是故障 —— 而且它
   不报错、不改状态码，只在答案质量上悄悄体现，是最难发现的那类问题。
3. **`ef_search` 的合法上限是 1000**（超出直接报错），所以"把 ef 调大就能换回召回"
   这条路有天花板：1000 时也才 0.764。

因此 `ANN_INDEX` 默认 `none`。索引能力完整保留（`none` / `hnsw` / `ivfflat`，
建索引与 `ef_search` 参数都可配），但它是一个**需要你先复测召回才该打开的开关**，
不是默认加速器。

方法论上有一条更要紧的教训：基准脚本第一版把过滤查询的 ground truth 算成了全局
top-10，于是"顺序扫描 + 过滤"测出 recall 0.12 ≈ 1/8（正好是每个库占总语料的比例）。
那是**指标的错**，不是数据库的错 —— 如果照那个数字就会得出"精确扫描也在丢结果，
所以该上 HNSW"的结论，方向彻底相反。评一个近似算法，必须先证明精确基线在同一个
口径下是 1.0。
