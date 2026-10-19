# 技术导览：设计、代码位置与验证方式

面试或代码审查时，可以从这里直接跳到实现。每一项都标注了要解决的问题、代码位置和可执行的验证方式。文中所写的限制都是当前实现真实存在的。

## 1. 双向量后端：同一套接口，同一套断言

要解决的问题是：本地开发希望零依赖，部署又需要 PostgreSQL 的事务与运维能力。如果两套实现各写一遍检索逻辑，换后端就等于换行为。

做法是：`VectorStore` 以鸭子类型定义接口，`ChromaStore` 与 `PGvectorStore` 各自实现。检索层只依赖 `search / count / delete_by_doc / delete_by_kb / get_all_documents`。pgvector 用余弦距离 `<=>`，换算成 `1 - distance`，与 Chroma 的相关度口径一致，因此同一个阈值在两边都成立。

代码在：`app/rag/vector_store.py`、`app/rag/registry.py`、`tests/contract/conftest.py`

验证方式是：`docker compose up -d` 后运行 `pytest tests -m contract -q`。两个后端跑同一组断言，包括 KB 隔离、级联删除、跨后端分数口径。CI 里缺少 PostgreSQL 会判失败，不会跳过。

代价是：跨引擎没有事务。Chroma + SQLite 模式下，摄取失败靠补偿把状态落成 `failed`；只有双 PG 后端能用真事务。契约测试断言的是可观察结果，不是有没有 `BEGIN`。

## 2. 混合检索：改写、BM25、向量、RRF、重排

要解决的问题是：纯向量检索对编号、专有名词这类字面匹配不敏感。项目早期「年假提前几天」的正确 chunk 只排到第二位。

做法是：查询改写只喂 BM25，向量检索用原问题，两路各取 top-20；用 RRF 融合排名，不比较原始分数；融合后的前 8 个候选交给 Cross-Encoder 重排，sigmoid 之后以 0.5 作为相关分界。BM25 用带倒排表的实现，查询只遍历包含查询词的 chunk，重建时复用正文分词缓存。BM25 按知识库惰性构建。

代码在：`app/rag/retriever.py`（`rrf_fuse`、`HybridRetriever.aretrieve`）、`app/rag/bm25_index.py`、`app/rag/reranker.py`、`app/rag/query_rewriter.py`

验证方式是：`pytest tests/test_retriever.py tests/test_bm25_index.py tests/test_reranker.py -q` 覆盖融合、截断与拒答分支。端到端召回用 `python scripts/eval_recall.py`，口径与限制见 [EVALUATION.md](EVALUATION.md)。

代价是：RRF 只看排名，省去了归一化与权重调参，代价是丢掉分数强度的信息。重排只处理 8 个候选，是精度与 CPU 成本的折中；本机单次重排约 0.7 秒，是检索阶段的主要耗时。

## 3. 两段式拒答

要解决的问题是：知识库外的提问如果直接进生成模型，会花掉一次调用，还容易编造。

做法是：检索前先看 BM25 是否零命中且向量最高分低于阈值；重排后再过滤一遍 0.5 以下的候选。任一阶段为空，直接返回固定拒答话术，不调用 LLM。重排器不可用时降级为不重排，并且不套用重排阈值，避免用占位的 0 分把结果清空。

代码在：`app/rag/retriever.py` 的 `_finalize`，`app/rag/generator.py` 的 `REFUSAL_ANSWER`

验证方式是：`tests/test_retriever.py` 覆盖两条拒答路径与降级分支；`tests/test_generator.py` 断言拒答时不触发 LLM。

代价是：拒答阈值是全局常数，没有按知识库或语料调优。缺少负例评测，误拒率未知，见 [EVALUATION.md](EVALUATION.md) 第 3.4 节。

## 4. 语义缓存与并发去重

要解决的问题是：重复问题会重复调用 LLM。知识库更新后，旧缓存不能继续返回旧答案。并发相同的请求还会同时打到模型。

做法是：问答缓存键是「问题 + 命中文档的内容哈希集合」。检索结果一变，键就变，旧缓存自然失效。多轮请求完全绕过缓存，因为键里没有历史。异步路径在缓存未命中后，用 `SingleFlight` 合并同一键的并发请求，并在真正调用模型前再查一次缓存。

代码在：`app/rag/generator.py`（`_cache_key`、`agenerate`）、`app/core/flight.py`、`app/core/cache.py`

验证方式是：`tests/test_wave2_hardening.py` 用计数 LLM 断言 8 个并发相同问题只调用 1 次；不同问题不会被合并；多轮请求既不缓存也不去重；leader 失败会广播给所有等待者。

代价是：缓存键不含模型版本。换生成模型后，旧答案会保留到 TTL 过期。`SingleFlight` 只在进程内生效，多进程部署时每个进程各有一份。

## 5. 异步管线与分池

要解决的问题是：同步端点等待 LLM 时会占用线程。CPU 密集的 embedding、重排、BM25 与元数据查询混在一个线程池里，会互相排队。

做法是：接口改成 `async def`，LLM 全链路用 `ainvoke`。CPU 计算、元数据读写、文档摄取分成三个有界线程池。关键词检索与向量检索两路并行，改写只影响关键词那一路。ONNX 推理线程数显式限制。

代码在：`app/api/chat.py`、`app/rag/pipeline.py`（`aask`、`astream`）、`app/rag/retriever.py`（`aretrieve`）、`app/core/executor.py`

验证方式是：`tests/test_executor.py`、`tests/test_wave2_hardening.py` 断言两条检索路径并行执行的总耗时接近最大值，而不是两者之和。

代价是：线池大小是经验值，绑定在本机 CPU 配置上。换机器需要重新压测，脚本是 `scripts/bench.py`。历史压测数字不可复现，见 [EVALUATION.md](EVALUATION.md) 第 5 节。

## 6. 多知识库、血缘与事务写入

要解决的问题是：删除文档或知识库时，元数据和向量内容必须一起消失。上传失败不能留下「有 chunk 数但没有内容」的记录。

做法是：三层模型是知识库、文档、chunk。chunk 主键是 `doc_id + 内容哈希`，同一段文本在不同文档或不同知识库里各自独立，避免互相改写归属。双 PG 后端让 `chunks` 与 `documents` 落在同一个连接、同一个事务里，并且先算完向量再拿连接，防止批量推理占满连接池。

代码在：`app/rag/vector_store.py`（`chunk_uid`、`transaction`、`add_documents_on`）、`app/rag/registry.py`、`app/rag/ingest.py`

验证方式是：`tests/contract/test_store_contract.py` 与 `test_registry_contract.py` 覆盖删除级联、跨库隔离、重复内容归属。`tests/test_ingest_script.py` 断言失败时只留一条 `failed` 记录。

代价是：Chroma + SQLite 模式没有分布式事务。摄取过程中的失败由调用方补偿，可能短暂出现元数据与向量不一致。

## 7. 跨进程检索缓存失效

要解决的问题是：BM25 索引按知识库放在进程内存里。异步摄取在 worker 进程完成，API 进程的索引不会自动知道内容变了，表现为「文档传上去了但搜不到，重启后恢复」。

做法是：每个知识库在 Redis 里维护一个版本号。写入方完成后自增，读取方在复用进程内索引前比对版本，不一致就重建。版本读取有一秒的本地轮询窗口，避免每个请求都多一次 Redis 往返。Redis 不可用时退回只信进程内失效通知。

代码在：`app/core/kb_version.py`、`app/rag/ingest.py`（`bump_kb_version`）、`app/rag/retriever.py`（`_bm25_for`、`invalidate`）

验证方式是：`tests/test_registry.py` 与 `tests/test_retriever.py` 覆盖版本变化触发重建。异步摄取的端到端路径由契约测试在真实 Redis 上验证。

代价是：变更可见性最多延迟一秒。知识库版本变化后仍要重新拉取该库全部 chunk 重建索引；分词有缓存，读取与建表仍是 O(N)。十万 chunk 级的落点仍是数据库侧倒排（PostgreSQL tsvector + GIN），当前没有做。

## 8. SSE 流式协议

要解决的问题是：流式回答一旦中途出错，客户端可能永远停在「正在生成」。代理的空闲超时也会掐断长时间没有输出的连接。

做法是：事件顺序固定为 `citations`、`token*`、`done`。拒答和缓存命中复用同一协议。空转时发 SSE 注释帧保活，不污染事件类型。出现异常时先发 `error`，再补一个 `done`，保证客户端只需要判断 `done` 就能收尾。客户端断开时取消上游任务，避免继续消耗模型 token。

代码在：`app/api/chat.py`（`_with_heartbeat`、`chat_stream`）、`app/rag/generator.py`（`astream_answer`）

验证方式是：`tests/test_stream.py` 覆盖事件顺序、拒答、缓存命中与错误终止。

代价是：流式路径不参与 `SingleFlight` 去重。缓存命中时仍然是一次性输出整段文本，不是逐 token。

## 9. 测试体系的切分

要解决的问题是：全部打桩的离线测试跑得很快，但看不见 SQL。项目曾经有一个 PostgreSQL 占位符错误带着全绿 CI 合进主分支。

做法是：测试分成两半。离线套件不连数据库、不下载模型，覆盖编排、缓存、拒答、流式与并发关系。契约套件启动真实 pgvector 与 Chroma，跑同一组断言。CI 里缺 PostgreSQL 直接失败。

代码在：`tests/`、`tests/contract/`、`.github/workflows/ci.yml`、`pytest.ini`

验证方式是：`pytest tests -m "not contract" -q` 与 `RAG_CONTRACT_STRICT=1 pytest tests -m contract -q`。

代价是：契约测试使用确定性伪向量，不验证真实模型的召回质量。模型冒烟标记为 `contract_model`，默认不跑，因为它需要下载约 1.2GB 的模型文件。

## 10. ANN 索引的实测结论

要解决的问题是：给 `chunks.embedding` 加 HNSW 看起来是显然的优化。

做法是：先测。50000 行、8 个知识库、100 条查询，对比顺序扫描与 HNSW 的延迟和召回，并记录每条测量实际走到的执行计划。结论是默认不建索引。

代码在：`scripts/bench_ann.py`、`data/bench/ann_report.json`、`app/config.py`（`ann_index`）、`app/rag/vector_store.py`（`_ensure_ann_index`）

验证方式是：`python scripts/bench_ann.py --rows 50000 --kbs 8 --queries 100`。报告里每个模式都带 `plan_node`，能区分「真的走了 HNSW」与「规划器改走精确计划」。

代价是：结论绑定在当前数据量与过滤选择率上。单个知识库内 chunk 数再涨一个量级时，需要复跑再决定。详见 [EVALUATION.md](EVALUATION.md) 第 4 节。
