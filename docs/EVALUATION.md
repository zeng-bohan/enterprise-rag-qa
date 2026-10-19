# 评测方法与可信度说明

这个仓库里所有对外引用的数字，都能在 `data/` 下找到对应报告并重新生成。本文说明每个数字是怎么测出来的、能支持什么结论、不能支持什么结论。

结论先放在这里：检索召回和 ANN 基准可以在本地复现；RAGAS 分数是真实测量，但评测子集只覆盖一个源文档的 24 个 chunk，评审模型与生成模型同族，因此它是「这一段语料上的表现」，不能外推成系统整体质量；延迟压测的历史数字没有留存原始产物，只能当作方向性证据。

## 1. 评测数据

`data/qa_set/qa_300.json` 由 `scripts/gen_qa_set.py` 生成：对 `data/docs` 下每个切分后的 chunk 调用 LLM 生成 3 到 5 个问答对，答案要求只来自该 chunk，gold 标签取该 chunk 正文的内容哈希。

当前文件的实际组成：

| 项 | 数值 |
| --- | --- |
| QA 行数 | 334 |
| 唯一问题文本 | 333（有 1 条重复） |
| 唯一 gold chunk | 80 |
| 覆盖文档 | 8 篇 |
| 每 chunk 平均问题数 | 4.2 |

两个直接后果：

1. 行级指标被「问题多的 chunk」加权。一个 chunk 生出 5 个问题，它在行级指标里的权重就是另一个只生出 3 个问题的 chunk 的 1.67 倍。
2. 334 行不是 334 次独立试验。有效样本接近 80 个 chunk，置信区间的宽度应按 80 量级估算，不能按 334 估算。

`scripts/eval_recall.py` 现在同时输出行级 `recall@k` 和 chunk 级宏平均 `recall@k_chunk_macro`，并把数据组成写进 `recall_report.json`。历史报告只记录了行级结果，`recall_report.json` 目前仍是行级口径，未重新运行，见第 3 节。

## 2. 指标定义

### 2.1 Recall@k

对每个问题检索 top-5，gold chunk 出现在第 k 位以内记一次命中。

- 行级：命中行数 / 总行数。
- chunk 级宏平均：先算每个 gold chunk 自身的问题命中率，再对 80 个 chunk 取平均。

两个口径回答不同问题。行级反映「随机抽一个问题」的期望；chunk 级反映「随机抽一段知识」的期望。差异大时说明评测集在若干 chunk 上过采样。

仓库现有报告：混合检索 Recall@1 87.4%、Recall@3 97.9%、Recall@5 98.2%（n=334 行）。这是行级口径。

### 2.2 RAGAS Faithfulness 与 Answer Relevancy

| 指标 | 定义 | 本仓库的实现 |
| --- | --- | --- |
| Faithfulness | 答案中每个可验证论断能否在检索上下文中找到依据 | ragas 0.4 collections 指标，上下文用完整 chunk |
| Answer Relevancy | 答案是否切题 | 同上，语义相似度加反向问题生成 |
| 幻觉率 | 1 − Faithfulness | 由前者换算，不是独立测量 |

当前报告：Faithfulness 94.1%、Answer Relevancy 88.7%（n=100）。逐条分数保存在 `data/qa_set/ragas_detail.csv`，可以检查是少数低分样本拉低了均值，还是分数普遍偏高。

## 3. 已知偏差与限制

### 3.1 RAGAS 样本只覆盖一个文档

`scripts/eval_ragas.py` 的默认口径是取 QA 集前 100 条。QA 集按 `data/docs` 的文件顺序生成，前 100 条全部来自《中华人民共和国劳动合同法》，覆盖 24 个唯一 gold chunk。

所以 94.1% 的含义是「在一个法规文档的 24 段文本上，答案的可溯源性为 94.1%」。它不能说明员工手册、报销制度、PDF 文档上的表现。新加的 `--sample stratified` 会按来源文档轮流抽样，扩大覆盖，但历史报告没有用它，重新评测需要 API Key 与费用。

### 3.2 评审模型与生成模型同族

生成回答用 DeepSeek，RAGAS 判决也用 DeepSeek。自评偏差在这个配置下无法排除。要排除，需要换一个独立厂商的模型做评审，或者人工标注一部分样本。仓库没有做这两件事。

### 3.3 评测集存在结构性泄漏

问题是由 chunk 正文生成的，答案也要求只来自该 chunk。这样一来问题措辞天然贴近原文，检索难度低于真实用户提问。所有基于该评测集的召回数字都偏乐观。

这不是可以通过更换抽样方式解决的，需要在文档侧另做一套人工提问集。

### 3.4 缺少负例

评测集只有「有答案的问题」，没有「知识库答不了的问题」。因此没有拒答率、误拒率这类指标。系统实现了两段式拒答，但只有单元测试覆盖逻辑，没有量化评测。

### 3.5 没有置信区间

80 个独立样本、命中率 87.4% 附近，95% 区间大约有 ±7 个百分点。报告给的是点估计，没有区间。读数字时应按这个量级理解。

## 4. ANN 索引基准

脚本 `scripts/bench_ann.py`，报告 `data/bench/ann_report.json`。测量条件：50000 行、512 维合成单位向量、8 个知识库、100 条查询、单机 Docker 内的 pgvector/pg16、默认 shared_buffers。ground truth 由 numpy 在进程内精确计算。

2026-10-19 用同一组参数复跑，召回数字与仓库里的历史报告一致。复跑同时暴露了原报告的一个口径问题，已在脚本和报告里补上 `plan_node` 字段。

| 模式 | ef_search | p50 | recall@10 | 实际执行计划 |
| --- | --- | --- | --- | --- |
| 顺序扫描，全局 | 无 | 99.8ms | 1.000 | Sort |
| 顺序扫描，带 kb 过滤 | 无 | 16.1ms | 1.000 | Sort |
| HNSW，无过滤 | 100 | 3.3ms | 0.182 | Index Scan |
| HNSW，带 kb 过滤 | 40 | 13.4ms | 1.000 | **Sort（未走 HNSW）** |
| HNSW，带 kb 过滤 | 100 | 13.9ms | 1.000 | **Sort（未走 HNSW）** |
| HNSW，带 kb 过滤 | 200 | 4.8ms | 0.285 | Index Scan |
| HNSW，带 kb 过滤 | 1000 | 15.8ms | 0.739 | Index Scan |

读法：

1. `ef_search` 较低时，规划器放弃了 HNSW，改成按 `kb_id` 过滤后的精确排序。所以「带过滤的 HNSW 在 ef=100 时召回 1.0」其实是精确计划的成绩，不能算作 HNSW 的功劳。原设计文档按 HNSW 解释这一行，是误读。
2. 真正走 HNSW 且带过滤时，召回随 `ef_search` 上升，但到合法上限 1000 也只有 0.739。过滤条件会剪断图上的候选，这个代价无法靠调大 ef 完全补回。
3. 本服务的量级上，带 `kb_id` 过滤的顺序扫描 16ms 且完全精确。因此 `ANN_INDEX` 默认 `none`。这个结论与最初的设计预期相反，但复跑两次都成立。

基准脚本第一版还犯过一个指标错误：过滤查询的 ground truth 用了全局 top-10，于是精确扫描测出召回 0.12，约等于 1/8，正好是单库占全部语料的比例。修正方法是为过滤查询单独计算库内真值。

## 5. 延迟压测

`docs/DESIGN.md` 第 12 节记录了一组历史压测：8 并发下 P95 从 32s 降到 10s。这组数字的来源是 `scripts/bench.py` 加一次真实服务运行，但仓库里没有保存原始输出、环境快照或对应的服务版本，因此无法从仓库内复现。

能复现的部分：`scripts/bench.py` 的压测方法与指标口径；异步改造的代码路径有单元测试覆盖。

不能复现的部分：具体秒数。它依赖本机 CPU、模型缓存是否预热、DeepSeek 当时的响应速度。引用这组数字时应当说明是历史测量，不是当前可复现的基准。

## 6. 复现方式

```bash
# 离线测试，不联网、不下载模型
pip install -r requirements.txt -r requirements-dev.txt
pytest tests -m "not contract" -q

# 真实后端契约测试（需要 Docker）
docker compose up -d
RAG_CONTRACT_STRICT=1 pytest tests -m contract -q

# 检索召回（需要先灌库，且模型可下载）
python scripts/ingest_docs.py
python scripts/eval_recall.py
python scripts/eval_recall.py --baseline

# RAGAS（需要 API Key，产生费用）
pip install -r requirements-eval.txt
python scripts/eval_ragas.py --sample stratified

# ANN 基准（会清空 chunks 表，只在测试库运行）
python scripts/bench_ann.py --rows 50000 --kbs 8 --queries 100
```

2026-10 的审计环境可以运行前两组测试与 ANN 基准，但无法访问 HuggingFace，嵌入模型下载超时，因此 `ingest_docs.py`、`eval_recall.py` 与 RAGAS 没有在该环境重新执行。这类失败应如实记录为「未验证」，不能算作通过。

## 7. 结论分级

| 结论 | 等级 | 依据 |
| --- | --- | --- |
| 双后端行为一致、级联删除正确 | 已验证 | 42 个契约用例，CI 中缺少 PostgreSQL 即失败 |
| 离线测试不依赖外部服务 | 已验证 | 127 个离线用例，6 秒内跑完 |
| 带 `kb_id` 过滤的精确扫描在本量级够用 | 已验证 | ANN 基准复跑两次 |
| 混合检索 Recall@1 87.4%（行级） | 已测量，口径有限 | 报告存在；评测集有 3.3 与 3.4 的偏差 |
| Faithfulness 94.1% | 已测量，口径有限 | 单文档子集、同族评审，不可外推 |
| P95 32s→10s | 历史测量，不可复现 | 无原始产物 |
