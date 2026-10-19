# 评测方法与可信度说明

这个仓库里对外引用的数字，都能在 `data/` 下找到对应报告并重新生成。本文说明每个数字怎么测出来、能支持什么结论、不能支持什么结论。

2026-10-19 在本地重跑了检索、拒答与 RAGAS 三组评测，报告已更新。运行环境：Windows 11、Python 3.12.10、`VECTOR_BACKEND=chroma`（本地 Chroma + SQLite 注册中心）、BGE 嵌入与 bge-reranker-base 重排通过 FastEmbed 在本机 ONNX 推理。生成与评审都调用 DeepSeek API，请求的模型名是 `deepseek-chat`。

## 1. 评测数据

`data/qa_set/qa_300.json` 由 `scripts/gen_qa_set.py` 生成：对 `data/docs` 下每个切分后的 chunk 调用 LLM 生成 3 到 5 个问答对，答案要求只来自该 chunk，gold 标签取该 chunk 正文的内容哈希。

当前文件的实际组成：

| 项 | 数值 |
| --- | --- |
| QA 行数 | 334 |
| 唯一问题文本 | 333（有 1 条重复） |
| 唯一 gold chunk | 80 |
| 覆盖文档 | 8 篇 |
| 每 chunk 平均问题数 | 4.17 |

两个直接后果：

1. 行级指标被「问题多的 chunk」加权。一个 chunk 生出 5 个问题，它在行级指标里的权重就是另一个只生出 3 个问题的 chunk 的 1.67 倍。
2. 334 行不是 334 次独立试验。有效样本接近 80 个 chunk，置信区间的宽度应按 80 量级估算。

`scripts/eval_recall.py` 同时输出行级 `recall@k` 和 chunk 级宏平均 `recall@k_chunk_macro`，并把数据组成写进报告。两个口径在本次结果里只差 0.1 个百分点左右，说明重复采样没有造成明显偏斜。

## 2. 指标与本次结果

### 2.1 Recall@k

对每个问题检索 top-5，gold chunk 出现在第 k 位以内记一次命中。行级口径为命中行数除以总行数；chunk 级宏平均先算每个 gold chunk 自身的问题命中率，再对 80 个 chunk 取平均。

| 指标 | 混合检索 | 纯向量基线 | 差值 |
| --- | --- | --- | --- |
| Recall@1 | **87.43%** | 66.17% | +21.26pp |
| Recall@3 | **97.01%** | 88.92% | +8.09pp |
| Recall@5 | **97.90%** | 95.21% | +2.69pp |
| Recall@1（chunk 宏平均） | 87.31% | 66.44% | +20.87pp |

报告：`data/qa_set/recall_report.json` 与 `recall_report_baseline.json`。上一版报告只记录了混合检索，基线没有落盘；这次把两条路径都存了下来。

混合检索包含查询改写、BM25 与向量双路召回、RRF 融合、Cross-Encoder 重排。基线只做向量检索，用来回答「混合链路到底带来了多少提升」。

### 2.2 RAGAS Faithfulness 与 Answer Relevancy

| 指标 | 结果 | 定义 |
| --- | --- | --- |
| Faithfulness | **93.27%** | 答案中的可验证论断能否在检索上下文中找到依据 |
| Answer Relevancy | **89.29%** | 答案是否切题 |
| 幻觉率 | 6.73% | 1 − Faithfulness，由前者换算 |

报告：`data/qa_set/ragas_report.json`，逐条分数在 `ragas_detail.csv`，明细均值与汇总值已核对一致。

本次样本为按来源文档分层的 100 条，覆盖 8 篇文档、25 个唯一 gold chunk。上一版是 `qa_set[:100]`，只覆盖《劳动合同法》一篇文档、24 个 chunk。分层抽样降低了单一文档偏差，但没有消除评测集本身的结构性泄漏。

有 2 条样本的检索结果为空（系统拒答），无法做 Faithfulness 判定，按 0 分计入分母。把这类样本直接删掉会抬高平均分，因此保留并单独计数。

### 2.3 拒答率与误拒率

`scripts/eval_refusal.py` 用两组问题测两段式拒答：

- 负例：`data/qa_set/refusal_set.json` 里 20 个语料答不上的问题（年会、宿舍、期权等）。
- 正例：从 QA 集分层抽 50 条，语料可以回答。

| 指标 | 结果 |
| --- | --- |
| 负例拒答率 | **90%**（18/20） |
| 正例误拒率 | **4%**（2/50） |

报告：`data/qa_set/refusal_report.json`。有一道负例被回答了：离职证明开具时限。语料里没有这条规定，说明当前阈值下仍存在漏拒。负例集只有 20 条，这个比例适合看趋势，不适合当作精确指标。

## 3. 已知偏差与限制

### 3.1 评测集存在结构性泄漏

问题由 chunk 正文生成，答案也要求只来自该 chunk。问题措辞天然贴近原文，检索难度低于真实用户提问。所有基于该评测集的召回数字都偏乐观。这一点无法靠换抽样方式解决，需要在文档外另建人工提问集。

### 3.2 评审模型与生成模型同族

生成和判决都走 DeepSeek。自评偏差在这个配置下无法排除。要排除，需要换一个独立厂商的模型做评审，或者人工标注一部分样本。本次运行中，API 请求的模型是 `deepseek-chat`，服务端响应里标的模型是 `deepseek-flash`，生成与评审拿到的是同一路由。

### 3.3 没有置信区间

80 个独立样本、命中率 87.43% 附近，95% 区间大约有 ±7 个百分点。报告给的是点估计，没有区间。

### 3.4 拒答评测规模有限

20 条负例只覆盖了常见缺口类型，不能代表所有越界提问。阈值是按当前语料定的，换语料应重跑 `eval_refusal.py` 再调。

### 3.5 延迟数字不可复现

`docs/DESIGN.md` 第 12 节记录的历史压测（P95 32s → 10s）没有保留原始输出与环境快照，不能从仓库内复现。能复现的是压测方法与代码路径，具体秒数依赖当时机器与模型服务状态。

## 4. ANN 索引基准

脚本 `scripts/bench_ann.py`，报告 `data/bench/ann_report.json`。测量条件：50000 行、512 维合成单位向量、8 个知识库、100 条查询、单机 Docker 内的 pgvector/pg16、默认 shared_buffers。ground truth 由 numpy 在进程内精确计算。

2026-10-19 用同一组参数复跑，召回数字与历史报告一致。复跑同时暴露了原报告的一个口径问题，已在脚本和报告里补上 `plan_node` 字段。

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

1. `ef_search` 较低时，规划器放弃了 HNSW，改成按 `kb_id` 过滤后的精确排序。所以「带过滤的 HNSW 在 ef=100 时召回 1.0」其实是精确计划的成绩，不能算作 HNSW 的功劳。
2. 真正走 HNSW 且带过滤时，召回随 `ef_search` 上升，但到合法上限 1000 也只有 0.739。过滤条件会剪断图上的候选，这个代价无法靠调大 ef 完全补回。
3. 本服务的量级上，带 `kb_id` 过滤的顺序扫描 16ms 且完全精确，因此 `ANN_INDEX` 默认 `none`。

基准脚本第一版还犯过一个指标错误：过滤查询的 ground truth 用了全局 top-10，于是精确扫描测出召回 0.12，约等于 1/8，正好是单库占全部语料的比例。修正方法是为过滤查询单独计算库内真值。

## 5. 复现方式

```bash
# 离线测试，不联网、不下载模型
pip install -r requirements.txt -r requirements-dev.txt
pytest tests -m "not contract" -q

# 真实后端契约测试（需要 Docker）
docker compose up -d
RAG_CONTRACT_STRICT=1 pytest tests -m contract -q

# 检索召回（需要先灌库，且模型可下载或已缓存）
python scripts/ingest_docs.py
python scripts/eval_recall.py
python scripts/eval_recall.py --baseline

# 拒答率（需要 LLM Key）
python scripts/eval_refusal.py --positives 50

# RAGAS（需要 API Key 与评测依赖，产生费用）
pip install -r requirements-eval.txt
python scripts/eval_ragas.py --sample stratified

# ANN 基准（会清空 chunks 表，只在测试库运行）
python scripts/bench_ann.py --rows 50000 --kbs 8 --queries 100
```

本次复现的模型文件来自 ModelScope 的 `BAAI/bge-small-zh-v1.5` 与 `BAAI/bge-reranker-base` 权重，本地导出 ONNX 后放入 FastEmbed 缓存。模型来源不影响指标定义，但说明这次没有依赖 HuggingFace 下载。

## 6. 结论分级

| 结论 | 等级 | 依据 |
| --- | --- | --- |
| 双后端行为一致、级联删除正确 | 已验证 | 42 个契约用例，CI 中缺少 PostgreSQL 即失败 |
| 离线测试不依赖外部服务 | 已验证 | 134 个离线用例，约 10 秒跑完 |
| 带 `kb_id` 过滤的精确扫描在本量级够用 | 已验证 | ANN 基准复跑两次 |
| 混合检索优于纯向量基线 | 已验证（合成评测集） | Recall@1 87.43% vs 66.17% |
| 拒答率 90%、误拒率 4% | 已测量，样本有限 | 20 条负例、50 条正例 |
| Faithfulness 93.27% | 已测量，口径有限 | 分层子集，同族评审，不可外推 |
| P95 32s→10s | 历史测量，不可复现 | 无原始产物 |
