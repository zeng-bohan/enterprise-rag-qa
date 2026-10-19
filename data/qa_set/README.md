# 评测数据与报告

| 文件 | 内容 |
| --- | --- |
| `qa_300.json` | 334 条合成问答，覆盖 80 个唯一 gold chunk；文件名沿用早期 300 条目标，实际以行数为准 |
| `refusal_set.json` | 20 条语料外问题，用于拒答率评测 |
| `recall_report.json` | 混合检索 Recall@k，2026-10-19 复测 |
| `recall_report_baseline.json` | 纯向量基线 Recall@k，同一批问题 |
| `ragas_report.json` | RAGAS 汇总，分层抽样 100 行 |
| `ragas_detail.csv` | 逐条分数，均值与汇总一致 |
| `refusal_report.json` | 拒答率与误拒率 |

生成与复现命令见仓库根目录的 `docs/EVALUATION.md`。所有数字都来自本仓库的实际运行；报告里的 `measured_at` 记录测量时间，样本组成、评审模型、抽样方式这些字段用于解释分数覆盖了哪部分语料。

`qa_300.json` 的问题是脚本用 chunk 正文生成的，检索难度低于真实提问。引用召回数字时应同时说明这一点。
