"""ANN 索引基准（工单 09）：顺序扫描 vs HNSW，以及带 kb_id 过滤时的真实代价。

为什么需要这个脚本
------------------
v0.6 之前 `chunks.embedding` 上没有任何近似索引，`ORDER BY embedding <=> $1` 是
全表顺序扫描。加了 HNSW 之后，"是否真的更快、召回掉没掉"不能靠断言 —— 尤其因为
检索语句还带着 `WHERE kb_id = %s`，而 pgvector 的 HNSW 是**先在图上取候选、再按
条件筛**：过滤选择性强时，候选会被剪断，结果可能是既没快多少、又漏掉了正确 chunk。

所以这里同时测两类指标：延迟（p50/p95）和**召回**（相对 numpy 精确算出的 top-k）。
只看延迟会得出"HNSW 可用"的错误结论。

ground truth 用 numpy 在进程内精确算，不复用被待测方案污染的 SQL 路径。

用法
----
    docker compose up -d
    .venv/Scripts/python scripts/bench_ann.py --rows 20000 --kbs 8 --queries 200

产物：data/bench/ann_report.json（含每种模式的延迟分位数、召回与 EXPLAIN 片段）。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from psycopg import connect
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

BENCH_KB_PREFIX = "bench__"
REPORT = ROOT / "data" / "bench" / "ann_report.json"


def synth(rows: int, dim: int, kbs: int):
    """确定性伪随机单位向量（seed 固定，便于复跑对比）。"""
    rng = np.random.default_rng(20260922)
    vecs = rng.standard_normal((rows, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    kb_of = np.array([f"{BENCH_KB_PREFIX}{i % kbs}" for i in range(rows)])
    return vecs, kb_of


def load(conn, vecs, kb_of, batch=2000):
    """COPY 灌数。逐行 INSERT 在这个量级上会让基准本身变成瓶颈。"""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # 注意 f-string 里的 '{{}}'：SQL 的 JSONB 默认值 '{}' 必须双写大括号，
    # 否则会被当成格式化字段（本脚本第一次跑就死在这里）
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL,"
        f" metadata JSONB NOT NULL DEFAULT '{{}}', embedding vector({settings.embed_dim}) NOT NULL,"
        f" kb_id TEXT NOT NULL DEFAULT '', doc_id TEXT NOT NULL DEFAULT '',"
        f" content_hash TEXT NOT NULL DEFAULT '')"
    )
    conn.execute("TRUNCATE TABLE chunks")
    with conn.cursor() as cur:
        for start in range(0, len(vecs), batch):
            part = vecs[start : start + batch]
            payload = [
                (
                    f"bench-{start + j}",
                    f"合成语料第 {start + j} 段",
                    "{}",
                    str(kb_of[start + j]),
                    "",
                    "",
                    "[" + ",".join(f"{x:.6f}" for x in part[j]) + "]",
                )
                for j in range(len(part))
            ]
            with cur.copy(
                "COPY chunks (id, content, metadata, kb_id, doc_id, content_hash, embedding)"
                " FROM STDIN"
            ) as cp:
                for row in payload:
                    cp.write_row(row)
    conn.execute("ANALYZE chunks")


def exact_topk(matrix, queries, k, mask=None):
    """numpy 精确余弦相似度（向量已归一，所以点积即余弦）。

    mask 很关键：带 `WHERE kb_id = %s` 的查询，其正确 top-k 只在**该知识库内**排名。
    拿全局 top-k 当 ground truth 去评过滤查询，会把「本来就不属于这个库的 chunk」
    算成漏召回 —— 本脚本第一版就犯了这个错，测出 seq_scan 过滤后 recall 只有 0.12
    （≈1/8，正好是每个库占总语料的比例），那是指标的错，不是数据库的错。
    """
    sims = queries @ matrix.T
    if mask is not None:
        sims = np.where(mask[None, :], sims, -np.inf)
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def measure(conn, queries, truth_ids, mode, kb, ef, k=10, enable_index=True):
    # SET 不接受绑定参数（Postgres 语法），只能用字面量；这里的值全部来自本脚本内部，
    # 不来自任何用户输入
    flag = "on" if enable_index else "off"
    conn.execute(f"SET enable_indexscan = {flag}")
    conn.execute(f"SET enable_bitmapscan = {flag}")
    if mode.startswith("hnsw"):
        conn.execute(f"SET hnsw.ef_search = {int(ef)}")
    lat, hits_at = [], []
    for q in queries:
        vec = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
        # 绑定顺序必须与 SQL 里占位符的出现顺序一致：过滤分支中 kb_id 先于向量
        if kb is None:
            sql = "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s"
            params = [vec, k]
        else:
            sql = "SELECT id FROM chunks WHERE kb_id = %s ORDER BY embedding <=> %s::vector LIMIT %s"
            params = [kb, vec, k]
        t0 = time.perf_counter()
        got = [r[0] for r in conn.execute(sql, params).fetchall()]
        lat.append(time.perf_counter() - t0)
        want = truth_ids[len(hits_at)]
        hits_at.append(len(set(got) & set(want)) / float(k))
    row = {
        "mode": mode,
        "kb_filtered": kb is not None,
        "ef_search": ef if mode.startswith("hnsw") else None,
        "p50_ms": round(statistics.median(lat) * 1000, 2),
        "p95_ms": round(statistics.quantiles(lat, n=20)[18] * 1000, 2),
        "recall@10": round(statistics.mean(hits_at), 4),
    }
    print(f"  {row['mode']:<20} ef={str(row['ef_search']):>5} p50={row['p50_ms']:>7}ms "
          f"p95={row['p95_ms']:>7}ms recall@10={row['recall@10']}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="pgvector ANN 基准")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--kbs", type=int, default=8)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--keep", action="store_true", help="保留合成语料，便于手动 EXPLAIN")
    args = ap.parse_args()

    dim = settings.embed_dim
    rng = np.random.default_rng(7)
    vecs, kb_of = synth(args.rows, dim, args.kbs)
    q = rng.standard_normal((args.queries, dim), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    with connect(settings.postgres_dsn, autocommit=True) as conn:
        register_vector(conn)
        print(f"[bench] 灌入 {args.rows} 行 × {dim} 维，{args.kbs} 个知识库…")
        t0 = time.perf_counter()
        load(conn, vecs, kb_of)
        print(f"[bench] 灌入耗时 {time.perf_counter() - t0:.1f}s")

        truth_all = exact_topk(vecs, q, 10)
        truth_ids_all = [[f"bench-{i}" for i in row] for row in truth_all]
        probe_kb = f"{BENCH_KB_PREFIX}0"
        mask = kb_of == probe_kb
        print(f"[bench] 探针知识库 {probe_kb} 含 {int(mask.sum())} 行（占 {mask.mean():.1%}）")
        truth_ids_kb = [[f"bench-{i}" for i in row] for row in exact_topk(vecs, q, 10, mask=mask)]

        results = []
        print("[bench] 顺序扫描（无索引基线，全局精确）…")
        results.append(measure(conn, q, truth_ids_all, "seq_scan", None, None, enable_index=False))
        print("[bench] 顺序扫描 + kb 过滤（库内精确，这是过滤查询的召回上限）…")
        results.append(measure(conn, q, truth_ids_kb, "seq_scan_filtered", probe_kb, None, enable_index=False))

        print("[bench] 建 HNSW 索引…")
        t0 = time.perf_counter()
        conn.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
        conn.execute(
            f"CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
            f" WITH (m = {settings.hnsw_m}, ef_construction = {settings.hnsw_ef_construction})"
        )
        build_s = round(time.perf_counter() - t0, 1)
        print(f"[bench] 建索引耗时 {build_s}s")

        # ef_search 的合法区间是 1..1000（pgvector 0.7 的硬上限，超出直接报错），
        # 所以「靠无限加大 ef 换回召回」这条路是有天花板的 —— 这本身就是基准的结论之一
        for ef in (40, settings.hnsw_ef_search, 200, 400, 800, 1000):
            results.append(measure(conn, q, truth_ids_all, "hnsw", None, ef))
            results.append(measure(conn, q, truth_ids_kb, "hnsw_filtered", probe_kb, ef))

        plan = [r[0] for r in conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM chunks WHERE kb_id = 'bench__0'"
            " ORDER BY embedding <=> (SELECT embedding FROM chunks LIMIT 1)::vector LIMIT 10"
        ).fetchall()]

    report = {
        "conditions": {
            "rows": args.rows,
            "dim": dim,
            "knowledge_bases": args.kbs,
            "queries": args.queries,
            "kb_filter_selectivity": round(1 / args.kbs, 4),
            "hnsw_m": settings.hnsw_m,
            "hnsw_ef_construction": settings.hnsw_ef_construction,
            "index_build_seconds": build_s,
            "pg_settings": settings.postgres_dsn.split("@")[-1],
            "note": "单机 Docker 容器内测量；shared_buffers 为镜像默认值。",
        },
        "results": results,
        "explain_sample": plan,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'mode':<20}{'ef':>6}{'p50ms':>9}{'p95ms':>9}{'recall@10':>11}")
    for r in results:
        print(f"{r['mode']:<20}{str(r['ef_search']):>6}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['recall@10']:>11}")
    print(f"\n[bench] 报告已写入 {REPORT}")

    if not args.keep:
        with connect(settings.postgres_dsn, autocommit=True) as conn:
            conn.execute("TRUNCATE TABLE chunks")
            conn.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
        print("[bench] 已清空合成语料与索引（--keep 可保留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
