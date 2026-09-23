"""chunk 主键迁移：md5(content) → doc_id:md5(content)（工单 03，带反向回滚）。

为什么需要它
------------
v0.6 及之前，`chunks.id` 就是 chunk 正文的 md5，同时它是 PRIMARY KEY，而写入语句是
`ON CONFLICT (id) DO UPDATE SET kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id`。
于是同一段文本第二次入库时，那一行不是新增而是**被改挂到新文档 / 新知识库名下**：
删掉任一相关文档都会物理带走另一个文档的 chunk，跨库上传会让原库内容静默消失。

本脚本把 id 改写为 `{doc_id}:{md5}`，并把内容哈希单独落到 content_hash 列，
使 ON CONFLICT 的语义收敛为「同一文档内去重」。

用法
----
    # 演练：只报告将要改多少行，不写
    .venv/Scripts/python scripts/migrate_chunk_id.py --dry-run

    # 正向迁移（可重复执行，幂等）
    .venv/Scripts/python scripts/migrate_chunk_id.py

    # 回滚
    .venv/Scripts/python scripts/migrate_chunk_id.py --reverse

反向迁移的一个硬约束
--------------------
历史上如果两份文档（或两个知识库）曾经共享过同一段内容，那么在旧方案下它们是**同一行**，
新方案下会变成**两行**。回滚要把两行压回一行，就会撞 PRIMARY KEY。
所以 --reverse 在真正动手前会先检测这种碰撞：存在即中止并给出清单，
需要人工决定保留哪一份归属。这个检查是有意为之——宁可停下，也不要静默丢内容。

只对 pgvector 后端有意义（Chroma 后端请用 scripts/ingest_docs.py 重灌）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def _connect():
    from psycopg import connect

    return connect(settings.postgres_dsn, autocommit=False)


def forward(conn, dry_run: bool) -> int:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)")
        cur.execute(
            "SELECT count(*) FROM chunks WHERE id NOT LIKE '%:\\%%' OR content_hash = ''"
        )
        pending = cur.fetchone()[0]
        print(f"[forward] 待迁移行数：{pending}")
        if dry_run or pending == 0:
            conn.rollback()
            return 0
        # 旧 id 本身就是 md5(content)，所以 content_hash 直接取旧 id
        cur.execute(
            "UPDATE chunks SET content_hash = id WHERE content_hash = ''"
        )
        cur.execute(
            "UPDATE chunks SET id = doc_id || ':' || content_hash"
            " WHERE id NOT LIKE '%:\\%%'"
        )
        moved = cur.rowcount
    conn.commit()
    print(f"[forward] 已改写 {moved} 行主键")
    return moved


def reverse(conn, dry_run: bool) -> int:
    with conn.cursor() as cur:
        # 碰撞检测：回滚后 (md5) 必须仍然唯一，否则 ON CONFLICT 的老问题会以
        # PRIMARY KEY violation 的形式炸在迁移中途
        cur.execute(
            "SELECT content_hash, count(*), array_agg(DISTINCT kb_id), array_agg(id)"
            " FROM chunks WHERE content_hash <> ''"
            " GROUP BY content_hash HAVING count(DISTINCT doc_id) > 1"
        )
        dupes = cur.fetchall()
        if dupes:
            print(f"[reverse] 中止：{len(dupes)} 段内容归属于多个文档，回滚会撞主键")
            for digest, n, kbs, ids in dupes[:10]:
                print(f"  content_hash={digest} 文档数={n} 知识库={kbs}")
            print("  请人工决定保留哪一份归属后再回滚；或保持当前（正确的）多行状态。")
            conn.rollback()
            return -1
        cur.execute("SELECT count(*) FROM chunks WHERE id LIKE '%:\\%%'")
        pending = cur.fetchone()[0]
        print(f"[reverse] 待回滚行数：{pending}")
        if dry_run or pending == 0:
            conn.rollback()
            return 0
        cur.execute(
            "UPDATE chunks SET id = content_hash WHERE id LIKE '%:\\%%'"
        )
        moved = cur.rowcount
    conn.commit()
    print(f"[reverse] 已回滚 {moved} 行主键")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description="chunk 主键迁移（工单 03）")
    parser.add_argument("--reverse", action="store_true", help="回滚到 md5(content) 主键")
    parser.add_argument("--dry-run", action="store_true", help="只统计将要改动的行数")
    args = parser.parse_args()

    if settings.vector_backend != "pgvector":
        print(f"当前 VECTOR_BACKEND={settings.vector_backend}，本脚本只处理 pgvector 后端。")
        print("本地 Chroma 模式请直接用 scripts/ingest_docs.py 重灌数据。")
        return 1

    conn = _connect()
    try:
        rc = reverse(conn, args.dry_run) if args.reverse else forward(conn, args.dry_run)
    finally:
        conn.close()
    return 1 if rc == -1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
