"""知识库 / 文档注册中心：企业知识库管理的元数据层。

主流企业知识库产品（Dify / FastGPT / RAGFlow / Glean）的三层模型：
  知识库 KB → 文档 Document → chunk（向量库内容）
chunk 本体在向量库（含 kb_id / doc_id 血缘元数据），本模块管前两层的元数据：

- create_kb / list_kbs / delete_kb          知识库生命周期
- add_document / list_documents / get_document / delete_document
                                            文档生命周期（status: indexed / failed）

双实现与向量库后端共用同一个开关（VECTOR_BACKEND）：
- pgvector（生产）：kbs / documents 两张表，与 chunks 同库同事务生态；
- chroma（本地零依赖）：标准库 sqlite3 单文件（data/registry.db），不引入新服务。

文档删除的完整语义（两步，由 API 层编排）：
  registry.delete_document（元数据） + store.delete_by_doc（chunk 血缘清理）。
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.config import settings


def new_doc_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KBError(RuntimeError):
    """知识库不存在 / 重名等业务错误，message 面向 API 直接返回。"""


class KBRegistry:
    """注册中心抽象接口（同步方法：元数据操作快，经 run_cpu 进线程池即可）。"""

    def ensure_default(self) -> str:
        raise NotImplementedError

    def create_kb(self, name: str, description: str = "") -> dict:
        raise NotImplementedError

    def list_kbs(self) -> List[dict]:
        raise NotImplementedError

    def get_kb(self, kb_id: str) -> dict:
        raise NotImplementedError

    def kb_id_by_name(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def delete_kb(self, kb_id: str) -> None:
        raise NotImplementedError

    def add_document(
        self, kb_id: str, filename: str, chunk_count: int, status: str = "indexed", error: str = ""
    ) -> dict:
        raise NotImplementedError

    def list_documents(self, kb_id: str) -> List[dict]:
        raise NotImplementedError

    def get_document(self, kb_id: str, doc_id: str) -> dict:
        raise NotImplementedError

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        raise NotImplementedError

    def delete_documents_of_kb(self, kb_id: str) -> None:
        raise NotImplementedError


class SQLiteKBRegistry(KBRegistry):
    """本地模式：单文件 SQLite（stdlib，无新依赖）。"""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kbs (
                kb_id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id)")
        self._conn.commit()

    def ensure_default(self) -> str:
        kb_id = self.kb_id_by_name(settings.default_kb)
        if kb_id:
            return kb_id
        return self.create_kb(settings.default_kb, "默认知识库（脚本入库 / 单库问答）")["kb_id"]

    def create_kb(self, name: str, description: str = "") -> dict:
        kb_id = uuid.uuid4().hex[:12]
        try:
            self._conn.execute(
                "INSERT INTO kbs (kb_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (kb_id, name, description, _now()),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise KBError(f"知识库名已存在: {name}") from exc
        return {"kb_id": kb_id, "name": name, "description": description}

    def list_kbs(self) -> List[dict]:
        rows = self._conn.execute("SELECT kb_id, name, description, created_at FROM kbs ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def get_kb(self, kb_id: str) -> dict:
        row = self._conn.execute("SELECT kb_id, name, description, created_at FROM kbs WHERE kb_id = ?", (kb_id,)).fetchone()
        if row is None:
            raise KBError(f"知识库不存在: {kb_id}")
        return dict(row)

    def kb_id_by_name(self, name: str) -> Optional[str]:
        row = self._conn.execute("SELECT kb_id FROM kbs WHERE name = ?", (name,)).fetchone()
        return row["kb_id"] if row else None

    def delete_kb(self, kb_id: str) -> None:
        self.get_kb(kb_id)  # 不存在则抛 KBError
        self._conn.execute("DELETE FROM documents WHERE kb_id = ?", (kb_id,))
        self._conn.execute("DELETE FROM kbs WHERE kb_id = ?", (kb_id,))
        self._conn.commit()

    def add_document(
        self, kb_id: str, filename: str, chunk_count: int, status: str = "indexed", error: str = ""
    ) -> dict:
        self.get_kb(kb_id)
        doc_id = new_doc_id()
        now = _now()
        self._conn.execute(
            "INSERT INTO documents (doc_id, kb_id, filename, status, chunk_count, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, kb_id, filename, status, chunk_count, error, now, now),
        )
        self._conn.commit()
        return {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": filename,
            "status": status,
            "chunk_count": chunk_count,
            "error": error,
        }

    def list_documents(self, kb_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
            " FROM documents WHERE kb_id = ? ORDER BY created_at",
            (kb_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, kb_id: str, doc_id: str) -> dict:
        row = self._conn.execute(
            "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
            " FROM documents WHERE kb_id = ? AND doc_id = ?",
            (kb_id, doc_id),
        ).fetchone()
        if row is None:
            raise KBError(f"文档不存在: {doc_id}")
        return dict(row)

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self.get_document(kb_id, doc_id)
        self._conn.execute("DELETE FROM documents WHERE kb_id = ? AND doc_id = ?", (kb_id, doc_id))
        self._conn.commit()

    def delete_documents_of_kb(self, kb_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE kb_id = ?", (kb_id,))
        self._conn.commit()


class PGKBRegistry(KBRegistry):
    """生产模式：PostgreSQL（与 chunks 同库，连接池复用 store 的配置）。"""

    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=1, max_size=4, kwargs={"autocommit": True}, open=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kbs (
                    kb_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'indexed',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id)")

    def ensure_default(self) -> str:
        kb_id = self.kb_id_by_name(settings.default_kb)
        if kb_id:
            return kb_id
        return self.create_kb(settings.default_kb, "默认知识库（脚本入库 / 单库问答）")["kb_id"]

    def create_kb(self, name: str, description: str = "") -> dict:
        kb_id = uuid.uuid4().hex[:12]
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "INSERT INTO kbs (kb_id, name, description) VALUES (?, ?, ?)"
                    " RETURNING kb_id, name, description",
                    (kb_id, name, description),
                ).fetchone()
        except Exception as exc:  # unique_violation 等
            raise KBError(f"知识库创建失败（重名？）: {name}") from exc
        return {"kb_id": row[0], "name": row[1], "description": row[2]}

    def list_kbs(self) -> List[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT kb_id, name, description FROM kbs ORDER BY created_at").fetchall()
        return [{"kb_id": r[0], "name": r[1], "description": r[2]} for r in rows]

    def get_kb(self, kb_id: str) -> dict:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT kb_id, name, description FROM kbs WHERE kb_id = %s", (kb_id,)
            ).fetchone()
        if row is None:
            raise KBError(f"知识库不存在: {kb_id}")
        return {"kb_id": row[0], "name": row[1], "description": row[2]}

    def kb_id_by_name(self, name: str) -> Optional[str]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT kb_id FROM kbs WHERE name = %s", (name,)).fetchone()
        return row[0] if row else None

    def delete_kb(self, kb_id: str) -> None:
        self.get_kb(kb_id)
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s", (kb_id,))
            conn.execute("DELETE FROM kbs WHERE kb_id = %s", (kb_id,))

    def add_document(
        self, kb_id: str, filename: str, chunk_count: int, status: str = "indexed", error: str = ""
    ) -> dict:
        self.get_kb(kb_id)
        doc_id = new_doc_id()
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO documents (doc_id, kb_id, filename, status, chunk_count, error)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " RETURNING doc_id, filename, status, chunk_count, error",
                (doc_id, kb_id, filename, status, chunk_count, error),
            ).fetchone()
        return {
            "doc_id": row[0],
            "kb_id": kb_id,
            "filename": row[1],
            "status": row[2],
            "chunk_count": row[3],
            "error": row[4],
        }

    def list_documents(self, kb_id: str) -> List[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, status, chunk_count, error FROM documents"
                " WHERE kb_id = %s ORDER BY created_at",
                (kb_id,),
            ).fetchall()
        return [
            {"doc_id": r[0], "kb_id": kb_id, "filename": r[1], "status": r[2], "chunk_count": r[3], "error": r[4]}
            for r in rows
        ]

    def get_document(self, kb_id: str, doc_id: str) -> dict:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT doc_id, filename, status, chunk_count, error FROM documents"
                " WHERE kb_id = %s AND doc_id = %s",
                (kb_id, doc_id),
            ).fetchone()
        if row is None:
            raise KBError(f"文档不存在: {doc_id}")
        return {"doc_id": row[0], "kb_id": kb_id, "filename": row[1], "status": row[2], "chunk_count": row[3], "error": row[4]}

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self.get_document(kb_id, doc_id)
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s AND doc_id = %s", (kb_id, doc_id))

    def delete_documents_of_kb(self, kb_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s", (kb_id,))


_registry: Optional[KBRegistry] = None


def get_registry() -> KBRegistry:
    """进程级单例工厂：与向量库后端共用 VECTOR_BACKEND 开关。"""
    global _registry
    if _registry is None:
        if settings.vector_backend == "pgvector":
            _registry = PGKBRegistry(settings.postgres_dsn)
        else:
            import pathlib

            path = pathlib.Path(settings.chroma_dir).parent / "registry.db"
            path.parent.mkdir(parents=True, exist_ok=True)
            _registry = SQLiteKBRegistry(str(path))
    return _registry
