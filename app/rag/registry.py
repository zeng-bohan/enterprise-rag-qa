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
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from app.config import settings


def new_doc_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KBError(RuntimeError):
    """知识库不存在 / 重名等业务错误，message 面向 API 直接返回。"""


class BackendError(RuntimeError):
    """存储后端故障（连不上 / 表缺失 / SQL 非法）。

    与 KBError 严格分开：KBError 是「业务上不该有结果」→ 404/409，
    BackendError 是「系统没能力回答」→ 503。两者混用会让运维对着
    「知识库不存在」排查一个连接池超时。
    """


# 领域异常翻译只认这一个唯一约束码，其余驱动异常一律归为后端故障
def as_domain_error(exc: Exception, duplicate_msg: str) -> RuntimeError:
    """把驱动层异常翻译成领域异常，绝不吞掉原始异常（保留 __cause__）。

    为什么单独有这个函数：这里原先写的是
        except Exception: raise KBError("知识库创建失败（重名？）")
    于是一个 SQL 语法错误（psycopg3 只认 %s，代码里写了 ?）也被翻译成「疑似重名」，
    把排障方向整个带偏。规则：只有 UniqueViolation 才是业务冲突，其他都是后端故障。

    返回异常对象而不直接 raise：调用方 `raise as_domain_error(...) from exc`，
    异常链由调用点保留（return 语句里不能写 from）。
    """
    first_line = (str(exc).strip().splitlines() or [""])[0]
    try:
        from psycopg import errors as pg_errors
    except ImportError:  # 本地模式未安装 psycopg，退化为通用后端故障
        return BackendError(f"存储后端故障（{type(exc).__name__}）：{first_line[:200]}")
    if isinstance(exc, pg_errors.UniqueViolation):
        return KBError(duplicate_msg)
    return BackendError(
        f"存储后端故障（{type(exc).__name__}）：{first_line[:200]}"
    )


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

    def update_document(
        self, kb_id: str, doc_id: str, status: str, chunk_count: Optional[int] = None, error: str = ""
    ) -> dict:
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
        now = _now()
        try:
            self._conn.execute(
                "INSERT INTO kbs (kb_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (kb_id, name, description, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise KBError(f"知识库名已存在: {name}") from exc
        return {"kb_id": kb_id, "name": name, "description": description, "created_at": now}

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
            "created_at": now,
            "updated_at": now,
        }

    def list_documents(self, kb_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
            " FROM documents WHERE kb_id = ? ORDER BY created_at",
            (kb_id,),
        ).fetchall()
        # kb_id 不在上面的 SELECT 里（它是过滤条件），要显式补回，
        # 否则与 PG 后端的返回形状不一致——契约测试 first-run 抓到的就是这条
        return [{**dict(r), "kb_id": kb_id} for r in rows]

    def get_document(self, kb_id: str, doc_id: str) -> dict:
        row = self._conn.execute(
            "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
            " FROM documents WHERE kb_id = ? AND doc_id = ?",
            (kb_id, doc_id),
        ).fetchone()
        if row is None:
            raise KBError(f"文档不存在: {doc_id}")
        return {**dict(row), "kb_id": kb_id}

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self.get_document(kb_id, doc_id)
        self._conn.execute("DELETE FROM documents WHERE kb_id = ? AND doc_id = ?", (kb_id, doc_id))
        self._conn.commit()

    def delete_documents_of_kb(self, kb_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE kb_id = ?", (kb_id,))
        self._conn.commit()

    def update_document(
        self, kb_id: str, doc_id: str, status: str, chunk_count: Optional[int] = None, error: str = ""
    ) -> dict:
        doc = self.get_document(kb_id, doc_id)
        count = doc["chunk_count"] if chunk_count is None else chunk_count
        self._conn.execute(
            "UPDATE documents SET status = ?, chunk_count = ?, error = ?, updated_at = ?"
            " WHERE kb_id = ? AND doc_id = ?",
            (status, count, error, _now(), kb_id, doc_id),
        )
        self._conn.commit()
        return self.get_document(kb_id, doc_id)


class PGKBRegistry(KBRegistry):
    """生产模式：PostgreSQL（与 chunks 同库，连接池复用 store 的配置）。"""

    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True},
            # timeout：PG 挂掉时快速失败（默认 30s 会让每个请求都卡在 checkout 上，
            # 表现为整站超时而不是一个明确的 503）
            timeout=10,
            # 复用的连接可能已被服务端回收（idle timeout / 重启），取回时先探一次
            check=ConnectionPool.check_connection,
            open=True,
        )
        self._init_schema()

    @contextmanager
    def _conn(self):
        """取连接并统一翻译驱动异常。

        读路径不关心「重名」这类业务语义，所以只需要「异常 → BackendError」这一条规则；
        写路径（create_kb / add_document）要区分 UniqueViolation，各自单独处理。
        """
        try:
            with self._pool.connection() as conn:
                yield conn
        except Exception as exc:
            raise as_domain_error(exc, str(exc)) from exc

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

    @staticmethod
    def _iso(value) -> str:
        """TIMESTAMPTZ(datetime) → ISO 字符串，与 SQLite 侧直接存文本的口径对齐。

        两个后端返回的 dict 形状必须完全一致（同一套键、同样的值类型），否则
        换后端就是换 API 契约。这一条以前没人管：PG 侧的读方法一律漏掉了
        created_at / updated_at，SQLite 侧带着 —— 见工单 05 的契约断言。
        """
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds")
        return str(value)

    def create_kb(self, name: str, description: str = "") -> dict:
        kb_id = uuid.uuid4().hex[:12]
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    # 占位符必须是 %s：psycopg3 不认 sqlite 风格的 ?，
                    # 混用会让语句原样发给 Postgres 当运算符解析并报语法错误
                    "INSERT INTO kbs (kb_id, name, description) VALUES (%s, %s, %s)"
                    " RETURNING kb_id, name, description, created_at",
                    (kb_id, name, description),
                ).fetchone()
        except Exception as exc:
            raise as_domain_error(exc, f"知识库名已存在: {name}") from exc
        return {
            "kb_id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": self._iso(row[3]),
        }

    def list_kbs(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT kb_id, name, description, created_at FROM kbs ORDER BY created_at"
            ).fetchall()
        return [
            {"kb_id": r[0], "name": r[1], "description": r[2], "created_at": self._iso(r[3])}
            for r in rows
        ]

    def get_kb(self, kb_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT kb_id, name, description, created_at FROM kbs WHERE kb_id = %s", (kb_id,)
            ).fetchone()
        if row is None:
            raise KBError(f"知识库不存在: {kb_id}")
        return {
            "kb_id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": self._iso(row[3]),
        }

    def kb_id_by_name(self, name: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT kb_id FROM kbs WHERE name = %s", (name,)).fetchone()
        return row[0] if row else None

    def delete_kb(self, kb_id: str) -> None:
        self.get_kb(kb_id)
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s", (kb_id,))
            conn.execute("DELETE FROM kbs WHERE kb_id = %s", (kb_id,))

    def add_document(
        self, kb_id: str, filename: str, chunk_count: int, status: str = "indexed", error: str = ""
    ) -> dict:
        self.get_kb(kb_id)
        doc_id = new_doc_id()
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "INSERT INTO documents (doc_id, kb_id, filename, status, chunk_count, error)"
                    " VALUES (%s, %s, %s, %s, %s, %s)"
                    " RETURNING doc_id, filename, status, chunk_count, error, created_at, updated_at",
                    (doc_id, kb_id, filename, status, chunk_count, error),
                ).fetchone()
        except Exception as exc:
            # 文档重名不是冲突（同名文档允许共存），这里只可能是真故障
            raise as_domain_error(exc, f"文档记录写入冲突: {filename}") from exc
        return {
            "doc_id": row[0],
            "kb_id": kb_id,
            "filename": row[1],
            "status": row[2],
            "chunk_count": row[3],
            "error": row[4],
            "created_at": self._iso(row[5]),
            "updated_at": self._iso(row[6]),
        }

    def list_documents(self, kb_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
                " FROM documents WHERE kb_id = %s ORDER BY created_at",
                (kb_id,),
            ).fetchall()
        return [self._document_row(kb_id, r) for r in rows]

    def get_document(self, kb_id: str, doc_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT doc_id, filename, status, chunk_count, error, created_at, updated_at"
                " FROM documents WHERE kb_id = %s AND doc_id = %s",
                (kb_id, doc_id),
            ).fetchone()
        if row is None:
            raise KBError(f"文档不存在: {doc_id}")
        return self._document_row(kb_id, row)

    @staticmethod
    def _document_row(kb_id: str, r) -> dict:
        return {
            "doc_id": r[0],
            "kb_id": kb_id,
            "filename": r[1],
            "status": r[2],
            "chunk_count": r[3],
            "error": r[4],
            "created_at": PGKBRegistry._iso(r[5]),
            "updated_at": PGKBRegistry._iso(r[6]),
        }

    def delete_document(self, kb_id: str, doc_id: str) -> None:
        self.get_document(kb_id, doc_id)
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s AND doc_id = %s", (kb_id, doc_id))

    def delete_documents_of_kb(self, kb_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE kb_id = %s", (kb_id,))

    def update_document(
        self, kb_id: str, doc_id: str, status: str, chunk_count: Optional[int] = None, error: str = ""
    ) -> dict:
        doc = self.get_document(kb_id, doc_id)
        count = doc["chunk_count"] if chunk_count is None else chunk_count
        with self._conn() as conn:
            conn.execute(
                "UPDATE documents SET status = %s, chunk_count = %s, error = %s, updated_at = now()"
                " WHERE kb_id = %s AND doc_id = %s",
                (status, count, error, kb_id, doc_id),
            )
        return self.get_document(kb_id, doc_id)


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
