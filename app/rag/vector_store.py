"""向量库抽象层：两个后端，接口完全一致。

- PGvectorStore：PostgreSQL 16 + pgvector 扩展（docker compose up -d 一键起），生产级；
- ChromaStore：本地持久化，零依赖快速验证（v0.1-v0.2 所用）。

检索层只依赖接口，切换后端只改 .env 的 VECTOR_BACKEND——
这就是「面向接口编程」在 RAG 工程里的落法。

v0.6 多知识库语义：
- chunk 写入时打上 kb_id / doc_id 血缘（列 + metadata 双写）；
- search / get_all_documents / count 支持按 kb 过滤；
- delete_by_doc / delete_by_kb 支撑文档与知识库的删除闭环。
"""
import hashlib
from typing import List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pgvector.psycopg import register_vector
from pgvector.vector import Vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings
from app.core.embeddings import BGEEmbeddings


def _doc_id(doc: Document) -> str:
    """按内容哈希生成稳定 chunk id：同一内容重复入库自动覆盖，脚本可幂等执行。"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


class PGvectorStore:
    """PostgreSQL + pgvector 后端。

    - 表结构：chunks(id, kb_id, doc_id, content, metadata jsonb, embedding vector(512))
    - 相似度：余弦距离（<=> 算子），score = 1 - distance，与 Chroma 后端口径一致
    - 并发：psycopg_pool 连接池（FastAPI 线程池并发请求时连接不共享、线程安全）
    """

    def __init__(self) -> None:
        self._embeddings = BGEEmbeddings()
        self._pool = ConnectionPool(
            settings.postgres_dsn,
            min_size=1,
            max_size=8,
            kwargs={"autocommit": True},
            open=True,
        )
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({settings.embed_dim}) NOT NULL
                )
                """
            )
            # v0.6 增量迁移：老库补血缘列
            conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kb_id TEXT NOT NULL DEFAULT ''")
            conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_id TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")

    def add_documents(self, docs: List[Document], kb_id: str, doc_id: str) -> int:
        """写入一批 chunk，血缘（kb_id/doc_id）同时进列与 metadata。"""
        for d in docs:
            d.metadata.setdefault("kb_id", kb_id)
            d.metadata.setdefault("doc_id", doc_id)
        vectors = self._embeddings.embed_documents([d.page_content for d in docs])
        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for doc, vec in zip(docs, vectors):
                    cur.execute(
                        "INSERT INTO chunks (id, kb_id, doc_id, content, metadata, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s)"
                        " ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,"
                        " kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id,"
                        " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                        (_doc_id(doc), kb_id, doc_id, doc.page_content, Jsonb(doc.metadata), Vector(vec)),
                    )
        return len(docs)

    def clear(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("TRUNCATE TABLE chunks")

    def delete_by_doc(self, doc_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("DELETE FROM chunks WHERE doc_id = %s RETURNING id", (doc_id,)).fetchall()
        return len(row)

    def delete_by_kb(self, kb_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("DELETE FROM chunks WHERE kb_id = %s RETURNING id", (kb_id,)).fetchall()
        return len(row)

    def search(
        self, query: str, top_k: Optional[int] = None, kb_id: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        k = top_k or settings.retrieval_top_k
        vec = self._embeddings.embed_query(query)
        with self._pool.connection() as conn:
            register_vector(conn)
            if kb_id is None:
                rows = conn.execute(
                    "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score"
                    " FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                    (Vector(vec), Vector(vec), k),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score"
                    " FROM chunks WHERE kb_id = %s"
                    " ORDER BY embedding <=> %s::vector LIMIT %s",
                    (Vector(vec), kb_id, Vector(vec), k),
                ).fetchall()
        return [
            (Document(page_content=r[0], metadata=r[1]), float(r[2])) for r in rows
        ]

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Document]:
        with self._pool.connection() as conn:
            if kb_id is None:
                rows = conn.execute("SELECT content, metadata FROM chunks").fetchall()
            else:
                rows = conn.execute(
                    "SELECT content, metadata FROM chunks WHERE kb_id = %s", (kb_id,)
                ).fetchall()
        return [Document(page_content=r[0], metadata=r[1]) for r in rows]

    def count(self, kb_id: Optional[str] = None) -> int:
        with self._pool.connection() as conn:
            if kb_id is None:
                return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            return conn.execute(
                "SELECT count(*) FROM chunks WHERE kb_id = %s", (kb_id,)
            ).fetchone()[0]


class ChromaStore:
    """Chroma 本地后端（降级/开发用）。血缘存在 chunk metadata 里。"""

    def __init__(self) -> None:
        self._embeddings = BGEEmbeddings()
        self._store = self._new_store()

    def _new_store(self) -> Chroma:
        return Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: List[Document], kb_id: str, doc_id: str) -> int:
        for d in docs:
            d.metadata.setdefault("kb_id", kb_id)
            d.metadata.setdefault("doc_id", doc_id)
        ids = [_doc_id(d) for d in docs]
        self._store.add_documents(docs, ids=ids)
        return len(docs)

    def clear(self) -> None:
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
        self._store = self._new_store()

    def delete_by_doc(self, doc_id: str) -> int:
        before = self.count()
        self._store.delete(where={"doc_id": doc_id})
        return before - self.count()

    def delete_by_kb(self, kb_id: str) -> int:
        before = self.count()
        self._store.delete(where={"kb_id": kb_id})
        return before - self.count()

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Document]:
        where = {"kb_id": kb_id} if kb_id else None
        data = self._store.get(where=where, include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]

    def search(
        self, query: str, top_k: Optional[int] = None, kb_id: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        k = top_k or settings.retrieval_top_k
        where = {"kb_id": kb_id} if kb_id else None
        # langchain_chroma 的 where 是 __query_collection 保留参数，过滤条件走 filter=
        return self._store.similarity_search_with_relevance_scores(query, k=k, filter=where)

    def count(self, kb_id: Optional[str] = None) -> int:
        if kb_id is None:
            return self._store._collection.count()
        return len(self._store.get(where={"kb_id": kb_id}, include=[])["ids"])


def get_store():
    """向量库工厂：按 .env 的 VECTOR_BACKEND 返回对应实现。"""
    if settings.vector_backend == "pgvector":
        return PGvectorStore()
    return ChromaStore()
