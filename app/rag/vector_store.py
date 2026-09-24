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
from contextlib import contextmanager

from psycopg.types.json import Jsonb

from app.config import settings
from app.core.embeddings import BGEEmbeddings
from app.core.pg_pool import get_pool


def content_hash(text: str) -> str:
    """chunk 正文的内容哈希。

    这是「这段文字是什么」的身份，与它属于哪个文档 / 哪个知识库无关。
    语义缓存键（app/rag/generator.py）与评测集的 gold 匹配用的都是它。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def chunk_uid(doc_id: str, text: str) -> str:
    """存储层主键：内容哈希 + 所属文档。

    为什么必须带 doc_id（工单 03）：以前主键就是 content_hash(text)，而写入语句是
    ON CONFLICT (id) DO UPDATE SET kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id。
    于是同一段文本第二次入库时，那一行不是新增、而是**被改挂到新文档名下**。
    触发场景都很日常：
      - 同一份文件重复上传 → 新 doc_id 抢走旧 doc 的全部 chunk，删除任一文档
        都会物理带走另一个文档的内容，而 registry 里的 chunk_count 还写着旧值；
      - 同一份制度文档传进两个知识库 → kb_id 被最后一次写入覆盖，
        原知识库里的这段内容静默消失（search 是带 WHERE kb_id 过滤的）。
    主键收敛到「同一文档内去重」之后，跨文档 / 跨库的同内容 chunk 各自独立。
    （冒号分隔符在 Chroma 的 id 校验下同样可用，已实测。）
    """
    return f"{doc_id}:{content_hash(text)}"


def _doc_id(doc: Document) -> str:
    """[兼容别名] 按内容哈希取 chunk 身份。

    名字有历史包袱（doc_id 指的是「文档」，这里返回的是「chunk 内容哈希」），
    新代码请用 content_hash(text) 或 chunk_uid(doc_id, text)；保留这个别名是因为
    生成层的缓存键与评测脚本都以「内容身份」表达 gold。
    """
    return content_hash(doc.page_content)


class PGvectorStore:
    """PostgreSQL + pgvector 后端。

    - 表结构：chunks(id, kb_id, doc_id, content, metadata jsonb, embedding vector(512))
    - 相似度：余弦距离（<=> 算子），score = 1 - distance，与 Chroma 后端口径一致
    - 并发：与注册中心共用一个进程级连接池（app/core/pg_pool.py，工单 10）——
      既为连接数，也为 chunks 与 documents 能落进同一个事务（工单 18）
    """

    def __init__(self) -> None:
        self._embeddings = BGEEmbeddings()
        # 共享池：见 app/core/pg_pool.py（工单 10）
        self._dsn = settings.postgres_dsn
        self._pool = get_pool(self._dsn, max_size=8, application_name="rag-app")
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
            # 工单 03：内容哈希独立成列（id 已改为 doc_id:content_hash，不再能从 id 反推），
            # 语义缓存与评测按内容寻址时用这一列，不必再算一遍 md5
            conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)")
            self._ensure_ann_index(conn)

    def _ensure_ann_index(self, conn) -> None:
        """embedding 列上的近似最近邻索引（工单 09）。

        修复前这里只有 kb_id / doc_id 两个 btree，`ORDER BY embedding <=> $1` 因此
        是全表顺序扫描 + 逐行算距离 + 排序：几千 chunk 时 P95 看不出来，十万级
        这一条就是主要延迟来源，README 里的 P95 数字随之失效。

        一个必须写下来的前提：带 `WHERE kb_id = %s` 过滤时 HNSW 会在图上取够候选
        之前就被过滤条件剪断，所以**索引不是免费的加速器** —— 过滤选择性强时，
        不加大 ef_search 反而可能既慢又漏。实测数据见 data/bench/ann_report.json。
        """
        if settings.ann_index == "none":
            conn.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
            return
        if settings.ann_index == "ivfflat":
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks"
                f" USING ivfflat (embedding vector_cosine_ops) WITH (lists = {settings.ivfflat_lists})"
            )
            return
        if settings.ann_index != "hnsw":
            raise ValueError(f"未知 ANN_INDEX：{settings.ann_index}（可选 hnsw / ivfflat / none）")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks"
            " USING hnsw (embedding vector_cosine_ops)"
            f" WITH (m = {settings.hnsw_m}, ef_construction = {settings.hnsw_ef_construction})"
        )

    def embed_for(self, docs: List[Document]) -> List[List[float]]:
        """预先算好整批向量。

        存在的理由（工单 18）：事务写入必须**在拿到连接之前**完成推理。
        否则一条 PG 连接会被批量 ONNX 计算按几十秒，池子里 8 条连接能同时被
        几个上传占光，在线检索直接饿死。
        """
        return self._embeddings.embed_documents([d.page_content for d in docs])

    def add_documents_on(self, conn, docs: List[Document], kb_id: str, doc_id: str,
                         vectors: List[List[float]]) -> int:
        """在**给定连接**上写入 chunk（不自己取连接），供外层事务包裹。"""
        register_vector(conn)
        with conn.cursor() as cur:
            for doc, vec in zip(docs, vectors):
                cur.execute(
                    "INSERT INTO chunks (id, kb_id, doc_id, content_hash, content, metadata, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    " ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,"
                    " kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id,"
                    " content_hash = EXCLUDED.content_hash,"
                    " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                    (
                        chunk_uid(doc_id, doc.page_content),
                        kb_id,
                        doc_id,
                        content_hash(doc.page_content),
                        doc.page_content,
                        Jsonb(doc.metadata),
                        Vector(vec),
                    ),
                )
        return len(docs)

    @contextmanager
    def transaction(self):
        """借一条连接开一个真事务，退出时**必定归还**给池。

        三个易错点都在这里处理掉：
        - getconn 之后必须 putconn，否则几个上传就能把 8 条连接永久借光；
        - 池里的连接是 autocommit 的，事务期间要关掉、归还前必须恢复，
          否则下一个借到这条连接的人会莫名其妙跑在一个未提交的事务里；
        - `with conn:` 只负责 commit/rollback，不负责还连接。
        """
        conn = self._pool.getconn()
        try:
            conn.autocommit = False
            with conn:  # 正常结束提交，异常回滚
                yield conn
        finally:
            conn.autocommit = True
            self._pool.putconn(conn)

    def add_documents(self, docs: List[Document], kb_id: str, doc_id: str) -> int:
        """写入一批 chunk，血缘（kb_id/doc_id）同时进列与 metadata。

        ON CONFLICT 的语义在工单 03 之后收敛为「同一文档内去重」：主键含 doc_id，
        所以重复内容不会再改写别的文档 / 别的知识库的归属行。
        脚本幂等性由 content_hash 保证（同一文档重灌 → 同一批主键 → 覆盖而非翻倍）。
        """
        for d in docs:
            d.metadata.setdefault("kb_id", kb_id)
            d.metadata.setdefault("doc_id", doc_id)
        vectors = self._embeddings.embed_documents([d.page_content for d in docs])
        with self._pool.connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for doc, vec in zip(docs, vectors):
                    digest = content_hash(doc.page_content)
                    cur.execute(
                        "INSERT INTO chunks (id, kb_id, doc_id, content_hash, content, metadata, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s)"
                        " ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,"
                        " kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id,"
                        " content_hash = EXCLUDED.content_hash,"
                        " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                        (
                            chunk_uid(doc_id, doc.page_content),
                            kb_id,
                            doc_id,
                            digest,
                            doc.page_content,
                            Jsonb(doc.metadata),
                            Vector(vec),
                        ),
                    )
        return len(docs)

    def clear(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("TRUNCATE TABLE chunks")

    def delete_by_doc(self, doc_id: str, kb_id: Optional[str] = None) -> int:
        """删除一个文档的全部 chunk；带 kb_id 时同时校验归属（防御性，见工单 03）。"""
        sql = "DELETE FROM chunks WHERE doc_id = %s"
        params: list = [doc_id]
        if kb_id is not None:
            sql += " AND kb_id = %s"
            params.append(kb_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount  # 以前是 DELETE ... RETURNING id + fetchall 只为数行数

    def delete_by_kb(self, kb_id: str) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE kb_id = %s", (kb_id,))
                return cur.rowcount

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

    def counts_by_kb(self) -> dict:
        """一次 GROUP BY 取回 {kb_id: chunk 数}（工单 17，替代每库一次的 count）。"""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT kb_id, count(*) FROM chunks GROUP BY kb_id"
            ).fetchall()
        return {r[0]: r[1] for r in rows}


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

    def embed_for(self, docs: List[Document]) -> List[List[float]]:
        return self._embeddings.embed_documents([d.page_content for d in docs])

    def transaction(self):
        """本地模式没有跨引擎事务，只有双 PG 后端有（见 PGvectorStore.transaction）。

        刻意抛错而不是"静默给一个假事务"：Chroma 是独立引擎，把它的写入包进
        PostgreSQL 的事务里只会让人以为得到了原子性。调用方用
        registry.shares_pool_with(store) 判断后根本不会走到这里。
        """
        raise NotImplementedError("ChromaStore 不支持跨表事务，请使用双 PG 后端")

    def add_documents(self, docs: List[Document], kb_id: str, doc_id: str) -> int:
        for d in docs:
            d.metadata.setdefault("kb_id", kb_id)
            d.metadata.setdefault("doc_id", doc_id)
        # 主键口径与 PG 后端完全一致（doc_id:content_hash），否则两后端的去重语义会分叉
        ids = [chunk_uid(doc_id, d.page_content) for d in docs]
        self._store.add_documents(docs, ids=ids)
        return len(docs)

    def clear(self) -> None:
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
        self._store = self._new_store()

    def _delete_returning_count(self, where: dict) -> int:
        """先按过滤条件取 ids 再删，返回真实删除数。

        旧写法是 count() 前后相减——两次全集合计数换一个小整数，Chroma 的 count()
        并不便宜；顺带修掉一个并发漏洞：相减法会把别人在此期间删掉的行算到自己头上。
        """
        ids = self._store.get(where=where, include=[])["ids"]
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    def delete_by_doc(self, doc_id: str, kb_id: Optional[str] = None) -> int:
        where: dict = {"doc_id": doc_id}
        if kb_id is not None:
            where = {"$and": [{"doc_id": doc_id}, {"kb_id": kb_id}]}
        return self._delete_returning_count(where)

    def delete_by_kb(self, kb_id: str) -> int:
        return self._delete_returning_count({"kb_id": kb_id})

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

    def counts_by_kb(self) -> dict:
        """Chroma 没有 GROUP BY，退化成「一次取全部 id+metadata，本地分组」。

        仍然优于旧写法的 N+1 次全集合扫描：这里只有一趟 get。include=[] 表示
        不要 documents/embeddings，只取元数据，是这个接口上最便宜的形式。
        """
        data = self._store.get(include=[], where=None)
        out: dict = {}
        for meta in data["metadatas"] or []:
            kb = (meta or {}).get("kb_id", "")
            out[kb] = out.get(kb, 0) + 1
        return out


def get_store():
    """向量库工厂：按 .env 的 VECTOR_BACKEND 返回对应实现。"""
    if settings.vector_backend == "pgvector":
        return PGvectorStore()
    return ChromaStore()
