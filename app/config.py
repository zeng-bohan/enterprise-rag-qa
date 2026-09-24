"""全局配置：从 .env 读取，pydantic-settings 校验。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM（DeepSeek，OpenAI 兼容协议）----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ---- Embedding（本地 BGE，fastembed / ONNX Runtime）----
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    embed_dim: int = 512
    # 单条 ONNX 推理线程数。压测实测（i5-14400F，6P+4E）：
    # 默认（全核）单条推理吃满 CPU，并发请求零加速；显式限线程 + 专用有界线程池
    # （app/core/executor.py）后并发可叠加。4 是单请求延迟与吞吐的平衡点
    # （实测表见 docs/DESIGN.md 第 12 节）。
    onnx_threads: int = 4

    # ---- 向量库 ----
    # 后端切换：pgvector（生产，docker compose） / chroma（零依赖本地）
    vector_backend: str = "pgvector"
    postgres_dsn: str = "postgresql://ragkb:ragkb123@127.0.0.1:5432/rag_kb"
    chroma_dir: str = str(ROOT / "data" / "chroma")
    collection_name: str = "enterprise_kb"
    # pgvector 的 ANN 索引类型。
    #
    # 默认 none（精确顺序扫描）—— 这是**测出来的结论，不是怕麻烦**。
    # 50000 行 × 512 维、8 个知识库（data/bench/ann_report.json，可复跑）：
    #
    #   模式                      ef_search   p50 延迟    recall@10
    #   顺序扫描（全局精确）          —         107.4ms      1.000
    #   顺序扫描 + kb_id 过滤         —          17.1ms      1.000   ← 生产实际走的那条
    #   HNSW（无过滤）               100         3.5ms      0.177
    #   HNSW（无过滤）              1000        16.7ms      0.764
    #   HNSW + kb_id 过滤           100        14.8ms      1.000
    #
    # 三点结论：
    # 1. 本服务的检索**永远带 WHERE kb_id**，kb_id 上的 btree 先把候选缩到 1/8，
    #    再排序取 top-k —— 17ms 且完全精确。加 HNSW 在这个量级上一点便宜没占到。
    # 2. HNSW 在 pgvector 默认 ef_search=100 下把召回打到 0.177。对一个以引用溯源为
    #    卖点的系统，"快 30 倍但静默丢掉八成正确 chunk"不是优化，是故障。
    # 3. ef_search 的合法上限是 1000，也就是说**没法靠无限加大 ef 换回召回**。
    #
    # 什么时候该开：单个知识库内 chunk 数远超 ~10 万、且能接受先复测 recall 时。
    # 开关后请务必复跑 scripts/bench_ann.py 并同步更新上面的表。
    ann_index: str = "none"
    # HNSW 建索引参数（m=16 / ef_construction=64 是 pgvector 推荐的通用起点）
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    # ivfflat 的倒排列表数（只在 ANN_INDEX=ivfflat 时用到）；经验值 ~sqrt(row_count)
    ivfflat_lists: int = 100
    # 查询期候选队列长度。带 WHERE kb_id 过滤时 HNSW 是「先取图上的近邻、再按条件筛」，
    # 选择性越高、被筛掉的越多，召回掉得越狠 —— 调大 ef_search 是主要补偿手段。
    # 默认 100（pgvector 默认值）；实测影响见 data/bench/ann_report.json。
    hnsw_ef_search: int = 100

    # ---- 缓存 ----
    redis_url: str = "redis://127.0.0.1:6379/0"

    # ---- 检索 ----
    retrieval_top_k: int = 5
    # 余弦相关度阈值：低于该值视为「资料中无依据」，触发无答案拒答
    score_threshold: float = 0.35

    # ---- 多知识库与访问控制（v0.6）----
    # 脚本入库与单库问答的默认知识库名（不存在时服务启动自动创建）
    default_kb: str = "default"
    # 合法 X-API-Key（逗号分隔）；留空 = 关闭鉴权（本地 / 评测场景）
    api_keys: str = ""

    # ---- 边界防护（工单 20）----
    # 单次上传的体积上限（MB）。0 = 不限制，仅供本地调试；生产必须设。
    max_upload_mb: int = 20
    # 令牌桶速率（每秒补充多少请求），0 = 关闭限流。键 = X-API-Key，无鉴权时按客户端 IP。
    rate_limit_rps: float = 0.0
    rate_limit_burst: int = 20

    # ---- 流式（工单 19）----
    # 空闲多久发一次 SSE 注释帧保活。Nginx / 云负载均衡默认 60s 空闲断流，
    # 而 LLM 首 token 就可能等到十几秒 —— 不发心跳，长答案会在代理处被掐断。
    sse_heartbeat_seconds: float = 15.0

    # ---- 摄取模式（工单 12）----
    # sync：上传请求内同步索引（默认，保持既有部署与离线测试行为不变）
    # queue：入队交给 arq worker，响应变 202、chunk_count 稍后可见（破坏性，故不默认）
    ingest_mode: str = "sync"

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()
