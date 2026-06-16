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

    # ---- 向量库 ----
    # 后端切换：pgvector（生产，docker compose） / chroma（零依赖本地）
    vector_backend: str = "pgvector"
    postgres_dsn: str = "postgresql://ragkb:ragkb123@127.0.0.1:5432/rag_kb"
    chroma_dir: str = str(ROOT / "data" / "chroma")
    collection_name: str = "enterprise_kb"

    # ---- 缓存 ----
    redis_url: str = "redis://127.0.0.1:6379/0"

    # ---- 检索 ----
    retrieval_top_k: int = 5
    # 余弦相关度阈值：低于该值视为「资料中无依据」，触发无答案拒答
    score_threshold: float = 0.35

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()
