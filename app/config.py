import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://video:video@localhost:5432/videodb")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "video-frames")
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    device: str = os.getenv("DEVICE", "auto")
    siglip_model: str = os.getenv("SIGLIP_MODEL", "google/siglip2-base-patch16-224")
    dino_model: str = os.getenv("DINO_MODEL", "facebook/dinov2-base")
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 4)
    frame_sample_fps: float = _float("FRAME_SAMPLE_FPS", 0.5)
    faiss_dir: str = os.getenv("FAISS_DIR", "./faiss_index")
    faiss_index_type: str = os.getenv("FAISS_INDEX_TYPE", "hnsw").lower()
    faiss_hnsw_m: int = _int("FAISS_HNSW_M", 32)
    faiss_hnsw_ef_construction: int = _int("FAISS_HNSW_EF_CONSTRUCTION", 200)
    faiss_hnsw_ef_search: int = _int("FAISS_HNSW_EF_SEARCH", 64)
    search_fetch: int = _int("SEARCH_FETCH", 100)
    rrf_k: int = _int("RRF_K", 60)
    siglip_weight: float = _float("SIGLIP_WEIGHT", 0.60)
    dino_weight: float = _float("DINO_WEIGHT", 0.40)
    torch_num_threads: int = _int("TORCH_NUM_THREADS", 4)


settings = Settings()
