# FPV Vector Search V2.2 — PostgreSQL + FAISS

V2.2 is the vector-only branch of the project. Semantic/full-text/hybrid search is intentionally excluded.

Pipeline:

`video -> sampled frames -> SigLIP2 + DINOv2 -> PostgreSQL metadata/vector backup + FAISS HNSW -> search`

For image search:

`image -> SigLIP2 + DINOv2 -> FAISS -> top frame IDs -> PostgreSQL metadata -> best frame per video`

For text search:

`text -> SigLIP2 text encoder -> FAISS SigLIP -> top frame IDs -> PostgreSQL metadata -> best frame per video`

## What changed from V2.1

- Removed the semantic layer from application code and fresh database schema.
- Removed `/search/semantic` and all hybrid endpoints.
- Removed semantic helper scripts/files.
- Video discovery is now streaming by default: paths are yielded one by one and are not accumulated in memory.
- Global path sorting is disabled by default.
- Optional `--sorted` is available only for reproducible/debug runs and intentionally collects all video paths before processing.
- JPEG bytes are encoded once during frame extraction and reused for MinIO upload; the previous second JPEG encoding was removed.
- API version is `2.2.0`.

## Architecture

- **PostgreSQL** — durable metadata and binary backup of embeddings.
- **MinIO** — JPEG frame storage.
- **FAISS** — in-memory vector search, persisted to `faiss_index/`.
- **SigLIP2** — image-to-image and text-to-image embedding space.
- **DINOv2** — additional visual embedding used for image retrieval fusion.

No PostgreSQL vector similarity is used. PostgreSQL is queried only after FAISS returns candidate frame IDs.

## Startup

```bash
docker compose build
docker compose up -d

docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

Swagger: http://localhost:8000/docs

Web UI: http://localhost:8000/

## Ingest a single video

```bash
docker compose exec api python -m app.ingest --video /app/data/videos/video.mp4
```

## Ingest a directory — default streaming mode

```bash
docker compose exec api python -m app.ingest --dir /app/data/videos
```

The default directory mode does **not** create a global list of all video paths and does **not** sort them. Files are discovered incrementally with `Path.rglob()` and processed one by one.

## Optional deterministic ordering

```bash
docker compose exec api python -m app.ingest --dir /app/data/videos --sorted
```

Use this only when deterministic traversal order is actually needed. It collects all matching paths in memory before ingestion.

## Re-index an existing video

```bash
docker compose exec api python -m app.ingest --video /app/data/videos/video.mp4 --reset
```

## Image search

```bash
curl -X POST "http://localhost:8000/search/image?limit=20" -F "file=@query.jpg"
```

## Text search

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"text":"FPV drone flying low over a road","limit":20}'
```

## Rebuild FAISS

```bash
docker compose exec api python -m scripts.rebuild_faiss
```

The rebuild reads the binary backup vectors from PostgreSQL in batches and recreates the two FAISS indexes.

After rebuilding, restart the API so the new indexes are loaded into the running process:

```bash
docker compose restart api
```

## Configuration

```env
DEVICE=auto
SIGLIP_MODEL=google/siglip2-base-patch16-224
DINO_MODEL=facebook/dinov2-base
EMBED_BATCH_SIZE=4
FRAME_SAMPLE_FPS=0.5
FAISS_INDEX_TYPE=hnsw
FAISS_HNSW_M=32
FAISS_HNSW_EF_CONSTRUCTION=200
FAISS_HNSW_EF_SEARCH=64
SEARCH_FETCH=100
RRF_K=60
SIGLIP_WEIGHT=0.60
DINO_WEIGHT=0.40
TORCH_NUM_THREADS=4
```

`FAISS_INDEX_TYPE=flat` is still available as a simple exact-search baseline. The default remains HNSW; this V2.2 intentionally does not introduce IVF/PQ/sharding or other tens-of-millions-scale optimizations.

## Data schema

### PostgreSQL: `videos`

Stores source video metadata:

- `id`
- `source_path`
- `video_name`
- `fps`
- `frame_count`
- `width`
- `height`
- `duration_sec`

### PostgreSQL: `frames`

One row per sampled frame:

- `id`
- `video_id`
- `frame_number`
- `timestamp_sec`
- `frame_path`
- `frame_sha256`
- `siglip_vector`
- `dino_vector`

The vector columns are binary float32 backups used to rebuild FAISS. They are not B-tree indexed.

### FAISS

- `siglip.index` — 768-D SigLIP2 embeddings.
- `dino.index` — 768-D DINOv2 embeddings.
- FAISS ID == PostgreSQL `frames.id`.

### MinIO

JPEG objects are stored under:

`<video_uuid>/<frame_number>.jpg`

## Existing V2.1 database

V2.2 no longer creates or uses semantic columns/tables. If an existing PostgreSQL volume was previously initialized by V2.1, legacy semantic columns/tables may remain physically present; V2.2 simply ignores them. This avoids a destructive automatic migration.

For a completely clean V2.2 database, start with a new PostgreSQL volume/database.
