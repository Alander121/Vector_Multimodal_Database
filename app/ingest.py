from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
from PIL import Image

from .config import settings
from .db import close_pool, connection, init_db, open_pool
from .embeddings import EmbeddingModels
from .faiss_store import FaissStore
from .minio_store import FrameStore

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def video_uuid(path: Path) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, path.resolve().as_uri())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_video_paths(root: Path, *, sorted_output: bool = False) -> Iterator[Path]:
    """Yield video paths incrementally.

    Default mode is streaming: paths are never accumulated in memory and are
    not globally sorted. Use --sorted only when deterministic traversal order
    is required for debugging/reproducibility.
    """
    root = root.resolve()
    if sorted_output:
        paths = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        for path in sorted(paths):
            yield path
        return

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def inspect_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (count / fps) if fps > 0 else 0.0
    cap.release()
    return fps, count, width, height, duration


def ingest_video(
    path: Path,
    models: EmbeddingModels,
    frames_store: FrameStore,
    faiss_store: FaissStore,
    reset: bool = False,
) -> bool:
    path = path.resolve()
    fps, count, width, height, duration = inspect_video(path)
    vid = video_uuid(path)
    print(
        f"[video] {path.name}: {fps:.2f} FPS, {count} frames, "
        f"{width}x{height}, {duration:.2f}s"
    )

    with connection() as conn:
        existing = conn.execute(
            "SELECT id FROM videos WHERE source_path=%s", (str(path),)
        ).fetchone()
        if existing and not reset:
            print("[skip] already ingested")
            return False
        if existing and reset:
            conn.execute("DELETE FROM videos WHERE id=%s", (existing[0],))
            conn.commit()

        conn.execute(
            """
            INSERT INTO videos(
                id, source_path, video_name, fps, frame_count,
                width, height, duration_sec
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (str(vid), str(path), path.name, fps, count, width, height, duration),
        )
        conn.commit()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    sample_fps = settings.frame_sample_fps
    step = max(1, round(fps / sample_fps)) if fps > 0 and sample_fps > 0 else 1
    batch_images: list[Image.Image] = []
    batch_jpegs: list[bytes] = []
    batch_meta: list[tuple[int, float, str, str]] = []
    processed = 0
    frame_no = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % step != 0:
                frame_no += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            ok_jpg, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not ok_jpg:
                frame_no += 1
                continue

            jpg = encoded.tobytes()
            ts = (frame_no / fps) if fps > 0 else 0.0
            object_name = f"{vid}/{frame_no:08d}.jpg"
            frame_sha = sha256_bytes(jpg)

            batch_images.append(image)
            batch_jpegs.append(jpg)
            batch_meta.append((frame_no, ts, object_name, frame_sha))

            if len(batch_images) >= settings.embed_batch_size:
                batch_size = len(batch_images)
                _flush_batch(
                    vid,
                    batch_images,
                    batch_jpegs,
                    batch_meta,
                    models,
                    frames_store,
                    faiss_store,
                )
                processed += batch_size
                print(f"[ingest] {processed} sampled frames")
                batch_images, batch_jpegs, batch_meta = [], [], []
            frame_no += 1

        if batch_images:
            batch_size = len(batch_images)
            _flush_batch(
                vid,
                batch_images,
                batch_jpegs,
                batch_meta,
                models,
                frames_store,
                faiss_store,
            )
            processed += batch_size
    finally:
        cap.release()
    print(f"[done] {path.name}: {processed} frames indexed")
    return True


def _flush_batch(
    vid,
    images: list[Image.Image],
    jpegs: list[bytes],
    meta: list[tuple[int, float, str, str]],
    models: EmbeddingModels,
    frames_store: FrameStore,
    faiss_store: FaissStore,
) -> None:
    sig, dino = models.encode_images(images)
    rows = []
    for i, (frame_no, ts, object_name, frame_sha) in enumerate(meta):
        frames_store.put_jpeg(object_name, jpegs[i])
        rows.append(
            (
                str(vid),
                frame_no,
                ts,
                object_name,
                frame_sha,
                sig[i].tobytes(),
                dino[i].tobytes(),
            )
        )

    with connection() as conn:
        inserted_ids = []
        for row in rows:
            cur = conn.execute(
                """
                INSERT INTO frames(
                    video_id, frame_number, timestamp_sec, frame_path,
                    frame_sha256, siglip_vector, dino_vector
                ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                row,
            )
            inserted_ids.append(cur.fetchone()[0])
        conn.commit()

    faiss_store.add(
        np.asarray(inserted_ids, dtype=np.int64),
        sig,
        dino,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest videos into PostgreSQL + MinIO + FAISS"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="single video file")
    group.add_argument("--dir", help="root directory; video paths are streamed by default")
    parser.add_argument("--reset", action="store_true", help="re-ingest already indexed videos")
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="sort all discovered video paths before ingest; uses extra RAM and is disabled by default",
    )
    args = parser.parse_args()

    open_pool()
    try:
        init_db()
        models = EmbeddingModels()
        store = FrameStore()
        store.ensure_bucket()
        faiss_store = FaissStore()

        if args.video:
            paths: Iterable[Path] = (Path(args.video),)
        else:
            print(
                "[scan] streaming video discovery enabled"
                + ("; sorting requested" if args.sorted else "")
            )
            paths = iter_video_paths(Path(args.dir), sorted_output=args.sorted)

        discovered = 0
        ingested = 0
        skipped = 0
        for path in paths:
            discovered += 1
            if ingest_video(path, models, store, faiss_store, reset=args.reset):
                ingested += 1
            else:
                skipped += 1

        print(f"[summary] discovered: {discovered:,}; ingested: {ingested:,}; skipped: {skipped:,}")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
