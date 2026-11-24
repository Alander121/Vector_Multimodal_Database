from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from psycopg_pool import ConnectionPool

from .config import settings

pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1,
    max_size=8,
    open=False,
)


def open_pool() -> None:
    pool.open()


def close_pool() -> None:
    pool.close()


@contextmanager
def connection() -> Iterator:
    with pool.connection() as conn:
        yield conn


def init_db() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS videos (
        id UUID PRIMARY KEY,
        source_path TEXT NOT NULL UNIQUE,
        video_name TEXT NOT NULL,
        fps DOUBLE PRECISION NOT NULL,
        frame_count BIGINT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        duration_sec DOUBLE PRECISION NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS frames (
        id BIGSERIAL PRIMARY KEY,
        video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
        frame_number INTEGER NOT NULL,
        timestamp_sec DOUBLE PRECISION NOT NULL,
        frame_path TEXT NOT NULL,
        frame_sha256 TEXT,
        siglip_vector BYTEA NOT NULL,
        dino_vector BYTEA NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(video_id, frame_number)
    );

    CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames(video_id);
    CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(video_id, timestamp_sec);
    CREATE INDEX IF NOT EXISTS idx_frames_sha256 ON frames(frame_sha256);
    """
    with connection() as conn:
        conn.execute(ddl)
        conn.commit()
