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
