from __future__ import annotations

from collections import defaultdict

from .config import settings
from .db import connection
from .embeddings import EmbeddingModels
from .faiss_store import FaissStore


class SearchService:
    def __init__(self, models: EmbeddingModels, faiss_store: FaissStore) -> None:
        self.models = models
        self.faiss = faiss_store

    def image_search(self, image, limit: int = 20) -> list[dict]:
        sig, dino = self.models.encode_image(image)
        return self._fused(sig, dino, limit)

    def text_search(self, text: str, limit: int = 20) -> list[dict]:
        sig = self.models.encode_text(text)
        return self._fused(sig, None, limit)

    def _fused(self, sig, dino, limit: int) -> list[dict]:
        fetch = max(settings.search_fetch, limit * 5)
        raw = self.faiss.search(sig, dino, fetch)
        scores = defaultdict(float)
        ranks: dict[int, dict[str, int]] = {}

        for name, (_, ids) in raw.items():
            weight = settings.siglip_weight if name == "siglip" else settings.dino_weight
            for rank, frame_id in enumerate(ids, start=1):
                frame_id = int(frame_id)
                if frame_id < 0:
                    continue
                ranks.setdefault(frame_id, {})[name] = rank
                scores[frame_id] += weight / (settings.rrf_k + rank)

        return self._rows_for_ids(scores, ranks, limit)

    @staticmethod
    def _best_per_video(results: list[dict], limit: int) -> list[dict]:
        best: dict[str, dict] = {}
        for item in results:
            key = item["video_id"]
            if key not in best or item["score"] > best[key]["score"]:
                best[key] = item
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:limit]

    def _rows_for_ids(
        self,
        scores: dict[int, float],
        ranks: dict[int, dict[str, int]],
        limit: int,
    ) -> list[dict]:
        if not scores:
            return []

        # Keep the SQL candidate set small. PostgreSQL is used only for
        # metadata lookup after FAISS has already reduced the search space.
        ordered_ids = [fid for fid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        candidate_ids = ordered_ids[: max(limit * 3, limit)]

        placeholders = ",".join(["%s"] * len(candidate_ids))
        with connection() as conn:
            rows = conn.execute(
                f"""
                SELECT f.id, f.video_id, v.video_name, v.source_path,
                       f.frame_number, f.timestamp_sec, f.frame_path
                FROM frames f
                JOIN videos v ON v.id=f.video_id
                WHERE f.id IN ({placeholders})
                """,
                candidate_ids,
            ).fetchall()

        by_id = {int(row[0]): row for row in rows}
        result = []
        for frame_id in candidate_ids:
            row = by_id.get(frame_id)
            if not row:
                continue
            rank_info = ranks.get(frame_id, {})
            result.append(
                {
                    "frame_id": frame_id,
                    "video_id": str(row[1]),
                    "video_name": row[2],
                    "source_path": row[3],
                    "frame_number": int(row[4]),
                    "timestamp_sec": float(row[5]),
                    "frame_path": row[6],
                    "frame_url": f"/frames/{frame_id}",
                    "score": float(scores[frame_id]),
                    "siglip_rank": rank_info.get("siglip"),
                    "dino_rank": rank_info.get("dino"),
                }
            )
        return self._best_per_video(result, limit)
