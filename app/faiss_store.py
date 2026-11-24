from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np

from .config import settings
from .db import connection

DIM = 768


class FaissStore:
    def __init__(self) -> None:
        self.dir = Path(settings.faiss_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.siglip_path = self.dir / "siglip.index"
        self.dino_path = self.dir / "dino.index"
        self.lock = threading.RLock()
        self.siglip = None
        self.dino = None
        self._load_or_rebuild()

    def _new_index(self):
        if settings.faiss_index_type == "flat":
            base = faiss.IndexFlatIP(DIM)
            return faiss.IndexIDMap2(base)
        if settings.faiss_index_type != "hnsw":
            raise ValueError("FAISS_INDEX_TYPE must be 'hnsw' or 'flat'")
        base = faiss.IndexHNSWFlat(DIM, settings.faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = settings.faiss_hnsw_ef_construction
        base.hnsw.efSearch = settings.faiss_hnsw_ef_search
        return faiss.IndexIDMap2(base)

    def _configure_search(self, index) -> None:
        if isinstance(index, faiss.IndexIDMap2):
            base = index.index
            if hasattr(base, "hnsw"):
                base.hnsw.efSearch = settings.faiss_hnsw_ef_search

    def _load_or_rebuild(self) -> None:
        with self.lock:
            if self.siglip_path.exists() and self.dino_path.exists():
                try:
                    self.siglip = faiss.read_index(str(self.siglip_path))
                    self.dino = faiss.read_index(str(self.dino_path))
                    self._configure_search(self.siglip)
                    self._configure_search(self.dino)
                    db_count = self.count_vectors()
                    if self.siglip.ntotal == db_count and self.dino.ntotal == db_count:
                        print(f"[faiss] loaded {db_count:,} vectors")
                        return
                    print("[faiss] index/database count mismatch; rebuilding")
                except Exception as exc:
                    print(f"[faiss] failed to load indexes: {exc}; rebuilding")
            self.rebuild()

    def count_vectors(self) -> int:
        with connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM frames WHERE siglip_vector IS NOT NULL AND dino_vector IS NOT NULL").fetchone()
            return int(row[0])

    def rebuild(self) -> None:
        sig = self._new_index()
        dino = self._new_index()
        batch = 50000
        last_id = 0
        total = 0
        while True:
            with connection() as conn:
                rows = conn.execute(
                    "SELECT id, siglip_vector, dino_vector FROM frames WHERE id > %s ORDER BY id LIMIT %s",
                    (last_id, batch),
                ).fetchall()
            if not rows:
                break
            ids = np.asarray([r[0] for r in rows], dtype=np.int64)
            sig_vecs = np.vstack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in rows])
            dino_vecs = np.vstack([np.frombuffer(bytes(r[2]), dtype=np.float32) for r in rows])
            sig.add_with_ids(sig_vecs, ids)
            dino.add_with_ids(dino_vecs, ids)
            total += len(rows)
            last_id = int(rows[-1][0])
        with self.lock:
            self.siglip, self.dino = sig, dino
            faiss.write_index(self.siglip, str(self.siglip_path))
            faiss.write_index(self.dino, str(self.dino_path))
        print(f"[faiss] rebuilt {total:,} vectors")

    def add(self, ids: np.ndarray, sig: np.ndarray, dino: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        sig = np.asarray(sig, dtype=np.float32)
        dino = np.asarray(dino, dtype=np.float32)
        with self.lock:
            self.siglip.add_with_ids(sig, ids)
            self.dino.add_with_ids(dino, ids)
            faiss.write_index(self.siglip, str(self.siglip_path))
            faiss.write_index(self.dino, str(self.dino_path))

    def search(self, sig_query: np.ndarray | None, dino_query: np.ndarray | None, k: int):
        with self.lock:
            results = {}
            if sig_query is not None:
                q = np.asarray(sig_query, dtype=np.float32).reshape(1, -1)
                scores, ids = self.siglip.search(q, k)
                results["siglip"] = (scores[0], ids[0])
            if dino_query is not None:
                q = np.asarray(dino_query, dtype=np.float32).reshape(1, -1)
                scores, ids = self.dino.search(q, k)
                results["dino"] = (scores[0], ids[0])
            return results

    def stats(self) -> dict:
        with self.lock:
            return {
                "siglip_ntotal": int(self.siglip.ntotal),
                "dino_ntotal": int(self.dino.ntotal),
                "index_type": settings.faiss_index_type,
                "dimension": DIM,
            }
