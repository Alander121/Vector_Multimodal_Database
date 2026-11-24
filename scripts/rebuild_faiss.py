from app.db import open_pool, close_pool, init_db
from app.faiss_store import FaissStore

open_pool()
try:
    init_db()
    # FaissStore itself loads the current index or rebuilds it when missing/mismatched.
    FaissStore()
finally:
    close_pool()
