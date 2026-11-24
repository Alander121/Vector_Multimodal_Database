from __future__ import annotations

import time
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from PIL import Image

from .db import close_pool, connection, init_db, open_pool
from .embeddings import EmbeddingModels
from .faiss_store import FaissStore
from .minio_store import FrameStore
from .search import SearchService


class TextQuery(BaseModel):
    text: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    open_pool()
    init_db()
    app.state.frames = FrameStore()
    app.state.frames.ensure_bucket()
    app.state.models = EmbeddingModels()
    app.state.faiss = FaissStore()
    app.state.search = SearchService(app.state.models, app.state.faiss)
    yield
    close_pool()


app = FastAPI(
    title="FPV Multimodal Search — PostgreSQL + FAISS",
    version="2.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    with connection() as conn:
        db_ok = conn.execute("SELECT 1").fetchone()[0] == 1
    return {
        "status": "ok" if db_ok else "degraded",
        "postgres": db_ok,
        "faiss": app.state.faiss.stats(),
    }


@app.get("/stats")
def stats():
    with connection() as conn:
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    return {
        "videos": int(videos),
        "frames": int(frames),
        "faiss": app.state.faiss.stats(),
    }


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), limit: int = 20):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    started = time.perf_counter()
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    results = app.state.search.image_search(image, limit=limit)
    return {
        "query_type": "image",
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "results": results,
    }


@app.post("/search/text")
def search_text(query: TextQuery):
    started = time.perf_counter()
    results = app.state.search.text_search(query.text, query.limit)
    return {
        "query_type": "text",
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "results": results,
    }


@app.get("/frames/{frame_id}")
def get_frame(frame_id: int):
    with connection() as conn:
        row = conn.execute(
            "SELECT frame_path FROM frames WHERE id=%s", (frame_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Frame not found")
    try:
        data = app.state.frames.get_bytes(row[0])
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Frame object not found") from exc
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drone Search — PostgreSQL + FAISS</title>
<style>
body{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}
textarea,input,button{font:inherit;padding:10px}.row{display:flex;gap:10px;align-items:center}
.results{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-top:20px}
.card{border:1px solid #ddd;border-radius:10px;overflow:hidden}.card img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.meta{padding:10px}.muted{color:#666;font-size:13px}.tabs{margin:15px 0}.hidden{display:none}
</style></head>
<body>
<h1>Drone Vector Search V2.2</h1>
<div class="tabs"><button onclick="mode('image')">Изображение</button><button onclick="mode('text')">Текст</button></div>
<div id="imageBox"><input id="file" type="file" accept="image/*"></div>
<div id="textBox" class="hidden"><textarea id="text" rows="3" style="width:100%" placeholder="например: fpv drone flying over a field"></textarea></div>
<div class="row" style="margin-top:10px"><label>Результатов: <input id="limit" type="number" value="20" min="1" max="200"></label><button onclick="search()">Искать</button><span id="status"></span></div>
<div id="results" class="results"></div>
<script>
let current='image';
function mode(m){current=m;document.getElementById('imageBox').classList.toggle('hidden',m!=='image');document.getElementById('textBox').classList.toggle('hidden',m!=='text');}
async function search(){
 const status=document.getElementById('status');status.textContent='поиск...';document.getElementById('results').innerHTML='';
 const limit=document.getElementById('limit').value;let r;
 try{
  if(current==='image'){
   const f=document.getElementById('file').files[0];if(!f){status.textContent='выберите файл';return;}
   const fd=new FormData();fd.append('file',f);r=await fetch('/search/image?limit='+limit,{method:'POST',body:fd});
  }else{
   r=await fetch('/search/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('text').value,limit:Number(limit)})});
  }
  const data=await r.json();
  if(!r.ok){status.textContent=data.detail||'ошибка';return;}
  status.textContent=(data.elapsed_sec??'?')+' s';
  document.getElementById('results').innerHTML=data.results.map(x=>`<div class="card"><img src="${x.frame_url}" loading="lazy"><div class="meta"><b>${x.video_name}</b><div class="muted">${x.timestamp_sec.toFixed(2)} s · score ${x.score.toFixed(5)}</div></div></div>`).join('');
 }catch(e){status.textContent='ошибка соединения';}
}
</script>
</body></html>
    """)
