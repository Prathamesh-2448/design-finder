"""Web app: upload a customer's photo, get the closest design numbers.

    uvicorn app:app --host 0.0.0.0 --port 8000

Environment (all optional):
    APP_USER / APP_PASSWORD   require HTTP Basic login (set both)
    MIN_SCORE                 below this best score the UI says "no confident match" (default 0.55)
    TOP_K                     how many candidates to return (default 5)
    CATALOG_DIR / DATA_DIR    default ./catalog and ./data
"""
import base64
import io
import os
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import core

CATALOG_DIR = Path(os.environ.get("CATALOG_DIR", "catalog"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TOP_K = int(os.environ.get("TOP_K", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.55"))
APP_USER = os.environ.get("APP_USER")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
MAX_UPLOAD = 15 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Design Finder", docs_url=None, redoc_url=None)

_lock = threading.Lock()       # one embedding at a time; keeps CPU use predictable
_embedder = None
_base = None            # catalog index as built by index.py
_index = None           # _base + confirmed customer photos; rebuilt when either changes
_index_stamp = None


def _meta_stamp():
    try:
        return (DATA_DIR / "meta.json").stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _current():
    """Return (embedder, index), loading the model once and reloading the index if index.py re-ran."""
    global _embedder, _base, _index, _index_stamp
    with _lock:
        stamp = _meta_stamp()
        if stamp is None:
            raise HTTPException(503, "The catalog has not been indexed yet. Run: python index.py")
        if _base is None or stamp != _index_stamp:
            try:
                _base = core.Index.load(DATA_DIR)
            except Exception as err:
                raise HTTPException(503, f"The index could not be loaded ({err}). Re-run: python index.py")
            _index_stamp = stamp
            _index = None
        if _embedder is None:
            _embedder = core.load_embedder()
        if _index is None:
            vectors, positions = core.load_feedback_vectors(DATA_DIR, _base.designs, _embedder)
            _index = _base.with_extra(vectors, positions)
        return _embedder, _index


# ----------------------------------------------------------------------- optional login

if APP_USER and APP_PASSWORD:
    _expected = base64.b64encode(f"{APP_USER}:{APP_PASSWORD}".encode()).decode()

    @app.middleware("http")
    async def basic_auth(request, call_next):
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "basic" and secrets.compare_digest(token.strip(), _expected):
            return await call_next(request)
        return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Design Finder"'})


# --------------------------------------------------------------------------- endpoints

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "indexed": _meta_stamp() is not None}


@app.post("/search")
def search(photo: UploadFile = File(...)):
    raw = photo.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is too large (limit 15 MB).")
    try:
        img = core.load_image(io.BytesIO(raw), max_side=1600)
    except Exception:
        raise HTTPException(400, "That file could not be read as an image.")

    embedder, index = _current()
    with _lock:
        vectors = embedder.embed(core.query_views(img))
    matches = [
        {
            "design_number": design,
            "score": round(score, 4),
            "thumb": f"/thumbs/{design}.jpg",
            "image": f"/catalog/{file}",
        }
        for design, file, score in index.top(vectors, TOP_K)
    ]
    return {"confident": bool(matches) and matches[0]["score"] >= MIN_SCORE, "matches": matches}


def _read_upload(photo: UploadFile) -> bytes:
    raw = photo.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is too large (limit 15 MB).")
    return raw


@app.post("/feedback")
def confirm_match(photo: UploadFile = File(...), design_number: str = Form(...)):
    """The manager confirmed which design the customer's photo really is. Remember it."""
    global _index
    raw = _read_upload(photo)
    embedder, index = _current()
    design = design_number.strip()
    if design not in index.designs:
        raise HTTPException(404, f"There is no design numbered {design!r} in the catalog.")
    try:
        with _lock:
            fid = core.save_feedback(DATA_DIR, design, raw, embedder)
            _index = None  # rebuilt (cheaply, from cached vectors) on the next search
    except Exception:
        raise HTTPException(400, "That file could not be read as an image.")
    return {"id": fid, "design_number": design, "saved_photos": core.feedback_count(DATA_DIR, design)}


@app.delete("/feedback/{design_number}/{fid}")
def undo_confirmation(design_number: str, fid: str):
    """Undo a mistaken confirmation."""
    global _index
    _, index = _current()
    if design_number not in index.designs or not core.valid_feedback_id(fid):
        raise HTTPException(404, "Nothing to undo.")
    with _lock:
        removed = core.delete_feedback(DATA_DIR, design_number, fid)
        _index = None
    return {"removed": removed}


CATALOG_DIR.mkdir(exist_ok=True)
(DATA_DIR / "thumbs").mkdir(parents=True, exist_ok=True)
app.mount("/thumbs", StaticFiles(directory=DATA_DIR / "thumbs"), name="thumbs")
app.mount("/catalog", StaticFiles(directory=CATALOG_DIR), name="catalog")
