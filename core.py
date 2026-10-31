"""Shared logic: turn images into vectors, build the index, find the closest design.

No web-framework code lives here, so index.py, app.py and evaluate.py can all use it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/dinov2-small")
INPUT_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Every catalog design is indexed as several square views, so a customer's
# close-up of a repeating pattern still has something to match. Each number is
# the side of the square as a fraction of the image's shorter side.
CATALOG_SCALES = (1.0, 0.6, 0.35)
# A customer's photo is embedded as its centre square plus a tighter centre crop.
QUERY_SCALES = (1.0, 0.6)
# Set QUERY_ROTATIONS=0,90,180,270 if photos often arrive sideways (4x slower search).
QUERY_ROTATIONS = tuple(int(r) for r in os.environ.get("QUERY_ROTATIONS", "0").split(","))
GRID_STRIDE = 0.9
MAX_POSITIONS = 6
INDEX_VERSION = 1


# --------------------------------------------------------------------------- images

def load_image(source, max_side: int | None = None) -> Image.Image:
    """Open a path or file object as an upright RGB image, optionally capped in size."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            img = background
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _positions(extent: int, side: int) -> list[int]:
    if side >= extent:
        return [0]
    n = min(MAX_POSITIONS, max(1, round((extent - side) / (side * GRID_STRIDE))) + 1)
    spots = {int(round(p)) for p in np.linspace(0, extent - side, n)}
    spots.add((extent - side) // 2)  # always include the centre
    return sorted(spots)


def catalog_views(img: Image.Image) -> list[Image.Image]:
    """The full design plus overlapping crops at a few zoom levels."""
    w, h = img.size
    short = min(w, h)
    views = []
    for scale in CATALOG_SCALES:
        side = min(short, max(8, int(round(scale * short))))
        for y in _positions(h, side):
            for x in _positions(w, side):
                views.append(img.crop((x, y, x + side, y + side)))
    return views


def query_views(img: Image.Image, rotations=None) -> list[Image.Image]:
    """Centre crops of a customer's photo (optionally at several rotations)."""
    views = []
    for angle in (QUERY_ROTATIONS if rotations is None else rotations):
        im = img if angle % 360 == 0 else img.rotate(angle, expand=True)
        w, h = im.size
        short = min(w, h)
        for scale in QUERY_SCALES:
            side = max(8, int(round(scale * short)))
            x, y = (w - side) // 2, (h - side) // 2
            views.append(im.crop((x, y, x + side, y + side)))
    return views


# ------------------------------------------------------------------------ embedder

class Embedder:
    """DINOv2 on CPU (GPU if present). Returns L2-normalised float32 vectors."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _prep(self, img: Image.Image) -> np.ndarray:
        resized = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        return (arr.transpose(2, 0, 1) - self._mean) / self._std

    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        normalize = torch.nn.functional.normalize
        out = []
        for i in range(0, len(images), batch_size):
            batch = np.stack([self._prep(im) for im in images[i:i + batch_size]])
            with torch.inference_mode():
                hidden = self.model(pixel_values=torch.from_numpy(batch).to(self.device)).last_hidden_state
                cls_token = normalize(hidden[:, 0], dim=1)
                patch_mean = normalize(hidden[:, 1:].mean(dim=1), dim=1)  # texture statistics
                feat = normalize(torch.cat([cls_token, patch_mean], dim=1), dim=1)
            out.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def load_embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------------------- index

def list_catalog(catalog_dir) -> dict[str, Path]:
    """Map design number (file name without extension) to its image file."""
    files: dict[str, Path] = {}
    for path in sorted(Path(catalog_dir).iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in files:
            raise ValueError(
                f"Two files share the design number {path.stem!r}: {files[path.stem].name} and {path.name}"
            )
        files[path.stem] = path
    return files


def _fingerprint(path: Path) -> str:
    st = path.stat()
    key = f"v{INDEX_VERSION}|{MODEL_NAME}|{INPUT_SIZE}|{CATALOG_SCALES}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode()).hexdigest()


def _save_thumb(img: Image.Image, path: Path, size: int = 480) -> None:
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    thumb.save(path, "JPEG", quality=82)


def build_index(catalog_dir, data_dir, embedder=None, log=print) -> dict:
    """Embed every catalog image (skipping unchanged ones) and write the index files.

    `embedder` may be None; the model is then loaded only if something needs embedding.
    """
    catalog_dir, data_dir = Path(catalog_dir), Path(data_dir)
    cache_dir, thumbs_dir = data_dir / "cache", data_dir / "thumbs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    files = list_catalog(catalog_dir)
    if not files:
        raise ValueError(f"No images found in {catalog_dir}/ (looking for {', '.join(sorted(IMAGE_EXTS))})")

    chunks, counts = [], []
    computed = reused = 0
    for n, (design, path) in enumerate(files.items(), start=1):
        fingerprint = _fingerprint(path)
        cache, thumb = cache_dir / f"{design}.npz", thumbs_dir / f"{design}.jpg"
        vectors = None
        if cache.exists() and thumb.exists():
            try:
                with np.load(cache) as z:
                    if str(z["fingerprint"]) == fingerprint:
                        vectors = z["vectors"]
            except Exception:
                vectors = None
        if vectors is None:
            if embedder is None:
                embedder = load_embedder()
            img = load_image(path, max_side=2400)
            vectors = embedder.embed(catalog_views(img))
            np.savez(cache, vectors=vectors, fingerprint=np.array(fingerprint))
            _save_thumb(img, thumb)
            computed += 1
            log(f"[{n}/{len(files)}] indexed {design} ({len(vectors)} views)")
        else:
            reused += 1
        chunks.append(vectors)
        counts.append(len(vectors))

    # Write the vectors first and meta.json last: the running app watches meta.json.
    vectors = np.concatenate(chunks).astype(np.float32)
    tmp_vectors = data_dir / "embeddings.tmp.npy"
    np.save(tmp_vectors, vectors)
    os.replace(tmp_vectors, data_dir / "embeddings.npy")

    meta = {
        "version": INDEX_VERSION,
        "model": MODEL_NAME,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "designs": list(files),
        "files": [p.name for p in files.values()],
        "views_per_design": counts,
        "computed": computed,
        "reused": reused,
    }
    tmp_meta = data_dir / "meta.tmp.json"
    tmp_meta.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_meta, data_dir / "meta.json")

    for stale in cache_dir.glob("*.npz"):  # designs removed from the catalog
        if stale.stem not in files:
            stale.unlink()
    for stale in thumbs_dir.glob("*.jpg"):
        if stale.stem not in files:
            stale.unlink()
    return meta


@dataclass
class Index:
    vectors: np.ndarray      # (views, dim), rows L2-normalised
    view_design: np.ndarray  # (views,) position of each view's design in `designs`
    designs: list[str]
    files: list[str]
    model: str

    @classmethod
    def load(cls, data_dir) -> "Index":
        data_dir = Path(data_dir)
        meta = json.loads((data_dir / "meta.json").read_text())
        vectors = np.load(data_dir / "embeddings.npy")
        counts = np.asarray(meta["views_per_design"])
        if len(vectors) != counts.sum():
            raise ValueError("embeddings.npy and meta.json do not match; re-run index.py")
        return cls(
            vectors=vectors,
            view_design=np.repeat(np.arange(len(counts)), counts),
            designs=meta["designs"],
            files=meta["files"],
            model=meta["model"],
        )

    def scores(self, query_vectors: np.ndarray) -> np.ndarray:
        """Best cosine similarity per design (max over the photo's views x the design's views)."""
        per_view = (query_vectors @ self.vectors.T).max(axis=0)
        best = np.full(len(self.designs), -1.0, dtype=np.float32)
        np.maximum.at(best, self.view_design, per_view)
        return best

    def top(self, query_vectors: np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
        best = self.scores(query_vectors)
        order = np.argsort(-best)[:k]
        return [(self.designs[i], self.files[i], float(best[i])) for i in order]
