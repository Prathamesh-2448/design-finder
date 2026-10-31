"""Measure accuracy and suggest a MIN_SCORE (the "no confident match" cutoff).

Simulated photos (default): customer-style photos are made from your own catalog images
(random crop, tilt, brightness, blur, JPEG compression) and searched for.

Real photos (better): put customer photos in eval_photos/ named
    <design number>__<anything>.jpg      e.g.  D1042__whatsapp-1.jpg
and they are evaluated too.

    python evaluate.py
    python evaluate.py --trials 5 --rotate90
"""
import argparse
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import core


def simulate_photo(img: Image.Image, rng: np.random.Generator, rotate90: bool = False) -> Image.Image:
    w, h = img.size
    scale, aspect = rng.uniform(0.35, 1.0), rng.uniform(0.8, 1.25)
    cw = max(16, min(w, int(w * scale * math.sqrt(aspect))))
    ch = max(16, min(h, int(h * scale / math.sqrt(aspect))))
    x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    out = img.crop((x, y, x + cw, y + ch))

    out = out.rotate(float(rng.uniform(-20, 20)), resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    margin = int(0.12 * min(out.size))  # trim the corners the rotation left empty
    out = out.crop((margin, margin, out.width - margin, out.height - margin))
    if rotate90:
        out = out.rotate(int(rng.choice([0, 90, 180, 270])), expand=True)

    longest = int(rng.integers(600, 1400))
    if max(out.size) > longest:
        out.thumbnail((longest, longest), Image.LANCZOS)
    out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.65, 1.35)))
    out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.8, 1.2)))
    out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.8, 1.2)))
    blur = float(rng.uniform(0, 1.5))
    if blur > 0.3:
        out = out.filter(ImageFilter.GaussianBlur(blur))

    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(rng.integers(45, 90)))
    buf.seek(0)
    return core.load_image(buf)


def run_cases(cases, index: core.Index, embedder) -> list[dict]:
    position = {design: i for i, design in enumerate(index.designs)}
    rows = []
    for design, img in cases:
        if design not in position:
            print(f"  skipped: {design!r} is not in the index")
            continue
        scores = index.scores(embedder.embed(core.query_views(img)))
        true = position[design]
        rows.append({
            "design": design,
            "rank": int((scores > scores[true]).sum()) + 1,
            "score": float(scores[true]),
            # Best score among the *other* designs = what an unknown design would score.
            "wrong_best": float(np.delete(scores, true).max()) if len(scores) > 1 else -1.0,
            "predicted": index.designs[int(scores.argmax())],
        })
    return rows


def report(label: str, rows: list[dict], suggest: bool) -> None:
    if not rows:
        print(f"\n{label}: no photos evaluated")
        return
    ranks = np.array([r["rank"] for r in rows])
    correct = np.array([r["score"] for r in rows])
    print(f"\n{label}: {len(rows)} photos")
    print(f"  right design ranked 1st:      {np.mean(ranks == 1):6.1%}")
    print(f"  right design in the top 5:    {np.mean(ranks <= 5):6.1%}")
    print(f"  score of the right design:    median {np.median(correct):.2f}, 5th percentile {np.percentile(correct, 5):.2f}")

    confusions = Counter((r["design"], r["predicted"]) for r in rows if r["rank"] > 1)
    if confusions:
        print("  most common mix-ups (true -> returned first):")
        for (true, predicted), n in confusions.most_common(8):
            print(f"    {true} -> {predicted}  x{n}")

    if suggest:
        wrong = np.array([r["wrong_best"] for r in rows])
        low, high = float(np.percentile(correct, 5)), float(np.percentile(wrong, 95))
        print(f"  best score when the design is NOT the right one: median {np.median(wrong):.2f}, 95th percentile {high:.2f}")
        if low > high:
            print(f"  -> clean separation. Suggested MIN_SCORE = {(low + high) / 2:.2f}")
        else:
            print(f"  -> the ranges overlap. MIN_SCORE = {low:.2f} accepts 95% of correct matches, "
                  f"but some unknown designs will still look confident.")
        print("  (simulated photos are easier than real ones: confirm with real photos in eval_photos/)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="catalog")
    parser.add_argument("--data", default="data")
    parser.add_argument("--photos", default="eval_photos", help="folder of real customer photos (optional)")
    parser.add_argument("--trials", type=int, default=3, help="simulated photos per design")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotate90", action="store_true", help="also turn simulated photos sideways / upside down")
    args = parser.parse_args()

    index = core.Index.load(args.data)
    embedder = core.load_embedder()
    rng = np.random.default_rng(args.seed)

    def simulated():
        for design, path in core.list_catalog(args.catalog).items():
            img = core.load_image(path, max_side=2400)
            for _ in range(args.trials):
                yield design, simulate_photo(img, rng, args.rotate90)

    print(f"Index: {len(index.designs)} designs, {len(index.vectors)} views, model {index.model}")
    report("Simulated customer photos", run_cases(simulated(), index, embedder), suggest=True)

    photos = Path(args.photos)
    if photos.is_dir():
        def real():
            for path in sorted(photos.iterdir()):
                if path.suffix.lower() in core.IMAGE_EXTS:
                    yield path.stem.split("__")[0], core.load_image(path, max_side=1600)
        report("Real customer photos", run_cases(real(), index, embedder), suggest=True)
    else:
        print(f"\n(No {photos}/ folder, so only simulated photos were tested.)")


if __name__ == "__main__":
    main()
