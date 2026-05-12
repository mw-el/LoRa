#!/usr/bin/env python3
from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np


IN_DIR = Path("sources/correia/Revised")
OUT_DIR = Path("sources/correia/re-drawn")


def list_non_mirrored_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if p.stem.endswith("-Mirrored") or p.stem.endswith("_mirrored") or p.stem.lower().endswith("mirrored"):
            continue
        images.append(p)
    return sorted(images, key=lambda x: x.name.lower())


def paper_texture(h: int, w: int, seed: int = 1337) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Warm paper base (BGR).
    base = np.full((h, w, 3), (252, 251, 248), dtype=np.float32)
    noise = rng.normal(0.0, 4.0, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, min(h, w) * 0.02))
    fibers = rng.normal(0.0, 1.4, size=(h, w)).astype(np.float32)
    fibers = cv2.GaussianBlur(fibers, (0, 0), sigmaX=max(1.0, min(h, w) * 0.08))
    # Very subtle illumination variation so it reads as scanned paper, not flat digital white.
    illum = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    illum = cv2.GaussianBlur(illum, (0, 0), sigmaX=max(4.0, min(h, w) * 0.20))
    texture = base + (noise * 0.85 + fibers * 1.15 + illum * 0.35)[:, :, None]
    return np.clip(texture, 235, 255).astype(np.uint8)


def build_paper_alpha(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2)).astype(np.float32)

    # Paper candidate: (1) obvious paper and (2) slightly underexposed/gray paper.
    # Use a low-gradient heuristic to avoid eating into painted edges/washes.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    paper_strict = (s < 28.0) & (v > 185.0) & (dist < 42.0)
    paper_gray = (s < 40.0) & (v > 135.0) & (dist < 120.0) & (grad < 10.0)
    paper = paper_strict | paper_gray

    # Ensure the existing added border becomes paper (but don't wipe real paint).
    margin = int(math.ceil(0.10 * min(h, w)))
    yy, xx = np.indices((h, w), dtype=np.int32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy]).astype(np.float32)
    edge_paper = (edge_dist < float(margin)) & (s < 55.0) & (v > 120.0) & (dist < 160.0)

    paper = paper | edge_paper

    alpha = paper.astype(np.float32)
    # Soften only a little to avoid bleeding into painted edges.
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.6)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def apply_paper(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    alpha = build_paper_alpha(img_bgr)[:, :, None]
    paper = paper_texture(h, w, seed=1337).astype(np.float32)
    out = img_bgr.astype(np.float32) * (1.0 - alpha) + paper * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # User asked to delete existing results and replace.
    for p in OUT_DIR.iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)

    images = list_non_mirrored_images(IN_DIR)
    if not images:
        raise SystemExit(f"No non-mirrored images found in {IN_DIR}")

    src = images[0]
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {src}")

    fixed = apply_paper(img)

    out_img = OUT_DIR / src.name
    cv2.imwrite(str(out_img), fixed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Simple side-by-side for review.
    h = min(img.shape[0], fixed.shape[0])
    w1 = int(round(img.shape[1] * (h / img.shape[0])))
    w2 = int(round(fixed.shape[1] * (h / fixed.shape[0])))
    a = cv2.resize(img, (w1, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(fixed, (w2, h), interpolation=cv2.INTER_AREA)
    sheet = cv2.hconcat([a, b])
    sheet_path = OUT_DIR / f"{src.stem}__before_after.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    (OUT_DIR / f"{src.stem}__notes.txt").write_text(
        "This is NOT a generative redraw.\n"
        "It preserves the painted strokes and only replaces 'paper-like' pixels\n"
        "(incl. the synthetic outer white border) with a watercolor-paper texture,\n"
        "so underexposed gray paper becomes clean paper.\n",
        encoding="utf-8",
    )

    print(f"Source: {src}")
    print(f"Saved: {out_img}")
    print(f"Review sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
