#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _stable_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & 0xFFFFFFFF


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _subject_from_revised_description(desc: str | None) -> str | None:
    if not desc:
        return None
    prefix = "Watercolor Image of "
    if desc.startswith(prefix):
        rest = desc[len(prefix) :]
        # Take first sentence.
        dot = rest.find(".")
        return (rest[:dot] if dot != -1 else rest).strip() or None
    return None


def _paper_sample(img_bgr: np.ndarray, border_frac: float = 0.05) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bh = max(1, int(round(h * border_frac)))
    bw = max(1, int(round(w * border_frac)))
    strips = [
        img_bgr[:bh, :, :],
        img_bgr[-bh:, :, :],
        img_bgr[:, :bw, :],
        img_bgr[:, -bw:, :],
    ]
    sample = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0).astype(np.float32)
    # Prefer bright pixels (paper) for estimation.
    lum = 0.0722 * sample[:, 0] + 0.7152 * sample[:, 1] + 0.2126 * sample[:, 2]  # BGR luminance approx
    candidates = sample[lum > 220]
    if candidates.shape[0] < 500:  # fallback if border is too painted
        candidates = sample
    return np.median(candidates, axis=0)


def _white_balance(img_bgr: np.ndarray, paper_bgr: np.ndarray) -> np.ndarray:
    paper_bgr = np.clip(paper_bgr, 1.0, 255.0)
    gains = 255.0 / paper_bgr
    out = img_bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _main_motif_mask(img_bgr_bal: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr_bal, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2)).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr_bal, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Paint tends to be either farther from white OR has some saturation with slightly lower value.
    mask = (dist > 25.0) | ((s > 18.0) & (v < 248.0))
    mask_u8 = (mask.astype(np.uint8)) * 255

    # Clean small speckles and fill small gaps.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask_u8 > 0).astype(np.uint8)
    # Pick the largest non-background component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas) + 1)
    return (labels == best).astype(np.uint8)


def _gaussian_focus(h: int, w: int, cx: float, cy: float, sigma_frac: float = 0.22) -> np.ndarray:
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    sx = max(1.0, w * sigma_frac)
    sy = max(1.0, h * sigma_frac)
    g = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    return g.astype(np.float32)


def _paper_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 255, dtype=np.float32)
    noise = rng.normal(0.0, 3.0, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, min(h, w) * 0.02))
    texture = base + noise[:, :, None]
    return np.clip(texture, 245, 255).astype(np.uint8)


def _reduce_palette_and_contrast(paint_bgr: np.ndarray, focus: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(paint_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # More reduction away from the focus.
    sat_factor = 0.70 + 0.25 * focus
    s = np.clip(s * sat_factor, 0, 255)

    # Slightly lift midtones and reduce contrast.
    v = np.clip(v * 0.97 + 4.0, 0, 255)
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Smooth washes (keeps edges relatively intact).
    return cv2.bilateralFilter(out, d=0, sigmaColor=35, sigmaSpace=5)


def redraw_variant(img_bgr: np.ndarray, name_for_seed: str) -> tuple[np.ndarray, dict[str, float]]:
    h, w = img_bgr.shape[:2]
    seed = _stable_seed(name_for_seed)

    paper_bgr = _paper_sample(img_bgr)
    balanced = _white_balance(img_bgr, paper_bgr)

    main_mask = _main_motif_mask(balanced)  # 0/1
    mask_area = float(main_mask.mean())

    moments = cv2.moments((main_mask * 255).astype(np.uint8))
    if moments["m00"] > 1e-3:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = w / 2.0, h / 2.0
    focus = _gaussian_focus(h, w, cx, cy)

    # Build an "importance" map: edges + darkness, biased to the focus.
    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edges = edges / (edges.max() + 1e-6)
    darkness = (255.0 - gray) / 255.0
    importance = 0.55 * edges + 0.45 * darkness
    importance = importance * (0.65 + 0.35 * focus)
    importance = importance / (importance.max() + 1e-6)

    # Paint alpha: keep mostly where the motif exists and where importance suggests it matters.
    alpha = (main_mask.astype(np.float32)) * (importance ** 0.75)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0)

    # Irregular, "breathing" edges (low-frequency noise applied at the transition).
    rng = np.random.default_rng(seed)
    lf = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    lf = cv2.GaussianBlur(lf, (0, 0), sigmaX=max(3.0, min(h, w) * 0.015))
    alpha = np.clip(alpha + 0.07 * lf, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)

    # Enforce a minimum paper margin around the whole image (>=10% of the shorter side).
    margin = int(math.ceil(0.10 * min(h, w)))
    feather = max(8.0, margin * 0.25)
    yy, xx = np.indices((h, w), dtype=np.float32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    # Slightly vary the margin threshold so it doesn't read as a rigid frame.
    edge_thresh = margin + (lf * (0.10 * margin))
    edge_alpha = np.clip((edge_dist - edge_thresh) / feather, 0.0, 1.0)
    alpha *= edge_alpha

    # Prepare paper and paint.
    paper = _paper_texture(h, w, seed=seed + 1)
    paint = _reduce_palette_and_contrast(balanced, focus=focus)

    # Final blend.
    alpha3 = alpha[:, :, None].astype(np.float32)
    out = paint.astype(np.float32) * alpha3 + paper.astype(np.float32) * (1.0 - alpha3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    metrics = {
        "mask_area_ratio": mask_area,
        "focus_x": float(cx / max(1.0, w)),
        "focus_y": float(cy / max(1.0, h)),
    }
    return out, metrics


def iter_images(in_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        if stem.endswith("-Mirrored") or stem.endswith("_mirrored") or stem.lower().endswith("mirrored"):
            continue
        paths.append(p)
    return sorted(paths, key=lambda x: x.name.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="sources/correia/Revised")
    ap.add_argument("--out-dir", default="sources/correia/re-drawn")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(in_dir)
    if not images:
        raise SystemExit(f"No images found in: {in_dir}")

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        revised_desc = _read_text_if_exists(img_path.with_suffix(".txt"))
        subject = _subject_from_revised_description(revised_desc) or img_path.stem.replace("-", " ")

        out_img, metrics = redraw_variant(img, name_for_seed=img_path.name)
        out_img_path = out_dir / img_path.name
        cv2.imwrite(str(out_img_path), out_img)

        analysis = (
            f"Motifklaerung: Hauptmotiv = {subject}. Sekundaeres wird zugunsten eines klaren, zentrierten Motivs reduziert.\n"
            "Weissraum: Papierweiss dominiert; das Motiv schwebt im Bild mit atmendem, unregelmaessigem Uebergang zum Rand.\n"
            "Tonwerte: Pseudo-Weissbereiche werden aufgehellt (Papierwirkung), Kontrast im Randbereich reduziert.\n"
            "Farbauftrag: Farbflächen sind sparsamer; Details bleiben vor allem im Fokusbereich erhalten.\n"
            "Raender: Kanten laufen weich aus; Pigment bricht aus und duennt in den Weissraum aus.\n"
            f"Schwerpunkt: Fokus naeherungsweise bei ({metrics['focus_x']:.3f}, {metrics['focus_y']:.3f}) relativ zur Bildflaeche.\n"
            f"Kennzahlen: mask_area_ratio={metrics['mask_area_ratio']:.4f}.\n"
            "Schlusskontrolle: Wirkt es luftiger und koennte man noch mehr weglassen, ohne das Motiv zu verlieren?\n"
        )
        (out_dir / f"{img_path.stem}.analysis.txt").write_text(analysis, encoding="utf-8")

        prompt = (
            f"{subject}, centered and floating on clean white textured paper, lots of negative space, "
            "minimal watercolor wash, limited calm palette, soft irregular edges that fade into paper, "
            "subtle pigment blooms, few restrained splatters, sharpest detail only at the focal area, "
            "everything else suggested, airy composition"
        )
        negative = (
            "hard outline, hard frame, full-bleed painting, filled background, digital flat white, "
            "heavy saturation, photo, realism, text, watermark, signature"
        )
        (out_dir / f"{img_path.stem}.prompt.txt").write_text(
            f"PROMPT: {prompt}\nNEGATIVE: {negative}\n", encoding="utf-8"
        )

    print(f"Wrote {len(images)} re-drawn variants to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
