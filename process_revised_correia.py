#!/usr/bin/env python3
"""Generate revised descriptions, slugs, and canvases for the Correia images."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required; please install with 'pip install pillow' and rerun this script.") from exc

SRC_DIR = Path("sources/correia")
DST_DIR = SRC_DIR / "Revised"
DST_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
STOP_WORDS = {"watercolor", "image", "painting", "artwork", "photo"}

slug_counts: dict[str, int] = {}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s_]+", " ", text)
    text = re.sub(r"(?:" + "|".join(map(re.escape, STOP_WORDS)) + r")", "", text)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    text = text.strip("-")
    return text or "image"


def unique_slug(base: str) -> str:
    count = slug_counts.get(base, 0)
    slug_counts[base] = count + 1
    return base if count == 0 else f"{base}-{count + 1}"


def subject_from_text(path: Path, fallback: str) -> str:
    if path.exists():
        text = " ".join(path.read_text(encoding="utf-8").split())
        if text:
            return text
    return fallback


def clean_subject(subject: str) -> str:
    if not subject:
        return ""
    subject = subject.strip()
    subject = re.sub(r"[\s_]+", " ", subject)
    subject = re.sub(r"\b(?:" + "|".join(map(re.escape, STOP_WORDS)) + r")\b", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\s+", " ", subject).strip()
    return subject


def build_description(subject: str) -> str:
    subject_sentence = subject.capitalize() if subject else "the scene"
    return (
        f"Watercolor Image of {subject_sentence}. "
        "The layered washes capture subtle shifts in light while the pigments gently bloom across the surface. "
        "Soft edges and warm tones invite a quiet focus, with highlights and shadows choreographing a calm, reflective stage."
    )


def add_border(image: Image.Image, border: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    new_size = (width + 2 * border, height + 2 * border)
    canvas = Image.new("RGB", new_size, "white")
    canvas.paste(image, (border, border))
    return canvas


def process_image(path: Path, destination: Path, border: int) -> None:
    with Image.open(path) as img:
        img = add_border(img, border)
        img.save(destination, quality=95)


def relevant_images() -> Iterable[Path]:
    for path in SRC_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.lower().endswith("_mirrored"):
            continue
        yield path


def main() -> None:
    processed = 0
    for source_image in relevant_images():
        base_name = source_image.name
        base_txt = source_image.with_suffix(".txt")
        subject_raw = subject_from_text(base_txt, source_image.stem)
        subject_clean = clean_subject(subject_raw)
        slug = unique_slug(slugify(subject_clean))
        ext = source_image.suffix.lower()
        with Image.open(source_image) as preview:
            short_side = min(preview.size)
        border = max(1, math.ceil(short_side * 0.1))

        description = build_description(subject_clean if subject_clean else subject_raw)
        desc_path = DST_DIR / f"{slug}.txt"
        desc_path.write_text(description + "\n", encoding="utf-8")

        dest_image = DST_DIR / f"{slug}{ext}"
        process_image(source_image, dest_image, border)

        mirrored_path = SRC_DIR / f"{source_image.stem}_mirrored{ext}"
        if mirrored_path.exists():
            mirrored_slug = f"{slug}-Mirrored"
            mirror_desc = description.rstrip() + " The mirrored counterpart reflects the same motifs in reverse."
            mirror_desc_path = DST_DIR / f"{mirrored_slug}.txt"
            mirror_desc_path.write_text(mirror_desc + "\n", encoding="utf-8")
            process_image(mirrored_path, DST_DIR / f"{mirrored_slug}{ext}", border)
        processed += 1

    print(f"Processed {processed} base images into {DST_DIR}.")


if __name__ == "__main__":
    main()
