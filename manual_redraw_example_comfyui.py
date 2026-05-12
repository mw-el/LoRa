#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_DIR = Path("/home/matthias/_AA_ComfyUI/ComfyUI")
COMFY_INPUT = COMFY_DIR / "input"
COMFY_OUTPUT = COMFY_DIR / "output"
WORKFLOW_TEMPLATE = Path("/home/matthias/_AA_ComfyUI/workflows/charcoal_img2img.json")

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


def _paper_texture(h: int, w: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (252, 251, 248), dtype=np.float32)  # warm-ish paper
    noise = rng.normal(0.0, 4.0, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, min(h, w) * 0.02))
    fibers = rng.normal(0.0, 1.2, size=(h, w)).astype(np.float32)
    fibers = cv2.GaussianBlur(fibers, (0, 0), sigmaX=max(1.0, min(h, w) * 0.08))
    texture = base + (noise * 0.9 + fibers * 1.1)[:, :, None]
    return np.clip(texture, 238, 255).astype(np.uint8)


def prep_init_image(img_bgr: np.ndarray, enforce_paper_margin: float = 0.10) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist_from_white = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2))
    paint = (dist_from_white > 35.0) | (s > 35.0) | (v < 210.0)

    bg = ~paint
    bg = bg & (s < 40.0) & (v > 155.0)

    # Force the outer margin to be paper, regardless of what the input has there.
    margin = int(math.ceil(min(h, w) * enforce_paper_margin))
    yy, xx = np.indices((h, w), dtype=np.int32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    bg = bg | (edge_dist < margin)

    bg_u8 = (bg.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, k, iterations=2)
    bg_u8 = cv2.GaussianBlur(bg_u8, (0, 0), sigmaX=2.0)
    bg_alpha = (bg_u8.astype(np.float32) / 255.0)[:, :, None]

    paper = _paper_texture(h, w, seed=1337)

    out = img_bgr.astype(np.float32) * (1.0 - bg_alpha) + paper.astype(np.float32) * bg_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_to_max_side(img_bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        # Still make it divisible by 8.
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w == w and new_h == h:
            return img_bgr
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale = max_side / float(side)
    new_w = max(64, int(round((w * scale) / 8.0)) * 8)
    new_h = max(64, int(round((h * scale) / 8.0)) * 8)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def comfy_post(path: str, payload: dict) -> dict:
    url = f"http://{COMFY_HOST}:{COMFY_PORT}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def comfy_get_json(path: str, query: dict | None = None) -> dict:
    url = f"http://{COMFY_HOST}:{COMFY_PORT}{path}"
    if query:
        url = url + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_prompt(prompt_id: str, timeout_s: int = 900) -> dict:
    start = time.time()
    while True:
        hist = comfy_get_json(f"/history/{prompt_id}")
        if prompt_id in hist and "outputs" in hist[prompt_id]:
            return hist[prompt_id]
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")
        time.sleep(1.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # User asked to delete existing result files.
    for p in OUT_DIR.iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)

    images = list_non_mirrored_images(IN_DIR)
    if not images:
        raise SystemExit(f"No non-mirrored images found in {IN_DIR}")
    src = images[0]

    src_img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if src_img is None:
        raise SystemExit(f"Failed to read image: {src}")

    init = prep_init_image(src_img, enforce_paper_margin=0.10)
    init = resize_to_max_side(init, max_side=1024)

    init_name = f"_redraw_init_{src.stem}_{init.shape[1]}x{init.shape[0]}.png"
    init_path = COMFY_INPUT / init_name
    cv2.imwrite(str(init_path), init)

    # Build prompt text from the existing Revised description if present.
    desc_path = src.with_suffix(".txt")
    desc = desc_path.read_text(encoding="utf-8", errors="replace").strip() if desc_path.exists() else ""
    subject = src.stem.replace("-", " ")
    if desc.startswith("Watercolor Image of "):
        rest = desc[len("Watercolor Image of ") :]
        subject = (rest.split(".", 1)[0]).strip() or subject

    positive = (
        f"{subject}, redraw as a minimal airy watercolor on textured paper, "
        "main subject centered and floating, discard secondary elements, "
        "at least 10% paper margin all around, irregular breathing white space, "
        "paper white (not gray), subtle paper grain visible, "
        "reduced pigment load, calm limited palette, "
        "soft edges that feather and break into paper, rare restrained splatters, "
        "highest detail only at face/gesture focal area, everything else suggested"
    )
    negative = (
        "hard outline, hard frame, full-bleed painting, filled background, "
        "flat digital white, photorealism, glossy, oversaturated, neon, "
        "text, watermark, signature, logo"
    )

    wf = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))
    wf["1"]["inputs"]["ckpt_name"] = "dreamshaper_8_fp16.safetensors"

    # Replace charcoal LoRA with Correia style LoRA; add a second LoRA for soft washes.
    wf["2"]["inputs"]["lora_name"] = "eudes-correia-style_20251128_0548.safetensors"
    wf["2"]["inputs"]["strength_model"] = 0.85
    wf["2"]["inputs"]["strength_clip"] = 0.85

    wf["2b"] = {
        "inputs": {
            "model": ["2", 0],
            "clip": ["2", 1],
            "lora_name": "aquarelle_softwash_v10.safetensors",
            "strength_model": 0.70,
            "strength_clip": 0.70,
        },
        "class_type": "LoraLoader",
    }

    wf["3"]["inputs"]["text"] = positive
    wf["3"]["inputs"]["clip"] = ["2b", 1]
    wf["4"]["inputs"]["text"] = negative
    wf["4"]["inputs"]["clip"] = ["2b", 1]
    wf["5"]["inputs"]["image"] = init_name
    wf["6"]["inputs"]["vae"] = ["1", 2]

    # Ensure the sampler uses the stacked LoRA model.
    wf["7"]["inputs"]["model"] = ["2b", 0]
    wf["7"]["inputs"]["steps"] = 28
    wf["7"]["inputs"]["cfg"] = 6.0
    wf["7"]["inputs"]["denoise"] = 0.70
    wf["7"]["inputs"]["seed"] = 20260209

    # Disable latent resize (keep same resolution as init).
    wf["6a"]["inputs"]["width"] = int(init.shape[1])
    wf["6a"]["inputs"]["height"] = int(init.shape[0])

    prefix = f"redrawn_example_{src.stem}"
    wf["9"]["inputs"]["filename_prefix"] = prefix

    res = comfy_post("/prompt", {"prompt": wf, "client_id": "manual_redraw_example"})
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {res}")

    result = wait_for_prompt(prompt_id)
    outputs = result.get("outputs", {})
    # SaveImage node is '9' in this template.
    images_out = outputs.get("9", {}).get("images", [])
    if not images_out:
        raise RuntimeError(f"No images in history output for prompt_id={prompt_id}: {result}")
    out_meta = images_out[0]
    filename = out_meta["filename"]
    subfolder = out_meta.get("subfolder", "")
    ftype = out_meta.get("type", "output")
    if ftype != "output":
        raise RuntimeError(f"Unexpected output type: {ftype}")

    generated_path = COMFY_OUTPUT / subfolder / filename
    if not generated_path.exists():
        raise FileNotFoundError(f"ComfyUI output missing: {generated_path}")

    # Copy to our requested result folder with the same name as the source.
    out_img = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
    if out_img is None:
        raise SystemExit(f"Failed to read generated image: {generated_path}")
    out_img_path = OUT_DIR / src.name
    cv2.imwrite(str(out_img_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    (OUT_DIR / f"{src.stem}.prompt.txt").write_text(
        f"POSITIVE: {positive}\nNEGATIVE: {negative}\n"
        f"MODEL: dreamshaper_8_fp16.safetensors\n"
        f"LORA_1: eudes-correia-style_20251128_0548.safetensors (0.85)\n"
        f"LORA_2: aquarelle_softwash_v10.safetensors (0.70)\n"
        f"STEPS: 28\nCFG: 6.0\nDENOISE: 0.70\nSEED: 20260209\n"
        f"INIT: {init_name}\n"
        f"COMFY_OUTPUT: {generated_path}\n",
        encoding="utf-8",
    )

    print(f"Source: {src}")
    print(f"Init uploaded: {init_path}")
    print(f"Generated: {generated_path}")
    print(f"Saved result: {out_img_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

