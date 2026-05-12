#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
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
COMFY_START_SCRIPT = Path("/home/matthias/_AA_ComfyUI/start-comfyui.sh")

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

    # Consider "paint" anything that is not close-to-paper in color OR is saturated OR is dark.
    paint = (dist_from_white > 35.0) | (s > 35.0) | (v < 210.0)

    # Background candidate: low saturation + bright-ish.
    bg = ~paint
    bg = bg & (s < 45.0) & (v > 150.0)

    # Force the outer margin to be paper (replaces hard white frame, too).
    margin = int(math.ceil(min(h, w) * enforce_paper_margin))
    yy, xx = np.indices((h, w), dtype=np.int32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    bg = bg | (edge_dist < margin)

    bg_u8 = (bg.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, k, iterations=2)
    bg_u8 = cv2.GaussianBlur(bg_u8, (0, 0), sigmaX=2.2)
    bg_alpha = (bg_u8.astype(np.float32) / 255.0)[:, :, None]

    paper = _paper_texture(h, w, seed=1337)

    out = img_bgr.astype(np.float32) * (1.0 - bg_alpha) + paper.astype(np.float32) * bg_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_to_max_side(img_bgr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        new_w = max(64, int(round(w / 8.0)) * 8)
        new_h = max(64, int(round(h / 8.0)) * 8)
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


def ensure_comfyui_running(timeout_s: int = 90) -> None:
    try:
        comfy_get_json("/system_stats")
        return
    except Exception:
        pass

    log_dir = Path("/home/matthias/_AA_ComfyUI/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "codex_redraw_start.log"
    with log_path.open("ab") as log_fp:
        subprocess.Popen(
            ["bash", str(COMFY_START_SCRIPT)],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    start = time.time()
    while True:
        try:
            comfy_get_json("/system_stats")
            return
        except Exception:
            if time.time() - start > timeout_s:
                raise TimeoutError("ComfyUI did not become ready on port 8188 in time.")
            time.sleep(1.0)


def subject_from_revised_txt(txt_path: Path, fallback: str) -> str:
    if not txt_path.exists():
        return fallback
    desc = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    prefix = "Watercolor Image of "
    if desc.startswith(prefix):
        rest = desc[len(prefix) :]
        return (rest.split(".", 1)[0]).strip() or fallback
    return fallback


def build_workflow(
    init_name: str,
    w: int,
    h: int,
    positive: str,
    negative: str,
    seed: int,
    denoise: float,
    cfg: float,
    steps: int,
    lora_correia_strength: float,
    lora_wash_strength: float,
    filename_prefix: str,
) -> dict:
    wf = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))

    # Base model present in /home/matthias/_AA_ComfyUI/models/checkpoints.
    wf["1"]["inputs"]["ckpt_name"] = "realisticVisionV60B1_v51HyperVAE.safetensors"

    # Replace charcoal LoRA with Correia style LoRA and add watercolor wash LoRA stacked after it.
    wf["2"]["inputs"]["lora_name"] = "eudes-correia-style_20251128_0548.safetensors"
    wf["2"]["inputs"]["strength_model"] = float(lora_correia_strength)
    wf["2"]["inputs"]["strength_clip"] = float(lora_correia_strength)

    wf["2b"] = {
        "inputs": {
            "model": ["2", 0],
            "clip": ["2", 1],
            "lora_name": "aquarelle_softwash_v10.safetensors",
            "strength_model": float(lora_wash_strength),
            "strength_clip": float(lora_wash_strength),
        },
        "class_type": "LoraLoader",
    }

    wf["3"]["inputs"]["text"] = positive
    wf["3"]["inputs"]["clip"] = ["2b", 1]
    wf["4"]["inputs"]["text"] = negative
    wf["4"]["inputs"]["clip"] = ["2b", 1]

    wf["5"]["inputs"]["image"] = init_name
    wf["6"]["inputs"]["vae"] = ["1", 2]

    wf["7"]["inputs"]["model"] = ["2b", 0]
    wf["7"]["inputs"]["seed"] = int(seed)
    wf["7"]["inputs"]["steps"] = int(steps)
    wf["7"]["inputs"]["cfg"] = float(cfg)
    wf["7"]["inputs"]["denoise"] = float(denoise)

    # Keep same resolution as init.
    wf["6a"]["inputs"]["width"] = int(w)
    wf["6a"]["inputs"]["height"] = int(h)

    wf["9"]["inputs"]["filename_prefix"] = filename_prefix
    return wf


def make_contact_sheet(paths: list[Path], out_path: Path) -> None:
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(img)
    if not imgs:
        return
    # Normalize heights.
    target_h = min(i.shape[0] for i in imgs)
    norm = []
    for i in imgs:
        h, w = i.shape[:2]
        new_w = int(round((w / h) * target_h))
        norm.append(cv2.resize(i, (new_w, target_h), interpolation=cv2.INTER_AREA))
    sheet = cv2.hconcat(norm)
    cv2.imwrite(str(out_path), sheet)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    ensure_comfyui_running(timeout_s=90)

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

    src_img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if src_img is None:
        raise SystemExit(f"Failed to read image: {src}")

    init = prep_init_image(src_img, enforce_paper_margin=0.10)
    init = resize_to_max_side(init, max_side=args.max_side)

    init_name = f"_redraw_init_{src.stem}_{init.shape[1]}x{init.shape[0]}.png"
    init_path = COMFY_INPUT / init_name
    cv2.imwrite(str(init_path), init)

    subject = subject_from_revised_txt(src.with_suffix(".txt"), fallback=src.stem.replace("-", " "))

    positive = (
        f"{subject}, redraw as a minimal airy watercolor on textured paper, "
        "keep the same pose, silhouette, proportions and clothing as the reference image, "
        "main subject centered and floating, discard secondary elements, "
        "paper white (not gray), subtle paper grain visible, "
        "at least 10% paper margin all around with irregular breathing white space, "
        "reduced pigment load, calm limited palette, "
        "soft edges that feather and break into paper, background stays blank paper (no shapes), "
        "rare restrained splatters, "
        "highest detail only at face/gesture focal area, everything else suggested"
    )
    negative = (
        "hard outline, hard frame, full-bleed painting, filled background, "
        "flat digital white, photorealism, glossy, oversaturated, neon, "
        "different person, different pose, different clothing, extra limbs, "
        "ribbon, swoosh, wave, abstract shape, background object, "
        "text, watermark, signature, logo"
    )

    variants = [
        # name, denoise, cfg, steps, correia_strength, wash_strength, seed
        ("v1_strict", 0.32, 4.6, 30, 0.70, 0.55, 2026020901),
        ("v2_close", 0.40, 4.9, 34, 0.75, 0.55, 2026020902),
        ("v3_looser", 0.48, 5.2, 36, 0.80, 0.50, 2026020903),
    ]

    saved: list[Path] = []
    for name, denoise, cfg, steps, s1, s2, seed in variants:
        prefix = f"redrawn_{src.stem}_{name}"
        wf = build_workflow(
            init_name=init_name,
            w=int(init.shape[1]),
            h=int(init.shape[0]),
            positive=positive,
            negative=negative,
            seed=seed,
            denoise=denoise,
            cfg=cfg,
            steps=steps,
            lora_correia_strength=s1,
            lora_wash_strength=s2,
            filename_prefix=prefix,
        )

        res = comfy_post("/prompt", {"prompt": wf, "client_id": f"manual_redraw_{name}"})
        prompt_id = res.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Unexpected /prompt response: {res}")
        result = wait_for_prompt(prompt_id)
        outputs = result.get("outputs", {})
        images_out = outputs.get("9", {}).get("images", [])
        if not images_out:
            raise RuntimeError(f"No images in history output for prompt_id={prompt_id}: {result}")
        out_meta = images_out[0]
        generated_path = COMFY_OUTPUT / out_meta.get("subfolder", "") / out_meta["filename"]
        if not generated_path.exists():
            raise FileNotFoundError(f"ComfyUI output missing: {generated_path}")

        out_img = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        if out_img is None:
            raise SystemExit(f"Failed to read generated image: {generated_path}")

        out_path = OUT_DIR / f"{src.stem}__{name}.jpg"
        cv2.imwrite(str(out_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(out_path)

        (OUT_DIR / f"{src.stem}__{name}.prompt.txt").write_text(
            f"POSITIVE: {positive}\nNEGATIVE: {negative}\n"
            f"MODEL: realisticVisionV60B1_v51HyperVAE.safetensors\n"
            f"LORA_1: eudes-correia-style_20251128_0548.safetensors ({s1})\n"
            f"LORA_2: aquarelle_softwash_v10.safetensors ({s2})\n"
            f"STEPS: {steps}\nCFG: {cfg}\nDENOISE: {denoise}\nSEED: {seed}\n"
            f"INIT: {init_name}\n"
            f"COMFY_OUTPUT: {generated_path}\n",
            encoding="utf-8",
        )

    sheet_path = OUT_DIR / f"{src.stem}__contact_sheet.jpg"
    make_contact_sheet(saved, sheet_path)

    print(f"Source: {src}")
    print(f"Init uploaded: {init_path}")
    for p in saved:
        print(f"Saved: {p}")
    print(f"Sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
