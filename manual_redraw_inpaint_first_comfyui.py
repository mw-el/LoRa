#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
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
COMFY_START_SCRIPT = Path("/home/matthias/_AA_ComfyUI/start-comfyui.sh")

WORKFLOW_TEMPLATE = Path("/home/matthias/_AA_ComfyUI/workflows/image_outpaint.json")

IN_DIR = Path("sources/correia/Revised")
OUT_DIR = Path("sources/correia/re-drawn")


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


def ensure_comfyui_running(timeout_s: int = 90) -> None:
    try:
        comfy_get_json("/system_stats")
        return
    except Exception:
        pass

    log_dir = Path("/home/matthias/_AA_ComfyUI/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "codex_redraw_inpaint_start.log"
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


def subject_from_revised_txt(txt_path: Path, fallback: str) -> str:
    if not txt_path.exists():
        return fallback
    desc = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    prefix = "Watercolor Image of "
    if desc.startswith(prefix):
        rest = desc[len(prefix) :]
        return (rest.split(".", 1)[0]).strip() or fallback
    return fallback


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


def subject_mask_largest_component(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2))

    paint = (dist > 30.0) | (s > 28.0) | (v < 235.0)
    paint_u8 = (paint.astype(np.uint8) * 255)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    paint_u8 = cv2.morphologyEx(paint_u8, cv2.MORPH_OPEN, k, iterations=1)
    paint_u8 = cv2.morphologyEx(paint_u8, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((paint_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        mask = (paint_u8 > 0).astype(np.uint8)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas) + 1)
        mask = (labels == best).astype(np.uint8)

    # Slight dilation to include edges/details, but keep it relatively tight.
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, dil, iterations=1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3.0)
    return np.clip(mask * 255.0, 0, 255).astype(np.uint8)


def background_mask(img_bgr: np.ndarray, subject_mask_u8: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    subj = (subject_mask_u8.astype(np.float32) / 255.0)
    bg = 1.0 - subj

    # Force 10% outer margin into background mask (so border becomes paper).
    margin = int(math.ceil(0.10 * min(h, w)))
    yy, xx = np.indices((h, w), dtype=np.int32)
    edge_dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy]).astype(np.float32)
    edge = (edge_dist < float(margin)).astype(np.float32)

    bg = np.clip(np.maximum(bg, edge), 0.0, 1.0)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=2.0)
    return np.clip(bg * 255.0, 0, 255).astype(np.uint8)


def write_mask_png(mask_u8: np.ndarray, path: Path) -> None:
    # ComfyUI's LoadImage provides a MASK output from the image alpha channel.
    # Write an RGBA PNG with alpha=mask to ensure the mask is read correctly.
    rgba = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = mask_u8
    rgba[:, :, 1] = mask_u8
    rgba[:, :, 2] = mask_u8
    rgba[:, :, 3] = mask_u8
    cv2.imwrite(str(path), rgba)


def write_image_png(img_bgr: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img_bgr)


def wait_for_prompt(prompt_id: str, timeout_s: int = 900) -> dict:
    start = time.time()
    while True:
        hist = comfy_get_json(f"/history/{prompt_id}")
        if prompt_id in hist and "outputs" in hist[prompt_id]:
            return hist[prompt_id]
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")
        time.sleep(1.0)


def build_outpaint_workflow(
    image_name: str,
    mask_name: str,
    positive: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    lora_correia_strength: float,
    lora_wash_strength: float,
    filename_prefix: str,
) -> dict:
    wf = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))

    wf["1"]["inputs"]["ckpt_name"] = "realisticVisionV60B1_v51HyperVAE.safetensors"

    # First LoRA node in this template.
    wf["2"]["inputs"]["lora_name"] = "eudes-correia-style_20251128_0548.safetensors"
    wf["2"]["inputs"]["strength_model"] = float(lora_correia_strength)
    wf["2"]["inputs"]["strength_clip"] = float(lora_correia_strength)

    # Stack a wash LoRA after it.
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

    wf["5"]["inputs"]["image"] = image_name
    wf["5b"]["inputs"]["image"] = mask_name

    wf["7"]["inputs"]["model"] = ["2b", 0]
    wf["7"]["inputs"]["seed"] = int(seed)
    wf["7"]["inputs"]["steps"] = int(steps)
    wf["7"]["inputs"]["cfg"] = float(cfg)
    wf["7"]["inputs"]["denoise"] = float(denoise)

    wf["9"]["inputs"]["filename_prefix"] = filename_prefix
    return wf


def comfy_run_and_fetch(prompt: dict) -> Path:
    res = comfy_post("/prompt", {"prompt": prompt, "client_id": "codex_manual_inpaint"})
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
    return generated_path


def make_contact_sheet(paths: list[Path], out_path: Path) -> None:
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(img)
    if not imgs:
        return
    target_h = min(i.shape[0] for i in imgs)
    norm = []
    for i in imgs:
        h, w = i.shape[:2]
        new_w = int(round((w / h) * target_h))
        norm.append(cv2.resize(i, (new_w, target_h), interpolation=cv2.INTER_AREA))
    sheet = cv2.hconcat(norm)
    cv2.imwrite(str(out_path), sheet)


def main() -> int:
    ensure_comfyui_running(timeout_s=90)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    src_img = resize_to_max_side(src_img, max_side=1024)
    subject = subject_from_revised_txt(src.with_suffix(".txt"), fallback=src.stem.replace("-", " "))

    # Build masks on the resized image.
    subj_mask = subject_mask_largest_component(src_img)
    bg_mask = background_mask(src_img, subj_mask)

    # Upload reference and masks to ComfyUI.
    img_name = f"_inpaint_ref_{src.stem}_{src_img.shape[1]}x{src_img.shape[0]}.png"
    bg_mask_name = f"_inpaint_mask_bg_{src.stem}_{src_img.shape[1]}x{src_img.shape[0]}.png"
    subj_mask_name = f"_inpaint_mask_subj_{src.stem}_{src_img.shape[1]}x{src_img.shape[0]}.png"

    img_path = COMFY_INPUT / img_name
    bg_mask_path = COMFY_INPUT / bg_mask_name
    subj_mask_path = COMFY_INPUT / subj_mask_name

    write_image_png(src_img, img_path)
    write_mask_png(bg_mask, bg_mask_path)
    write_mask_png(subj_mask, subj_mask_path)

    # Variant A: fix paper/background only (subject unmodified).
    pos_a = (
        "blank watercolor paper background only, warm natural paper tone, subtle paper grain, "
        "no paint strokes, no wash, no splatters, no background shapes, no objects, lots of clean negative space"
    )
    neg_a = (
        "person, human, figure, face, body, clothing, object, shape, ribbon, wave, feather, "
        "text, watermark, signature, logo"
    )
    wf_a = build_outpaint_workflow(
        image_name=img_name,
        mask_name=bg_mask_name,
        positive=pos_a,
        negative=neg_a,
        seed=2026020911,
        steps=28,
        cfg=3.8,
        denoise=0.70,
        lora_correia_strength=0.10,
        lora_wash_strength=0.10,
        filename_prefix=f"redrawA_bg_{src.stem}",
    )
    gen_a = comfy_run_and_fetch(wf_a)
    out_a = OUT_DIR / f"{src.stem}__A_bg_paper.png"
    shutil.copy2(gen_a, out_a)

    # Variant B: restyle subject only, very low denoise to preserve pose/silhouette.
    pos_b = (
        f"{subject}, redraw the SAME subject in the SAME pose, silhouette and proportions as the reference, "
        "wearing the same t-shirt and the same shorts as the reference, "
        "minimal airy watercolor on textured paper, reduced pigment load, calm limited palette, "
        "soft edges that feather and break into paper, highest detail only at face/gesture focal area"
    )
    neg_b = (
        "shirtless, bare chest, nude, different person, different pose, different clothing, extra limbs, "
        "background object, ribbon, wave, feather, abstract shape, "
        "text, watermark, signature, logo"
    )
    wf_b = build_outpaint_workflow(
        image_name=img_name,
        mask_name=subj_mask_name,
        positive=pos_b,
        negative=neg_b,
        seed=2026020912,
        steps=34,
        cfg=4.2,
        denoise=0.14,
        lora_correia_strength=0.55,
        lora_wash_strength=0.45,
        filename_prefix=f"redrawB_subj_{src.stem}",
    )
    gen_b = comfy_run_and_fetch(wf_b)
    out_b = OUT_DIR / f"{src.stem}__B_subject_wash.png"
    shutil.copy2(gen_b, out_b)

    # Variant C: two-step (A then B) to get paper right and keep subject close.
    # Use A output as the new reference image.
    two_ref = cv2.imread(str(out_a), cv2.IMREAD_COLOR)
    if two_ref is None:
        raise SystemExit(f"Failed to read intermediate: {out_a}")
    two_ref = resize_to_max_side(two_ref, max_side=1024)
    two_img_name = f"_inpaint_ref2_{src.stem}_{two_ref.shape[1]}x{two_ref.shape[0]}.png"
    two_img_path = COMFY_INPUT / two_img_name
    write_image_png(two_ref, two_img_path)

    # Rebuild subject mask on the A output (paper may change edges slightly).
    two_subj_mask = subject_mask_largest_component(two_ref)
    two_subj_mask_name = f"_inpaint_mask2_subj_{src.stem}_{two_ref.shape[1]}x{two_ref.shape[0]}.png"
    two_subj_mask_path = COMFY_INPUT / two_subj_mask_name
    write_mask_png(two_subj_mask, two_subj_mask_path)

    wf_c = build_outpaint_workflow(
        image_name=two_img_name,
        mask_name=two_subj_mask_name,
        positive=pos_b,
        negative=neg_b,
        seed=2026020913,
        steps=36,
        cfg=4.2,
        denoise=0.16,
        lora_correia_strength=0.60,
        lora_wash_strength=0.45,
        filename_prefix=f"redrawC_two_{src.stem}",
    )
    gen_c = comfy_run_and_fetch(wf_c)
    out_c = OUT_DIR / f"{src.stem}__C_two_step.png"
    shutil.copy2(gen_c, out_c)

    sheet = OUT_DIR / f"{src.stem}__contact_sheet.png"
    make_contact_sheet([out_a, out_b, out_c], sheet)

    # Record params for reproducibility.
    (OUT_DIR / f"{src.stem}__run.txt").write_text(
        f"SRC: {src}\n"
        f"REF_UPLOADED: {img_path}\n"
        f"BG_MASK: {bg_mask_path}\n"
        f"SUBJ_MASK: {subj_mask_path}\n"
        f"OUT_A: {out_a}\nOUT_B: {out_b}\nOUT_C: {out_c}\nSHEET: {sheet}\n",
        encoding="utf-8",
    )

    print(f"Saved: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
