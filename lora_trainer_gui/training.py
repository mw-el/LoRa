from __future__ import annotations

import math
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
import traceback

import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    AttnProcessor,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.lora import LoRALinearLayer

# Setup error logging
_log_file = Path.home() / ".lora_trainer_errors.log"
_logger = logging.getLogger("lora_trainer")
_logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(_log_file)
_file_handler.setLevel(logging.DEBUG)
_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_file_handler.setFormatter(_formatter)
if not _logger.handlers:
    _logger.addHandler(_file_handler)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _default_cb(msg: str) -> None:
    return


def _default_flag() -> bool:
    return False


def _list_images(image_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    # Maintain aspect ratio and center crop to a square before resizing.
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _prepare_image(path: Path, resolution: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = _center_crop_resize(img, resolution)
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return (arr * 2.0) - 1.0


class StyleDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        captions: Dict[str, str],
        resolution: int,
    ) -> None:
        self.image_paths = image_paths
        self.captions = captions
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.image_paths[idx]
        pixel_values = _prepare_image(path, self.resolution)
        caption = self.captions[path.stem]
        return {"pixel_values": pixel_values, "caption": caption}


def _collate_fn(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    captions = [str(item["caption"]) for item in batch]
    return {"pixel_values": pixel_values, "captions": captions}


def _build_captions(
    image_paths: Iterable[Path],
    style_token: str,
    use_captions: bool,
) -> Dict[str, str]:
    captions: Dict[str, str] = {}
    for path in image_paths:
        caption_text = f"a painting in the style of {style_token}"
        if use_captions:
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption_text = txt_path.read_text(encoding="utf-8").strip() or caption_text
                except Exception:
                    caption_text = caption_text
        captions[path.stem] = caption_text
    return captions


def _encode_prompts(
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Manual prompt encoding to avoid errors from missing components on single-file checkpoints.
    Supports SDXL and SD1.5 based on available attributes on the pipeline.
    """
    # Ensure tokenizers/text encoders exist for pipelines loaded from single-file checkpoints.
    if getattr(pipe, "tokenizer", None) is None:
        pipe.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    if hasattr(pipe, "tokenizer_2") and getattr(pipe, "tokenizer_2", None) is None:
        pipe.tokenizer_2 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    if getattr(pipe, "text_encoder", None) is None:
        pipe.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device, dtype=dtype)
    if hasattr(pipe, "text_encoder_2") and getattr(pipe, "text_encoder_2", None) is None:
        pipe.text_encoder_2 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device=device, dtype=dtype)

    # SDXL path: requires tokenizer_2/text_encoder_2 and added_cond_kwargs.
    is_sdxl = hasattr(pipe, "text_encoder_2")
    if is_sdxl and getattr(pipe, "text_encoder_2", None) is not None and getattr(pipe, "tokenizer_2", None) is not None:
        tokenizer_1 = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
        text_encoder_1 = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        if tokenizer_1 is None or tokenizer_2 is None or text_encoder_1 is None or text_encoder_2 is None:
            raise RuntimeError("SDXL encoding requires both tokenizers and text encoders.")

        text_inputs = tokenizer_1(
            prompts,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = text_encoder_1(text_input_ids)[0].to(dtype)

        text_inputs_2 = tokenizer_2(
            prompts,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        prompt_embeds_2 = text_encoder_2(text_input_ids_2)[0].to(dtype)

        # Concatenate per diffusers SDXL convention.
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

        added_cond_kwargs = None
        if hasattr(pipe, "_get_add_time_ids"):
            add_time_ids = None
            errors_tried = []

            # Try different signatures for _get_add_time_ids
            height = pipe.unet.config.sample_size
            width = pipe.unet.config.sample_size
            batch_size = len(prompts)

            # Attempt 1: with dtype as kwarg and batch_size
            try:
                add_time_ids = pipe._get_add_time_ids(
                    (height, width),
                    (height, width),
                    dtype=prompt_embeds.dtype,
                    batch_size=batch_size,
                )
            except TypeError as e:
                errors_tried.append(("dtype + batch_size kwarg", str(e)))
                _logger.debug(f"Attempt 1 failed: {e}")

                # Attempt 2: without dtype, with batch_size
                try:
                    add_time_ids = pipe._get_add_time_ids(
                        (height, width),
                        (height, width),
                        batch_size=batch_size,
                    )
                except TypeError as e:
                    errors_tried.append(("batch_size kwarg only", str(e)))
                    _logger.debug(f"Attempt 2 failed: {e}")

                    # Attempt 3: dtype as positional arg
                    try:
                        add_time_ids = pipe._get_add_time_ids(
                            (height, width),
                            (height, width),
                            prompt_embeds.dtype,
                        )
                    except TypeError as e:
                        errors_tried.append(("dtype positional", str(e)))
                        _logger.debug(f"Attempt 3 failed: {e}")

                        # Attempt 4: no args beyond dimensions
                        try:
                            add_time_ids = pipe._get_add_time_ids(
                                (height, width),
                                (height, width),
                            )
                        except TypeError as e:
                            errors_tried.append(("no extra args", str(e)))
                            _logger.debug(f"Attempt 4 failed: {e}")

                            # Attempt 5: with device kwarg
                            try:
                                add_time_ids = pipe._get_add_time_ids(
                                    (height, width),
                                    (height, width),
                                    dtype=prompt_embeds.dtype,
                                    device=device,
                                )
                            except TypeError as e:
                                errors_tried.append(("dtype + device kwarg", str(e)))
                                _logger.debug(f"Attempt 5 failed: {e}")
                                msg = f"Failed to call _get_add_time_ids after 5 attempts. Tried: {[x[0] for x in errors_tried]}. Details: {errors_tried[-1][1]}"
                                _logger.error(msg)
                                raise RuntimeError(msg) from e

            if add_time_ids is None:
                raise RuntimeError("_get_add_time_ids returned None")

            add_time_ids = add_time_ids.to(device=device, dtype=prompt_embeds.dtype)
            pooled = text_encoder_2(text_input_ids_2).pooler_output.to(dtype)
            added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}
        return prompt_embeds, added_cond_kwargs

    # SD1.5/manual path.
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("Pipeline does not provide tokenizer/text_encoder for prompt encoding.")
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    prompt_embeds = text_encoder(input_ids)[0]
    return prompt_embeds.to(device=device, dtype=dtype), None


def _configure_lora(unet: nn.Module, rank: int) -> List[nn.Parameter]:
    # Prefer native attention processors; fallback to LoRALinear injection on BasicTransformerBlocks.
    lora_parameters: List[nn.Parameter] = []

    if hasattr(unet, "attn_processors"):
        attn_procs = {}
        for name, processor in unet.attn_processors.items():
            hidden_size = getattr(processor, "hidden_size", None)
            cross_attention_dim = getattr(processor, "cross_attention_dim", None)
            if hidden_size is None or cross_attention_dim is None:
                continue  # skip processors missing attributes; try fallback
            if isinstance(processor, AttnProcessor2_0):
                attn_procs[name] = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                )
            else:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                )
        if attn_procs:
            unet.set_attn_processor(attn_procs)
            for proc in unet.attn_processors.values():
                for param in proc.parameters():
                    param.requires_grad = True
                    lora_parameters.append(param)
            if lora_parameters:
                return lora_parameters

    # Fallback: inject LoRA into Linear layers inside BasicTransformerBlock.
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock):
            for attr in ["attn1", "attn2"]:
                attn_mod = getattr(module, attr, None)
                if attn_mod is None:
                    continue
                for lin_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                    # Handle nested attributes like to_out.0
                    target = attn_mod
                    for part in lin_name.split("."):
                        target = getattr(target, part, None)
                    if target is None or not isinstance(target, nn.Linear):
                        continue
                    if isinstance(target, LoRALinearLayer):
                        lora_parameters.extend(target.parameters())
                        continue
                    lora_layer = LoRALinearLayer(
                        in_features=target.in_features,
                        out_features=target.out_features,
                        rank=rank,
                        network_alpha=rank,
                    )
                    # Move existing weight/bias into the new layer for continuity.
                    lora_layer.weight = target.weight
                    if target.bias is not None and hasattr(lora_layer, "bias"):
                        lora_layer.bias = target.bias
                    parent = attn_mod
                    parts = lin_name.split(".")
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], lora_layer)
                    for p in lora_layer.parameters():
                        p.requires_grad = True
                        lora_parameters.append(p)

    if not lora_parameters:
        raise RuntimeError(
            "Failed to configure LoRA layers; unsupported attention structure on this checkpoint. "
            "Please try a standard SD1.5 or SDXL base checkpoint."
        )
    return lora_parameters


def train_lora(
    base_model: str,
    model_type: str,  # "sd15" or "sdxl"
    image_dir: Path,
    output_dir: Path,
    style_token: str,
    use_captions: bool,
    resolution: int,
    lora_rank: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    use_gradient_checkpointing: bool,
    use_fp16: bool,
    max_train_steps: Optional[int],
    progress_callback: Optional[Callable[[str], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Path:
    _logger.info(f"Starting training: model={model_type}, base={base_model}, images={image_dir}")
    progress = progress_callback or _default_cb
    status = status_callback or _default_cb
    stop = stop_flag or _default_flag

    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError("Image folder does not exist.")
    image_paths = _list_images(image_dir)
    if not image_paths:
        raise ValueError("Image folder is empty or contains no supported images.")
    if not style_token.strip():
        raise ValueError("Style token is required.")

    output_dir.mkdir(parents=True, exist_ok=True)

    captions = _build_captions(image_paths, style_token=style_token, use_captions=use_captions)
    dataset = StyleDataset(image_paths, captions, resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32

    pipe_cls = StableDiffusionXLPipeline if model_type.lower() == "sdxl" else StableDiffusionPipeline
    progress(f"[{_now_ts()}] Loading base model: {base_model}")
    base_path = Path(base_model)
    try:
        load_target = base_model
        if base_path.exists() and base_path.is_dir():
            # Pick the largest checkpoint file inside the directory.
            candidates = list(base_path.glob("*.safetensors")) + list(base_path.glob("*.ckpt"))
            if not candidates:
                raise ValueError(f"No .safetensors/.ckpt found in directory {base_model}")
            candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            load_target = str(candidates[0])
            progress(f"[{_now_ts()}] Using checkpoint file: {load_target}")
            base_path = Path(load_target)

        if base_path.exists() and base_path.is_file() and base_path.suffix.lower() in {".safetensors", ".ckpt"}:
            if base_path.stat().st_size < 1_000_000:
                raise ValueError(f"Base model file at {load_target} appears empty or incomplete.")
            pipe = pipe_cls.from_single_file(
                load_target,
                torch_dtype=dtype,
            )
        else:
            pipe = pipe_cls.from_pretrained(
                base_model,
                torch_dtype=dtype,
            )
    except Exception as exc:
        _logger.error(f"Failed to load base model: {exc}", exc_info=True)
        raise ValueError(
            f"Failed to load base model '{base_model}'. "
            "Use a valid checkpoint (.safetensors/.ckpt) for local files or a Hugging Face repo ID. "
            "For SDXL, supply an SDXL base (e.g., stabilityai/stable-diffusion-xl-base-1.0); "
            "for SD 1.5, supply an SD1.5 checkpoint. "
            f"Original error: {exc}"
        ) from exc
    pipe.to(device)
    pipe.unet.train()
    if use_gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    # Freeze everything except LoRA layers.
    for param in pipe.unet.parameters():
        param.requires_grad = False

    trainable_params = _configure_lora(pipe.unet, rank=lora_rank)
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters were created.")

    # Ensure all LoRA parameters are on the correct device
    for param in trainable_params:
        param.data = param.data.to(device)

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
    if hasattr(pipe, "text_encoder_2") and getattr(pipe, "text_encoder_2") is not None:
        for param in pipe.text_encoder_2.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scheduler = pipe.scheduler
    vae = pipe.vae
    status("Starting training")

    total_steps = max_train_steps
    if total_steps is None:
        total_steps = num_epochs * math.ceil(len(dataset) / batch_size)

    completed_steps = 0
    cancelled = False

    # Checkpoint directory
    checkpoint_dir = output_dir / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int, step: int):
        """Save training checkpoint for recovery."""
        try:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch:02d}_step{step:05d}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save LoRA weights
            lora_state = pipe.unet.attn_processors
            state_dict = {}
            for name, module in lora_state.items():
                state_dict[name] = module.state_dict()
            flat_state: Dict[str, torch.Tensor] = {}
            for prefix, module_state in state_dict.items():
                for k, v in module_state.items():
                    flat_state[f"{prefix}.{k}"] = v
            save_file(flat_state, checkpoint_path / "lora_weights.safetensors")

            # Save optimizer state
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")

            # Save training metadata
            metadata = {
                "epoch": epoch,
                "step": step,
                "total_steps": total_steps,
                "loss": float(loss.item()) if 'loss' in locals() else None,
                "timestamp": datetime.now().isoformat(),
            }
            with open(checkpoint_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            progress(f"[{_now_ts()}] Checkpoint saved: {checkpoint_path.name}")
        except Exception as e:
            _logger.warning(f"Failed to save checkpoint: {e}")

    try:
        for epoch in range(num_epochs):
            status(f"Epoch {epoch + 1}/{num_epochs}")
            for batch in dataloader:
                if stop():
                    cancelled = True
                    progress(f"[{_now_ts()}] Cancellation requested; stopping after current step.")
                    break
                try:
                    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                    captions_batch = batch["captions"]

                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                        latents = latents * scaling_factor
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(
                            0,
                            scheduler.config.num_train_timesteps,
                            (latents.shape[0],),
                            device=device,
                            dtype=torch.long,
                        )
                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                        prompt_embeds, added_cond_kwargs = _encode_prompts(
                            pipe, captions_batch, device=device, dtype=dtype
                        )

                    model_input = noisy_latents
                    unet_kwargs = {"encoder_hidden_states": prompt_embeds}
                    if added_cond_kwargs is not None:
                        unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
                    with torch.cuda.amp.autocast(enabled=use_fp16 and torch.cuda.is_available()):
                        model_pred = pipe.unet(
                            model_input,
                            timesteps,
                            **unet_kwargs,
                        ).sample

                    target = noise
                    loss = F.mse_loss(model_pred.float(), target.float())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    completed_steps += 1
                    if completed_steps % 5 == 0:
                        progress(f"[{_now_ts()}] Step {completed_steps}/{total_steps} - loss: {loss.item():.4f}")
                    if completed_steps >= total_steps:
                        break
                except Exception as e:
                    _logger.error(f"Error in training step {completed_steps}: {e}", exc_info=True)
                    raise

            # Save checkpoint at end of each epoch
            # (Disabled for now - not critical, final save is what matters)
            # if not cancelled and completed_steps < total_steps:
            #     save_checkpoint(epoch, completed_steps)

            if cancelled or completed_steps >= total_steps:
                break
    except Exception as e:
        _logger.error(f"Training failed at step {completed_steps}: {e}", exc_info=True)
        raise

    if cancelled:
        status("Cancelled")
        return output_dir / "cancelled"

    # Save LoRA weights using official diffusers method
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{style_token}_{timestamp}.safetensors"
    output_path = output_dir / filename
    saved = False

    try:
        progress(f"[{_now_ts()}] Saving LoRA weights...")

        # Official method: Use pipe.save_lora_weights() which is the most reliable
        # This method properly extracts LoRA weights from the pipeline
        from tempfile import TemporaryDirectory
        import shutil

        with TemporaryDirectory() as tmp_dir:
            # Save using the official pipeline method
            pipe.save_lora_weights(tmp_dir, safe_serialization=True)

            # Find the saved safetensors file
            tmp_path = Path(tmp_dir)
            saved_files = list(tmp_path.glob("*.safetensors"))

            if saved_files:
                # Copy the saved file to our output location with our naming
                source_file = saved_files[0]
                shutil.copy(str(source_file), str(output_path))
                saved = True
                file_size = output_path.stat().st_size / (1024 * 1024)
                _logger.info(f"Successfully saved LoRA weights via official pipeline method")
                progress(f"[{_now_ts()}] Successfully saved LoRA weights ({file_size:.2f} MB).")
            else:
                raise RuntimeError("Pipeline save_lora_weights created no output file")

    except Exception as e:
        # Fallback: Try extracting from state dict with broader pattern matching
        _logger.warning(f"Official save method failed, trying fallback: {e}")
        progress(f"[{_now_ts()}] Official method failed, trying fallback extraction...")

        try:
            flat_state: Dict[str, torch.Tensor] = {}

            # Get all state dicts that might contain LoRA weights
            # Check unet, text_encoder, and pipeline state dicts
            components_to_check = {
                "unet": pipe.unet.state_dict(),
            }

            # Also check text encoder if available
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                components_to_check["text_encoder"] = pipe.text_encoder.state_dict()

            # Extract anything that looks like it could be LoRA weights
            for component_name, state_dict in components_to_check.items():
                for key, value in state_dict.items():
                    # Look for attn_processors which should contain LoRA weights
                    if "attn_processors" in key and isinstance(value, torch.Tensor):
                        flat_state[f"{component_name}.{key}"] = value.cpu()
                    # Also look for any keys with down/up patterns that indicate LoRA
                    elif ("down" in key or "up" in key) and isinstance(value, torch.Tensor):
                        if key not in flat_state:  # Avoid duplicates
                            flat_state[f"{component_name}.{key}"] = value.cpu()

            if flat_state:
                save_file(flat_state, output_path)
                saved = True
                file_size = output_path.stat().st_size / (1024 * 1024)
                _logger.info(f"Saved {len(flat_state)} tensors via fallback method ({file_size:.2f} MB)")
                progress(f"[{_now_ts()}] Fallback save successful - {len(flat_state)} tensors saved.")
            else:
                raise RuntimeError("No LoRA weights found in any component state dicts")

        except Exception as fallback_error:
            _logger.error(f"Fallback save also failed: {fallback_error}", exc_info=True)
            progress(f"[{_now_ts()}] Fallback error: {fallback_error}")
            saved = False

    if not saved:
        raise RuntimeError(f"Failed to save LoRA weights to {output_path}")

    pipe.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    status("Finished")
    progress(f"[{_now_ts()}] Training completed. Saved LoRA to: {output_path}")
    return output_path
