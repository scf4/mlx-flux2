from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from .config import load_flux2_config, load_qwen3_config, load_vae_config
from .defaults import DEFAULT_REPO_ID, WEIGHT_FILES
from .image import array_to_pil
from .model import Flux2
from .sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from .text_encoder import Qwen3Embedder
from .tokenizer import Qwen3Tokenizer
from .utils import (
    align_and_load,
    align_and_load_from_torch,
    find_snapshot,
    list_safetensors,
    resolve_repo_path,
)
from .vae import AutoEncoder
from .weight_converter import convert_flux2_diffusers_weights, convert_vae_diffusers_weights


@dataclass
class GenerationConfig:
    width: int = 1024
    height: int = 1024
    num_steps: int = 4
    guidance: float = 1.0
    seed: int | None = None


class Flux2Pipeline:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        repo_path: Path | None = None,
        dtype: str = "bfloat16",
        quantize: str | None = None,
        safe_attn: bool = False,
        vae_fp16: bool = False,
        compile: bool = False,
    ):
        self.repo_id = repo_id
        self.repo_path = resolve_repo_path(repo_id, repo_path)
        if repo_path is None:
            snapshot = find_snapshot(repo_id)
            self.weights_path = snapshot if snapshot is not None else self.repo_path
        else:
            self.weights_path = self.repo_path
        self.safe_attn = safe_attn
        self.vae_fp16 = vae_fp16
        self.compile = compile

        if repo_path is not None:
            self.tokenizer_path = self.repo_path
        else:
            local_repo = Path.cwd() / repo_id.split("/")[-1]
            if (local_repo / "tokenizer" / "tokenizer.json").exists():
                self.tokenizer_path = local_repo
            else:
                self.tokenizer_path = self.weights_path
        self.dtype = getattr(mx, dtype)
        self.quantize_mode = quantize

        self._load_models()

    def _load_models(self):
        index_path = self.weights_path / "model_index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            self.is_distilled = bool(data.get("is_distilled", False))
        else:
            self.is_distilled = "base" not in self.repo_id.lower()

        flux_cfg = load_flux2_config(self.weights_path / "transformer" / "config.json")
        vae_cfg = load_vae_config(self.weights_path / "vae" / "config.json")
        qwen_cfg = load_qwen3_config(self.weights_path / "text_encoder" / "config.json")

        self.model = Flux2(flux_cfg)
        self.model.safe_attn = self.safe_attn
        self.vae = AutoEncoder(vae_cfg)

        tokenizer = Qwen3Tokenizer.from_repo(self.tokenizer_path)
        self.text_encoder = Qwen3Embedder(qwen_cfg, tokenizer, safe_attn=self.safe_attn)

        self.model.set_dtype(self.dtype)
        self.text_encoder.model.set_dtype(self.dtype)
        if self.vae_fp16:
            self.vae.force_upcast = False
            self.vae.set_dtype(mx.float16)
        elif vae_cfg.force_upcast:
            self.vae.set_dtype(mx.float32)
        else:
            self.vae.set_dtype(self.dtype)

        model_weight = None
        for weight_file in WEIGHT_FILES:
            candidate = self.weights_path / weight_file
            if candidate.exists():
                model_weight = candidate
                break
        if model_weight is not None:
            align_and_load(self.model, mx.load(str(model_weight)), strict=True)
        else:
            diffusers_path = self.weights_path / "transformer" / "diffusion_pytorch_model.safetensors"
            if not diffusers_path.exists():
                raise FileNotFoundError("Could not locate transformer weights in repo")
            raw = mx.load(str(diffusers_path))
            mapped = convert_flux2_diffusers_weights(raw, flux_cfg)
            align_and_load_from_torch(self.model, mapped, strict=True)

        vae_weight = self.weights_path / "vae" / "diffusion_pytorch_model.safetensors"
        if not vae_weight.exists():
            raise FileNotFoundError("Could not locate VAE weights")
        vae_raw = mx.load(str(vae_weight))
        vae_mapped = convert_vae_diffusers_weights(vae_raw)
        align_and_load_from_torch(self.vae, vae_mapped, strict=True)

        te_dir = self.weights_path / "text_encoder"
        shard_paths = list_safetensors(te_dir)
        if not shard_paths:
            shard_paths = list((te_dir).glob("model-*.safetensors"))

        if not shard_paths:
            base_repo_id = "black-forest-labs/FLUX.2-klein-4B"
            base_snapshot = find_snapshot(base_repo_id)
            if base_snapshot:
                te_dir = base_snapshot / "text_encoder"
                shard_paths = list_safetensors(te_dir)
                if not shard_paths:
                    shard_paths = list((te_dir).glob("model-*.safetensors"))

        if not shard_paths:
            raise FileNotFoundError(
                "Could not locate text encoder weights. "
                "Please ensure black-forest-labs/FLUX.2-klein-4B is downloaded."
            )

        te_weights = {}
        for sp in shard_paths:
            te_weights.update(mx.load(str(sp)))
        align_and_load_from_torch(self.text_encoder.model, te_weights, strict=True)

        if self.quantize_mode in {"int8", "int4"}:
            bits = 8 if self.quantize_mode == "int8" else 4
            nn.quantize(self.model, bits=bits, group_size=64)
            nn.quantize(self.text_encoder.model, bits=bits, group_size=64)

        self.model_forward = self.model
        self.vae_decode = self.vae.decode
        self._te_model_forward = self.text_encoder.model.__call__
        if self.compile:
            self.model_forward = mx.compile(self.model.__call__)
            self.vae_decode = mx.compile(self.vae.decode)
            self._te_model_forward = mx.compile(self.text_encoder.model.__call__)

        self._cached_empty_ctx = None

    def _encode_text(self, prompts: list[str]) -> mx.array:
        """Encode text prompts using (optionally compiled) text encoder."""
        input_ids, attention_mask = self.text_encoder.tokenize(prompts)
        return self._te_model_forward(input_ids, attention_mask)

    def _get_empty_ctx(self) -> mx.array:
        """Get cached empty prompt embedding for CFG."""
        if self._cached_empty_ctx is None:
            self._cached_empty_ctx = self._encode_text([""])
            mx.eval(self._cached_empty_ctx)
        return self._cached_empty_ctx

    def encode_prompt(self, prompt: str, guidance_distilled: bool) -> tuple[mx.array, mx.array]:
        if guidance_distilled:
            ctx = self._encode_text([prompt])
        else:
            ctx_empty = self._get_empty_ctx()
            ctx_prompt = self._encode_text([prompt])
            ctx = mx.concatenate([ctx_empty, ctx_prompt], axis=0)
        ctx, ctx_ids = batched_prc_txt(ctx)
        return ctx, ctx_ids

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int | None = None,
        input_images: Iterable[Image.Image] | None = None,
        guidance_distilled: bool | None = None,
        verbose: bool = False,
        eval_freq: int = 1,
    ) -> Image.Image:
        import time

        if guidance_distilled is None:
            guidance_distilled = self.is_distilled

        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("width and height must be divisible by 16")

        if seed is not None:
            mx.random.seed(seed)

        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        ctx, ctx_ids = self.encode_prompt(prompt, guidance_distilled)
        if verbose:
            mx.eval(ctx, ctx_ids)
        timings["text_encode"] = time.perf_counter() - t0

        if verbose:
            print(f"[{timings['text_encode']*1000:7.1f}ms] Text encode: {ctx.shape[1]} tokens, shape {ctx.shape}")

        img_cond_seq, img_cond_seq_ids = (None, None)
        if input_images:
            t0 = time.perf_counter()
            img_cond_seq, img_cond_seq_ids = encode_image_refs(self.vae, list(input_images))
            if verbose:
                mx.eval(img_cond_seq, img_cond_seq_ids)
            timings["ref_encode"] = time.perf_counter() - t0
            if verbose:
                print(f"[{timings['ref_encode']*1000:7.1f}ms] Ref encode: {img_cond_seq.shape[1]} tokens")

        t0 = time.perf_counter()
        shape = (1, 128, height // 16, width // 16)
        noise = mx.random.normal(shape=shape, dtype=self.dtype)
        x, x_ids = batched_prc_img(noise)
        if verbose:
            mx.eval(x, x_ids)
        timings["noise_init"] = time.perf_counter() - t0

        timesteps = get_schedule(num_steps, x.shape[1])
        if verbose:
            print(f"[{timings['noise_init']*1000:7.1f}ms] Noise init: {x.shape[1]} latent tokens")

        t0 = time.perf_counter()
        if img_cond_seq_ids is not None:
            img_input_ids = mx.concatenate([x_ids, img_cond_seq_ids], axis=1)
        else:
            img_input_ids = x_ids
        if not guidance_distilled:
            img_input_ids = mx.concatenate([img_input_ids, img_input_ids], axis=0)
        pe_x = self.model.pe_embedder(img_input_ids)
        pe_ctx = self.model.pe_embedder(ctx_ids)
        if verbose:
            mx.eval(pe_x, pe_ctx)
        timings["pe_embed"] = time.perf_counter() - t0

        if verbose:
            print(f"[{timings['pe_embed']*1000:7.1f}ms] Position embeddings")

        step_times: list[float] = []

        def _log_step(step: int, t_curr: float, t_prev: float, img: mx.array, pred: mx.array):
            step_time = step_times[-1] if step_times else 0
            if verbose:
                print(f"[{step_time*1000:7.1f}ms] Step {step+1}/{num_steps}  t={t_curr:.4f}→{t_prev:.4f}")

        t0 = time.perf_counter()
        if guidance_distilled:
            x = denoise(
                self.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                log_fn=_log_step if verbose else None,
                pe_x=pe_x,
                pe_ctx=pe_ctx,
                model_fn=self.model_forward,
                step_times=step_times if verbose else None,
                eval_freq=eval_freq,
            )
        else:
            x = denoise_cfg(
                self.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                log_fn=_log_step if verbose else None,
                pe_x=pe_x,
                pe_ctx=pe_ctx,
                model_fn=self.model_forward,
                step_times=step_times if verbose else None,
                eval_freq=eval_freq,
            )
        timings["denoise"] = time.perf_counter() - t0

        if verbose:
            avg_step = timings["denoise"] / num_steps
            print(f"[{timings['denoise']*1000:7.1f}ms] Denoise total ({num_steps} steps, {avg_step*1000:.1f}ms/step avg)")

        t0 = time.perf_counter()
        x = mx.concatenate(scatter_ids(x, x_ids), axis=0)
        if x.shape[2] == 1:
            x = x.squeeze(axis=2)
        else:
            if verbose:
                print(f"         Warning: time dimension {x.shape[2]} > 1, using t=0 slice")
            x = x[:, :, 0, :, :]
        x = x.transpose(0, 2, 3, 1)
        mx.eval(x)
        timings["scatter"] = time.perf_counter() - t0

        if verbose:
            print(f"[{timings['scatter']*1000:7.1f}ms] Scatter/reshape")

        t0 = time.perf_counter()
        img = self.vae_decode(x)
        mx.eval(img)
        timings["vae_decode"] = time.perf_counter() - t0

        if verbose:
            print(f"[{timings['vae_decode']*1000:7.1f}ms] VAE decode")

        t0 = time.perf_counter()
        result = array_to_pil(img[0])
        timings["to_pil"] = time.perf_counter() - t0

        total_time = time.perf_counter() - total_start
        if verbose:
            print(f"[{timings['to_pil']*1000:7.1f}ms] To PIL")
            print(f"[{total_time*1000:7.1f}ms] TOTAL")
        else:
            print(f"Generated in {total_time*1000:.0f}ms")

        return result
