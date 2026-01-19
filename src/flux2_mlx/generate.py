from __future__ import annotations

import argparse
from pathlib import Path

from .defaults import (
    DEFAULT_REPO_ID,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    DEFAULT_DTYPE,
    DEFAULT_QUANTIZE,
    DEFAULT_OUTPUT,
)
from .pipeline import Flux2Pipeline
from .image import load_images


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLUX.2 MLX inference")
    _add_generate_args(p)
    return p


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add generation arguments to a parser."""
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--repo", type=Path, default=None, help="Local repo path")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="Optional reference images")
    parser.add_argument("--quantize", type=str, choices=["none", "int8", "int4"], default=DEFAULT_QUANTIZE)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default=DEFAULT_DTYPE)
    parser.add_argument("--vae-fp16", action="store_true", help="Run VAE in float16 (overrides force_upcast)")
    parser.add_argument("--safe-attn", action="store_true", help="Compute attention in fp32 for stability")
    parser.add_argument("--compile", action="store_true", help="Compile transformer forward with mx.compile")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    parser.add_argument("--eval-freq", type=int, default=1, help="Eval every N steps (higher = faster but more memory)")


def run_generate(args: argparse.Namespace) -> None:
    """Run image generation."""
    quant = None if args.quantize == "none" else args.quantize
    pipe = Flux2Pipeline(
        repo_id=args.repo_id,
        repo_path=args.repo,
        dtype=args.dtype,
        quantize=quant,
        safe_attn=args.safe_attn,
        vae_fp16=args.vae_fp16,
        compile=args.compile,
    )

    input_images = None
    if args.input:
        input_images = load_images(args.input)

    img = pipe.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        input_images=input_images,
        verbose=args.verbose,
        eval_freq=args.eval_freq,
    )
    img.save(args.output)
    print(f"Saved {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_generate(args)


if __name__ == "__main__":
    raise SystemExit(
        "Direct script execution is not supported.\n"
        "Use: flux2-mlx --prompt '...'  (after pip install)\n"
        "  or: python -m flux2_mlx --prompt '...'"
    )
