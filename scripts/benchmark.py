#!/usr/bin/env python3
"""FLUX.2 MLX Benchmark Script

Measures performance of the full generation pipeline and individual components.
Supports saving/comparing baselines and statistical reporting.

Usage:
    python scripts/benchmark.py                    # Full pipeline, 3 runs
    python scripts/benchmark.py -R 5               # More runs for statistics
    python scripts/benchmark.py -C vae-decode      # Component isolation
    python scripts/benchmark.py -Q int8            # Benchmark with int8 quantization
    python scripts/benchmark.py --save baseline.json   # Save results
    python scripts/benchmark.py --compare baseline.json  # Compare to baseline
    python scripts/benchmark.py -q                 # Quick mode (1 run, 256x256)
    python scripts/benchmark.py -f                 # Skip thermal throttling check

Short flags:
    -R/--runs      -W/--width     -H/--height    -T/--steps
    -S/--seed      -Q/--quantize  -C/--component
    -q/--quick     -f/--force
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx

from flux2_mlx import Flux2Pipeline
from flux2_mlx.sampling import batched_prc_img, batched_prc_txt, get_schedule

# Constants
DEFAULT_SEED = 42366355
DEFAULT_PROMPT = "A photo of a cute avocado robot playing with paperclips, in a beautiful forest"
DEFAULT_RUNS = 3
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 4
QUICK_WIDTH = 256
QUICK_HEIGHT = 256


def check_thermal_throttling() -> tuple[bool, str]:
    """Check if Apple Silicon GPU is being thermally throttled.

    Uses NSProcessInfo.thermalState() which returns:
        0: Nominal - No thermal issues
        1: Fair - Some thermal pressure (warn but allow continue)
        2: Serious - Significant throttling occurring
        3: Critical - Severe thermal pressure

    Returns:
        (is_throttled, message) tuple
    """
    THERMAL_STATES = {
        0: ("Nominal", False, "No thermal pressure"),
        1: ("Fair", True, "Slight thermal pressure - results may vary"),
        2: ("Serious", True, "Significant thermal throttling active"),
        3: ("Critical", True, "Severe thermal throttling - benchmarks unreliable"),
    }

    try:
        from Foundation import NSProcessInfo

        info = NSProcessInfo.processInfo()
        thermal_state = info.thermalState()

        if thermal_state in THERMAL_STATES:
            name, is_throttled, desc = THERMAL_STATES[thermal_state]
            msg = f"Thermal state: {name} ({thermal_state}) - {desc}"
            return is_throttled, msg
        else:
            return False, f"Unknown thermal state: {thermal_state}"

    except ImportError:
        # PyObjC not available, fall back to pmset
        return _check_thermal_pmset()
    except Exception as e:
        return False, f"Could not check thermal state: {e}"


def _check_thermal_pmset() -> tuple[bool, str]:
    """Fallback thermal check using pmset."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout

        is_throttled = False
        details = []

        for line in output.strip().split("\n"):
            line_lower = line.lower()

            if "cpu_speed_limit" in line_lower or "speed_limit" in line_lower:
                try:
                    value = int(line.split("=")[1].strip())
                    if value < 100:
                        is_throttled = True
                        details.append(f"CPU speed limited to {value}%")
                except (IndexError, ValueError):
                    pass

            if "thermal warning level" in line_lower and "no thermal" not in line_lower:
                is_throttled = True
                details.append("Thermal warning active")

            if "performance warning level" in line_lower and "no performance" not in line_lower:
                is_throttled = True
                details.append("Performance warning active")

        if is_throttled:
            return True, "Thermal throttling detected: " + ", ".join(details)

        return False, "No thermal throttling detected"

    except Exception as e:
        return False, f"Could not check thermal state: {e}"


def prompt_continue_throttled(message: str) -> bool:
    """Prompt user to continue when throttling is detected.

    Returns:
        True if user wants to continue, False otherwise
    """
    print()
    print("=" * 60)
    print("WARNING: GPU Thermal Throttling Detected")
    print("=" * 60)
    print(message)
    print()
    print("Benchmark results may be inaccurate due to thermal throttling.")
    print("Consider letting your machine cool down for more reliable results.")
    print()

    try:
        response = input("Continue anyway? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def stats(values: list[float]) -> dict:
    """Compute statistics without numpy dependency."""
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def fmt_time(ms: float, std: float | None = None) -> str:
    """Format time in ms with optional std deviation."""
    if std is not None:
        return f"{ms:7.0f}ms ± {std:4.0f}ms"
    return f"{ms:7.0f}ms"


def fmt_stats(s: dict) -> str:
    """Format stats dict as display string."""
    return f"{fmt_time(s['mean'], s['std'])}  (min: {s['min']:.0f}, max: {s['max']:.0f})"


def clear_cache():
    """Clear MLX cache if available."""
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def sync(tensors):
    """Synchronize MLX tensors (force lazy evaluation)."""
    if isinstance(tensors, (list, tuple)):
        mx.eval(*tensors)
    else:
        mx.eval(tensors)


def print_header(config: dict, model_name: str):
    """Print benchmark header."""
    print("=" * 60)
    print("FLUX.2 MLX Benchmark")
    print("=" * 60)
    quant_str = f", {config['quantize']}" if config.get("quantize") else ""
    print(
        f"Config: {config['width']}x{config['height']}, {config['steps']} steps, "
        f"seed={config['seed']}, {config['dtype']}{quant_str}"
    )
    print(f"Model: {model_name}")
    print()


def benchmark_full_pipeline(
    pipe: Flux2Pipeline,
    width: int,
    height: int,
    steps: int,
    seed: int,
    runs: int,
) -> dict:
    """Benchmark full generation pipeline."""
    timings = {
        "total": [],
        "text_encode": [],
        "denoise": [],
        "vae_decode": [],
        "other": [],
    }
    step_times_all = []

    print(f"Full Pipeline ({runs} runs):")

    # Warmup run (discarded)
    print("  Warmup run...", end="", flush=True)
    mx.random.seed(seed)
    _ = pipe.generate(
        prompt=DEFAULT_PROMPT,
        width=width,
        height=height,
        num_steps=steps,
        seed=seed,
    )
    print(" done")
    clear_cache()

    for i in range(runs):
        mx.random.seed(seed)
        step_times: list[float] = []

        total_start = time.perf_counter()

        # Text encode
        t0 = time.perf_counter()
        ctx, ctx_ids, _ = pipe.encode_prompt(DEFAULT_PROMPT, pipe.is_distilled)
        sync((ctx, ctx_ids))
        text_time = time.perf_counter() - t0

        # Setup for denoise
        t0 = time.perf_counter()
        latent_channels = pipe.model.in_channels
        shape = (1, latent_channels, height // 16, width // 16)
        noise = mx.random.normal(shape=shape, dtype=pipe.dtype)
        x, x_ids = batched_prc_img(noise)
        sync((x, x_ids))

        timesteps = get_schedule(steps, x.shape[1])
        img_input_ids = x_ids
        if not pipe.is_distilled:
            img_input_ids = mx.concatenate([img_input_ids, img_input_ids], axis=0)
        pe_x = pipe.model.pe_embedder(img_input_ids)
        pe_ctx = pipe.model.pe_embedder(ctx_ids)
        sync((pe_x, pe_ctx))
        setup_time = time.perf_counter() - t0

        # Denoise
        t0 = time.perf_counter()
        from flux2_mlx.sampling import denoise, denoise_cfg

        guidance_vec = mx.full((x.shape[0],), 1.0, dtype=pipe.dtype)
        txt_embedded = pipe.model.embed_txt(ctx)
        guidance_embedded = None
        if pipe.model.use_guidance_embed:
            guidance_embedded = pipe.model.embed_guidance(guidance_vec)
        sync(txt_embedded)
        if guidance_embedded is not None:
            sync(guidance_embedded)

        if pipe.is_distilled:
            x_out = denoise(
                pipe.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=1.0,
                pe_x=pe_x,
                pe_ctx=pe_ctx,
                model_fn=pipe.model_forward,
                step_times=step_times,
                txt_embedded=txt_embedded,
                guidance_embedded=guidance_embedded,
            )
        else:
            if not pipe.is_distilled:
                x = mx.concatenate([x, x], axis=0)
                x_ids = mx.concatenate([x_ids, x_ids], axis=0)
            x_out = denoise_cfg(
                pipe.model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=1.0,
                pe_x=pe_x,
                pe_ctx=pe_ctx,
                model_fn=pipe.model_forward,
                model_fn_cfg=pipe.model_forward_cfg,
                step_times=step_times,
                txt_embedded=txt_embedded,
            )
        sync(x_out)
        denoise_time = time.perf_counter() - t0

        # Reshape latent
        from flux2_mlx.sampling import scatter_ids

        t0 = time.perf_counter()
        x_final = mx.concatenate(scatter_ids(x_out, x_ids if pipe.is_distilled else x_ids[:1]), axis=0)
        if x_final.shape[2] == 1:
            x_final = x_final.squeeze(axis=2)
        else:
            x_final = x_final[:, :, 0, :, :]
        x_final = x_final.transpose(0, 2, 3, 1)
        sync(x_final)
        scatter_time = time.perf_counter() - t0

        # VAE decode
        t0 = time.perf_counter()
        img = pipe.vae_decode(x_final)
        sync(img)
        vae_time = time.perf_counter() - t0

        total_time = time.perf_counter() - total_start
        other_time = total_time - text_time - denoise_time - vae_time

        timings["total"].append(total_time * 1000)
        timings["text_encode"].append(text_time * 1000)
        timings["denoise"].append(denoise_time * 1000)
        timings["vae_decode"].append(vae_time * 1000)
        timings["other"].append(other_time * 1000)
        step_times_all.append([t * 1000 for t in step_times])

        print(f"  Run {i + 1}/{runs}: {total_time * 1000:.0f}ms")
        clear_cache()

    # Compute statistics
    results = {k: stats(v) for k, v in timings.items()}
    results["step_times"] = step_times_all

    # Print results
    print()
    print(f"  Total:        {fmt_stats(results['total'])}")
    print()
    print("  Breakdown:")
    print(f"    text_encode:    {fmt_stats(results['text_encode'])}")
    avg_step = results["denoise"]["mean"] / steps
    print(f"    denoise:       {fmt_stats(results['denoise'])}  ({avg_step:.0f}ms/step)")
    print(f"    vae_decode:    {fmt_stats(results['vae_decode'])}")
    print(f"    other:           {fmt_stats(results['other'])}")

    if step_times_all and step_times_all[0]:
        # Average step times across runs
        avg_steps = [sum(run[i] for run in step_times_all) / len(step_times_all) for i in range(len(step_times_all[0]))]
        print()
        print(f"Step times: {', '.join(f'{t:.0f}' for t in avg_steps)} ms")

    return results


def benchmark_vae_decode(
    pipe: Flux2Pipeline,
    width: int,
    height: int,
    runs: int,
) -> dict:
    """Benchmark VAE decode in isolation."""
    print(f"VAE Decode ({runs} runs):")

    # Create random latent of correct shape
    # VAE uses patch size (ps), latent is (h/8/ps_h, w/8/ps_w, z_channels * ps_h * ps_w)
    cfg = pipe.vae.params
    ps_h, ps_w = cfg.ps
    latent_c = cfg.z_channels * ps_h * ps_w
    z = mx.random.normal((1, height // 8 // ps_h, width // 8 // ps_w, latent_c), dtype=pipe.dtype)
    sync(z)

    # Warmup
    print("  Warmup...", end="", flush=True)
    _ = pipe.vae_decode(z)
    sync(_)
    print(" done")
    clear_cache()

    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        img = pipe.vae_decode(z)
        sync(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.0f}ms")
        clear_cache()

    result = stats(times)
    print()
    print(f"  Result: {fmt_stats(result)}")
    return {"vae_decode": result}


def benchmark_vae_encode(
    pipe: Flux2Pipeline,
    width: int,
    height: int,
    runs: int,
) -> dict:
    """Benchmark VAE encode in isolation."""
    print(f"VAE Encode ({runs} runs):")

    # Create random image tensor (NHWC format)
    x = mx.random.uniform(0, 1, (1, height, width, 3)).astype(pipe.dtype)
    sync(x)

    # Warmup
    print("  Warmup...", end="", flush=True)
    _ = pipe.vae.encode(x)
    sync(_)
    print(" done")
    clear_cache()

    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        z = pipe.vae.encode(x)
        sync(z)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.0f}ms")
        clear_cache()

    result = stats(times)
    print()
    print(f"  Result: {fmt_stats(result)}")
    return {"vae_encode": result}


def benchmark_text_encode(
    pipe: Flux2Pipeline,
    runs: int,
) -> dict:
    """Benchmark text encoding in isolation."""
    print(f"Text Encode ({runs} runs):")

    # Warmup
    print("  Warmup...", end="", flush=True)
    ctx, ctx_ids, _ = pipe.encode_prompt(DEFAULT_PROMPT, pipe.is_distilled)
    sync((ctx, ctx_ids))
    print(" done")
    clear_cache()

    times = []
    for i in range(runs):
        # Clear cached empty context for CFG models
        pipe._cached_empty_ctx = None

        t0 = time.perf_counter()
        ctx, ctx_ids, _ = pipe.encode_prompt(DEFAULT_PROMPT, pipe.is_distilled)
        sync((ctx, ctx_ids))
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.0f}ms")
        clear_cache()

    result = stats(times)
    print()
    print(f"  Result: {fmt_stats(result)}")
    return {"text_encode": result}


def benchmark_denoise_step(
    pipe: Flux2Pipeline,
    width: int,
    height: int,
    runs: int,
) -> dict:
    """Benchmark a single denoise step in isolation."""
    print(f"Single Denoise Step ({runs} runs):")

    # Setup minimal state
    mx.random.seed(DEFAULT_SEED)
    latent_channels = pipe.model.in_channels
    shape = (1, latent_channels, height // 16, width // 16)
    noise = mx.random.normal(shape=shape, dtype=pipe.dtype)
    x, x_ids = batched_prc_img(noise)

    ctx, ctx_ids, _ = pipe.encode_prompt(DEFAULT_PROMPT, pipe.is_distilled)
    sync((ctx, ctx_ids))

    img_input_ids = x_ids
    if not pipe.is_distilled:
        img_input_ids = mx.concatenate([img_input_ids, img_input_ids], axis=0)
    pe_x = pipe.model.pe_embedder(img_input_ids)
    pe_ctx = pipe.model.pe_embedder(ctx_ids)

    guidance_vec = mx.full((x.shape[0],), 1.0, dtype=pipe.dtype)
    txt_embedded = pipe.model.embed_txt(ctx)
    guidance_embedded = None
    if pipe.model.use_guidance_embed:
        guidance_embedded = pipe.model.embed_guidance(guidance_vec)

    t_vec = mx.full((x.shape[0],), 0.5, dtype=pipe.dtype)
    sync((x, x_ids, pe_x, pe_ctx, txt_embedded, t_vec))
    if guidance_embedded is not None:
        sync(guidance_embedded)

    # Warmup
    print("  Warmup...", end="", flush=True)
    _ = pipe.model_forward(x, x_ids, t_vec, ctx, ctx_ids, guidance_vec, pe_x, pe_ctx, txt_embedded, guidance_embedded)
    sync(_)
    print(" done")
    clear_cache()

    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        pred = pipe.model_forward(x, x_ids, t_vec, ctx, ctx_ids, guidance_vec, pe_x, pe_ctx, txt_embedded, guidance_embedded)
        sync(pred)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i + 1}/{runs}: {elapsed:.0f}ms")
        clear_cache()

    result = stats(times)
    print()
    print(f"  Result: {fmt_stats(result)}")
    return {"denoise_step": result}


def save_results(results: dict, config: dict, path: str):
    """Save benchmark results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def load_baseline(path: str) -> dict:
    """Load baseline results from JSON."""
    with open(path) as f:
        return json.load(f)


def compare_results(current: dict, baseline: dict):
    """Compare current results to baseline."""
    print()
    print("=" * 60)
    print("Comparison to Baseline")
    print("=" * 60)
    print(f"Baseline timestamp: {baseline['timestamp']}")
    print()

    for key in current:
        if key == "step_times":
            continue
        if key not in baseline["results"]:
            continue

        curr_mean = current[key]["mean"]
        base_mean = baseline["results"][key]["mean"]
        diff_ms = curr_mean - base_mean
        diff_pct = (diff_ms / base_mean) * 100 if base_mean > 0 else 0

        sign = "+" if diff_ms > 0 else ""
        indicator = "SLOWER" if diff_ms > 0 else "FASTER" if diff_ms < 0 else "SAME"

        print(
            f"  {key:15} {curr_mean:7.0f}ms vs {base_mean:7.0f}ms  " f"({sign}{diff_ms:.0f}ms, {sign}{diff_pct:.1f}%) {indicator}"
        )


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 MLX Benchmark", add_help=False)
    parser.add_argument("--help", action="help", help="Show this help message and exit")
    parser.add_argument(
        "-r", "--runs", type=int, default=DEFAULT_RUNS, help=f"Number of benchmark runs (default: {DEFAULT_RUNS})"
    )
    parser.add_argument("-w", "--width", type=int, default=DEFAULT_WIDTH, help=f"Image width (default: {DEFAULT_WIDTH})")
    parser.add_argument("-h", "--height", type=int, default=DEFAULT_HEIGHT, help=f"Image height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("-t", "--steps", type=int, default=DEFAULT_STEPS, help=f"Denoising steps (default: {DEFAULT_STEPS})")
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "-Q", "--quantize", type=str, default=None, choices=["none", "int8", "int4"], help="Quantization mode (default: none)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "-c",
        "--component",
        type=str,
        default=None,
        choices=["vae-decode", "vae-encode", "text-encode", "denoise-step"],
        help="Benchmark specific component instead of full pipeline",
    )
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--compare", type=str, default=None, help="Compare results to baseline JSON file")
    parser.add_argument("-q", "--quick", action="store_true", help=f"Quick mode: 1 run, {QUICK_WIDTH}x{QUICK_HEIGHT}")
    parser.add_argument("-f", "--force", action="store_true", help="Skip thermal throttling check")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.runs = 1
        args.width = QUICK_WIDTH
        args.height = QUICK_HEIGHT

    # Normalize quantize arg
    quantize = args.quantize if args.quantize and args.quantize != "none" else None

    config = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "quantize": quantize,
        "runs": args.runs,
        "component": args.component,
    }

    # Check for thermal throttling
    if not args.force:
        is_throttled, message = check_thermal_throttling()
        if is_throttled:
            if not prompt_continue_throttled(message):
                print("Benchmark cancelled.")
                sys.exit(0)
            print()

    # Load model
    print("Loading model...", end="", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2Pipeline(dtype=args.dtype, quantize=quantize)
    load_time = time.perf_counter() - t0
    print(f" done ({load_time:.1f}s)")
    print()

    model_name = pipe.repo_id.split("/")[-1]
    print_header(config, model_name)

    # Run benchmark
    if args.component == "vae-decode":
        results = benchmark_vae_decode(pipe, args.width, args.height, args.runs)
    elif args.component == "vae-encode":
        results = benchmark_vae_encode(pipe, args.width, args.height, args.runs)
    elif args.component == "text-encode":
        results = benchmark_text_encode(pipe, args.runs)
    elif args.component == "denoise-step":
        results = benchmark_denoise_step(pipe, args.width, args.height, args.runs)
    else:
        results = benchmark_full_pipeline(pipe, args.width, args.height, args.steps, args.seed, args.runs)

    print()
    print("=" * 60)

    # Save if requested
    if args.save:
        save_results(results, config, args.save)

    # Compare if requested
    if args.compare:
        baseline = load_baseline(args.compare)
        compare_results(results, baseline)


if __name__ == "__main__":
    main()
