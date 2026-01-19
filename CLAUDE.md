# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install (editable mode for development)
pip install -e .

# Generate an image
flux2-mlx --prompt "your prompt" --output output.png

# Or run as module
python -m flux2_mlx --prompt "your prompt"
```

### Common CLI Options
- `--width/--height` - Image dimensions (must be divisible by 16, default 512)
- `--steps` - Denoising steps (default 4)
- `--guidance` - Guidance scale (default 1.0)
- `--seed` - Random seed for reproducibility
- `--quantize int8/int4` - Enable quantization for lower memory
- `--dtype float16/bfloat16` - Model precision (default bfloat16)
- `--compile` - Use mx.compile for potential speedup
- `--verbose` - Show timing breakdown per step
- `--input` - Reference images for image-conditioned generation

## Architecture Overview

This is a FLUX.2 image generation implementation using Apple's MLX framework, optimized for Apple Silicon. Source code lives in `src/flux2_mlx/`.

### Core Pipeline (`src/flux2_mlx/pipeline.py`)
`Flux2Pipeline` is the main entry point. It orchestrates:
1. Loading models from HuggingFace cache or local path
2. Text encoding via Qwen3
3. Denoising loop
4. VAE decoding to final image

### Model Components

**Transformer (`src/flux2_mlx/model.py`)**
- `Flux2` - Main diffusion transformer with double-stream and single-stream blocks
- Double-stream blocks process image and text separately then cross-attend
- Single-stream blocks process concatenated image+text jointly
- Uses RoPE positional embeddings via `EmbedND`

**Text Encoder (`src/flux2_mlx/text_encoder.py`)**
- `Qwen3Embedder` wraps a Qwen3 backbone for text conditioning
- Extracts hidden states from specific layers (9, 18, 27) for multi-scale features
- `Qwen3Tokenizer` handles chat template formatting
- `--safe-attn` flag enables float32 attention for numerical stability

**VAE (`src/flux2_mlx/vae.py`)**
- `AutoEncoder` with patch-based encoding (configurable patch size)
- Includes batch normalization for latent normalization
- `force_upcast` option runs in float32 for stability

**Sampling (`src/flux2_mlx/sampling.py`)**
- `denoise()` - Standard denoising for distilled models with guidance embedding
- `denoise_cfg()` - Classifier-free guidance for base models (uses dedicated compiled CFG forward)
- `get_schedule()` - Computes timestep schedule with SNR shift

### Weight Loading (`src/flux2_mlx/utils.py`, `src/flux2_mlx/weight_converter.py`)
- Automatically finds weights in HuggingFace cache or local directories
- `align_and_load_from_torch()` handles PyTorch-to-MLX weight conversion (transposes, reshapes)
- `convert_flux2_diffusers_weights()` maps diffusers naming to this implementation

### Configuration (`src/flux2_mlx/config.py`, `src/flux2_mlx/defaults.py`)
- Configs loaded from HuggingFace-style JSON files in model directories
- `Flux2Config`, `VAEConfig`, `Qwen3Config` dataclasses define model architecture

## Python API

```python
from flux2_mlx import Flux2Pipeline

pipe = Flux2Pipeline(
    repo_id="black-forest-labs/FLUX.2-klein-4B",  # or local path via repo_path=
    dtype="bfloat16",
    quantize=None,  # or "int8"/"int4"
)
image = pipe.generate(
    prompt="a sunset over mountains",
    width=512,
    height=512,
    num_steps=4,
    guidance=1.0,
)
image.save("output.png")
```

## Key Patterns

- All models use MLX's lazy evaluation; call `mx.eval(tensor)` to force computation
- Image tensors are NHWC format (MLX convention), not NCHW
- Position IDs use 4D format: (time, height, width, channel_idx)
- Text is wrapped in chat template before encoding
