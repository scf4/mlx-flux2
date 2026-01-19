# FLUX.2 MLX

Implementation of FLUX.2 image generation using Apple's MLX framework.

⚡️ With **FLUX.2 [klein]**, this can generate 512x512px images in ~5-6 seconds on an M3 Max (36GB) MacBook Pro.

![Example Image](outputs/avocado.png)

## Quick Start

```bash
pip install -e .
flux2-mlx --prompt "A photo of a cute avocado robot playing with paperclips in a black forest"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--width` | 512 | Image width (divisible by 16) |
| `--height` | 512 | Image height (divisible by 16) |
| `--steps` | 4 | Denoising steps |
| `--guidance` | 1.0 | Guidance scale |
| `--seed` | random | Random seed |
| `--output` | output.png | Output path |
| `--repo-id` | black-forest-labs/FLUX.2-klein-4B | HuggingFace model |
| `--repo` | - | Local model path |
| `--input` | - | Reference images |
| `--dtype` | bfloat16 | Model dtype |
| `--quantize` | none | Quantization (none/int8/int4) |
| `--compile` | off | Use mx.compile |
| `--verbose` | off | Show timing breakdown |
| `--eval-freq` | 1 | Eval every N steps (higher = faster, more memory) |

## Python API

```python
from flux2_mlx import Flux2Pipeline

pipe = Flux2Pipeline()
image = pipe.generate(prompt="a sunset over mountains")
image.save("sunset.png")
```

## License

The code in this project is licensed under the MIT License.
