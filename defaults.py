"""Default configuration values for FLUX.2 MLX."""

DEFAULT_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
WEIGHT_FILES = [
    "flux-2-klein-4b-fp8.safetensors",
    "flux-2-klein-4b.safetensors",
    "flux-2-klein-base-4b.safetensors",
]
TOKENIZER_FALLBACK_DIR = "FLUX.2-klein-base-4B"
TEXT_ENCODER_MAX_LENGTH = 512
TEXT_ENCODER_OUTPUT_LAYERS = (9, 18, 27)
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0
DEFAULT_DTYPE = "bfloat16"
DEFAULT_QUANTIZE = "none"
DEFAULT_OUTPUT = "output.png"
