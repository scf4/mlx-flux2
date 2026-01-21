from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten


def resolve_repo_path(repo_id: str, local_path: Path | None = None, revision: str | None = None) -> Path:
    if local_path is not None:
        if not local_path.exists():
            raise FileNotFoundError(f"Local repo path does not exist: {local_path}")
        return local_path
    cwd_candidate = Path.cwd() / repo_id.split("/")[-1]
    if cwd_candidate.exists():
        return cwd_candidate
    return Path(snapshot_download(repo_id, revision=revision, local_files_only=True))


def load_safetensors(paths: Iterable[Path]) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for path in paths:
        w = mx.load(str(path))
        dupes = set(weights).intersection(w)
        if dupes:
            examples = sorted(list(dupes))[:5]
            raise ValueError(f"Duplicate weight keys in {path.name}: {examples}...")
        weights.update(w)
    return weights


def align_and_load(module, weights: Dict[str, mx.array], strict: bool = True) -> None:
    params = tree_flatten(module.parameters(), destination={})
    out = []
    for name, target in params.items():
        if name not in weights:
            continue
        w = weights[name]
        if w.shape != target.shape:
            if w.ndim == 4 and target.ndim == 4:
                if w.shape[0] == target.shape[0] and w.shape[1] == target.shape[3]:
                    w = w.transpose(0, 2, 3, 1)
        if w.dtype != target.dtype:
            w = w.astype(target.dtype)
        out.append((name, w))
    module.load_weights(out, strict=strict)


def align_and_load_from_torch(module, weights: Dict[str, mx.array], strict: bool = True) -> None:
    params = tree_flatten(module.parameters(), destination={})
    out = []
    for name, target in params.items():
        if name not in weights:
            continue
        w = weights[name]
        if w.ndim == 2:
            if target.ndim == 4 and target.shape[1] == 1 and target.shape[2] == 1:
                if w.shape == (target.shape[0], target.shape[3]):
                    w = w.reshape(target.shape[0], target.shape[3], 1, 1)
            if w.ndim == 2 and w.shape != target.shape and w.T.shape == target.shape:
                w = w.T
        if w.ndim == 4 and target.ndim == 4:
            if w.shape[0] == target.shape[0] and w.shape[1] == target.shape[3]:
                w = w.transpose(0, 2, 3, 1)
        if w.dtype != target.dtype:
            w = w.astype(target.dtype)
        out.append((name, w))
    module.load_weights(out, strict=strict)


def list_safetensors(dir_path: Path) -> Tuple[Path, ...]:
    return tuple(sorted(dir_path.glob("*.safetensors")))


def fuse_qkv_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Fuse separate Q/K/V projection weights into single QKV weight.

    Converts weights from format:
        model.layers.N.self_attn.q_proj.weight
        model.layers.N.self_attn.k_proj.weight
        model.layers.N.self_attn.v_proj.weight
    To:
        model.layers.N.self_attn.qkv_proj.weight

    This is a one-time transform at load time for fused QKV attention.
    """
    import re

    # Find all layer indices that have q_proj weights
    pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight")
    layer_indices = set()
    for key in weights:
        match = pattern.match(key)
        if match:
            layer_indices.add(int(match.group(1)))

    if not layer_indices:
        return weights  # No q_proj found, return unchanged

    # Create new dict with fused weights
    out = {}
    fused_prefixes = set()

    for idx in layer_indices:
        prefix = f"model.layers.{idx}.self_attn"
        q_key = f"{prefix}.q_proj.weight"
        k_key = f"{prefix}.k_proj.weight"
        v_key = f"{prefix}.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            # Concatenate Q, K, V weights along output dimension (axis 0 for transposed)
            # HuggingFace format: (out_features, in_features)
            # MLX Linear expects: (out_features, in_features) after transpose
            q_w = weights[q_key]
            k_w = weights[k_key]
            v_w = weights[v_key]
            qkv_w = mx.concatenate([q_w, k_w, v_w], axis=0)
            out[f"{prefix}.qkv_proj.weight"] = qkv_w
            fused_prefixes.add(prefix)

    # Copy all other weights, skipping the individual q/k/v that were fused
    for key, value in weights.items():
        skip = False
        for prefix in fused_prefixes:
            if key in (f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"):
                skip = True
                break
        if not skip:
            out[key] = value

    return out
