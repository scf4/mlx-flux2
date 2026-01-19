from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten


def find_snapshot(repo_id: str) -> Path | None:
    repo_dir = repo_id.replace("/", "--")
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_dir}" / "snapshots"
    if not base.exists():
        return None
    snapshots = list(base.glob("*"))
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def resolve_repo_path(repo_id: str, local_path: Path | None = None) -> Path:
    if local_path is not None and local_path.exists():
        return local_path
    cwd_candidate = Path.cwd() / repo_id.split("/")[-1]
    if cwd_candidate.exists():
        return cwd_candidate
    snapshot = find_snapshot(repo_id)
    if snapshot is None:
        raise FileNotFoundError(
            f"Could not find local repo or HF cache for {repo_id}."
        )
    return snapshot


def load_safetensors(paths: Iterable[Path]) -> Dict[str, mx.array]:
    weights: Dict[str, mx.array] = {}
    for path in paths:
        w = mx.load(str(path))
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
