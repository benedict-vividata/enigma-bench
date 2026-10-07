"""Reading and writing benchmark checkpoints.

The submitted file is a ``torch.save`` archive loaded with ``weights_only=True``
— it may contain tensors, plain containers and primitives, and nothing that can
execute code. Two shapes are accepted:

1. A wrapper dict ``{"format": 1, "state_dict": {...}, "metadata": {...}}``
   (what :func:`save_checkpoint` writes), or
2. a bare ``state_dict`` mapping parameter names to tensors.

``_orig_mod.`` (``torch.compile``) and ``module.`` (``DistributedDataParallel``)
prefixes are stripped, and half-precision weights are upcast to fp32, so a
checkpoint saved straight out of a compiled/AMP training loop loads fine.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .spec import (
    SIZE_NAME,
    SIZES,
    SPEC_VERSION,
    ModelSize,
    architecture_fingerprint,
    parameter_count,
    parameter_shapes,
)

CHECKPOINT_FORMAT = 1

_WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "weights")
_STRIPPABLE_PREFIXES = ("_orig_mod.", "module.")


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded into the fixed architecture."""


def save_checkpoint(
    model: nn.Module, path: Path, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Write ``model``'s weights in the benchmark's expected wrapper format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "spec_version": SPEC_VERSION,
        "model_size": SIZE_NAME,
        "architecture_fingerprint": architecture_fingerprint(),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return path


def _unwrap(payload: object, source: Path) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            f"{source}: expected a dict at the top level, got {type(payload).__name__}"
        )
    for key in _WRAPPER_KEYS:
        inner = payload.get(key)
        if isinstance(inner, Mapping):
            return inner
    return payload


def _normalise_key(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _STRIPPABLE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _other_size(state: Mapping[str, Tensor]) -> ModelSize | None:
    """The size these weights *are*, when they are a whole model of the wrong one.

    Training against a different ``ENIGMA_BENCH_SIZE`` than the run scores at
    produces a wall of shape mismatches. Recognising the actual cause lets the
    error name it instead.
    """
    shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    for name, size in SIZES.items():
        if name != SIZE_NAME and shapes == parameter_shapes(size):
            return size
    return None


def load_state_dict(path: Path) -> dict[str, Tensor]:
    """Load and validate a checkpoint against the frozen architecture.

    Raises:
        CheckpointError: if the file is missing, unreadable, contains
            non-tensor parameters, or does not match
            :func:`~enigma_bench.spec.parameter_shapes` exactly.
    """
    if not path.is_file():
        raise CheckpointError(f"{path}: no such file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises a wide variety here
        raise CheckpointError(
            f"{path}: could not be loaded with weights_only=True ({exc}). Save the file with "
            "torch.save of plain tensors — pickled objects are rejected."
        ) from exc

    raw = _unwrap(payload, path)
    state: dict[str, Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise CheckpointError(f"{path}: parameter name {key!r} is not a string")
        if not isinstance(value, Tensor):
            continue  # metadata sitting alongside the weights in a bare dict
        state[_normalise_key(key)] = value

    expected = parameter_shapes()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    mismatched = sorted(
        f"{name}: got {tuple(state[name].shape)}, want {expected[name]}"
        for name in set(state) & set(expected)
        if tuple(state[name].shape) != expected[name]
    )
    if missing or extra or mismatched:
        other = _other_size(state)
        if other is not None:
            raise CheckpointError(
                f"{path}: these weights are a complete '{other.name}' model "
                f"({parameter_count(other):,} parameters), but this run is "
                f"'{SIZE_NAME}' ({parameter_count():,}). Retrain at the run's size — "
                f"enigma_bench.model takes it from the environment, so instantiating "
                f"EnigmaEncoder() in this container is enough."
            )
        raise CheckpointError(
            f"{path}: checkpoint does not match the fixed architecture "
            f"(model size {SIZE_NAME}, fingerprint {architecture_fingerprint()}).\n"
            f"  missing parameters : {missing or 'none'}\n"
            f"  unexpected entries : {extra or 'none'}\n"
            f"  wrong shapes       : {mismatched or 'none'}"
        )
    return {name: tensor.detach().to(torch.float32) for name, tensor in state.items()}


def load_model(path: Path, device: str = "cpu") -> nn.Module:
    """Instantiate the fixed architecture and load ``path`` into it, in eval mode."""
    from .model import EnigmaEncoder  # local import: keeps torch off the import path of spec

    model = EnigmaEncoder()
    model.load_state_dict(load_state_dict(path))
    return model.to(device).eval()


def describe_checkpoint(path: Path) -> dict[str, object]:
    """Validate a checkpoint and summarise it, without scoring anything.

    Returns a dict with ``valid`` plus either an ``error`` string or the
    checkpoint's size, dtypes and any metadata the trainer embedded.
    """
    result: dict[str, object] = {
        "path": str(path),
        "model_size": SIZE_NAME,
        "architecture_fingerprint": architecture_fingerprint(),
        "expected_parameter_count": parameter_count(),
    }
    try:
        state = load_state_dict(path)
    except CheckpointError as exc:
        result["valid"] = False
        result["error"] = str(exc)
        return result

    result["valid"] = True
    result["parameter_count"] = sum(tensor.numel() for tensor in state.values())
    result["file_size_bytes"] = path.stat().st_size
    result["has_non_finite_weights"] = any(
        not bool(torch.isfinite(tensor).all()) for tensor in state.values()
    )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
            result["metadata"] = dict(payload["metadata"])
    except Exception:  # noqa: S110 - metadata is best-effort only
        pass
    return result
