"""The frozen benchmark specification.

Everything in this module is a contract. The eval harness, the reference
model, and any checkpoint an agent submits must agree on it exactly, so that a
weights file produced by one run can be scored by any other. Changing a value
here invalidates every previously logged run — bump :data:`SPEC_VERSION` if you
ever do.

Sequence layout (256 tokens)::

    idx      field                          vocabulary
    ------   ----------------------------   ------------------------------------
    0        reflector                      <ukw-B> | <ukw-C> | <mask>
    1..3     rotor order (left,mid,right)   <rotor-I..VIII> | <mask>
    4..6     ring settings (left,mid,right) a..z | <mask>
    7..9     ground setting (left,mid,right) a..z | <mask>
    10..35   plugboard image of a..z        a..z | <mask>
    36..255  ciphertext                     a..z

The model is a bidirectional (encoder-only, **no causal mask**) transformer. It
emits a 26-way distribution over ``a..z`` at every position; only positions
36..255 are scored, against the plaintext that produced the ciphertext.

Its *width and depth* are the one thing here the operator chooses: ``small``,
``medium`` or ``large`` (the default, and the original benchmark model), picked
with ``bench.sh run --size`` and carried into every process by
``ENIGMA_BENCH_SIZE``. The choice is made before a run starts and is fixed for
its whole life — the agent cannot change it, and a checkpoint trained at one
size will not load at another. Everything else above, the task itself, is the
same at all three.

Positions 4..35 of the header are letters too, so a training script may
supervise them as an auxiliary "recover the key" objective. The benchmark never
scores those positions — see :func:`enigma_bench.tokenizer.header_labels`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .machine import ALPHABET_SIZE, ROTOR_NAMES

SPEC_VERSION: Final = 1

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

PAD_ID: Final = 0
MASK_ID: Final = 1
LETTER_OFFSET: Final = 2
REFLECTOR_OFFSET: Final = LETTER_OFFSET + ALPHABET_SIZE  # 28
ROTOR_OFFSET: Final = REFLECTOR_OFFSET + 2  # 30
VOCAB_SIZE: Final = ROTOR_OFFSET + len(ROTOR_NAMES)  # 38

#: Human-readable token names, index-aligned with the vocabulary.
TOKEN_NAMES: Final[tuple[str, ...]] = (
    "<pad>",
    "<mask>",
    *(chr(97 + i) for i in range(ALPHABET_SIZE)),
    "<ukw-B>",
    "<ukw-C>",
    *(f"<rotor-{name}>" for name in ROTOR_NAMES),
)

# --------------------------------------------------------------------------- #
# Sequence layout
# --------------------------------------------------------------------------- #

REFLECTOR_SLOT: Final = 0
ROTOR_SLOTS: Final = slice(1, 4)
RING_SLOTS: Final = slice(4, 7)
GROUND_SLOTS: Final = slice(7, 10)
PLUGBOARD_SLOTS: Final = slice(10, 36)

HEADER_LEN: Final = 36
TEXT_LEN: Final = 220
SEQ_LEN: Final = HEADER_LEN + TEXT_LEN  # 256

#: Output classes: the 26 plaintext letters. Index ``i`` means letter ``a + i``.
NUM_CLASSES: Final = ALPHABET_SIZE

#: Label value that cross-entropy ignores (PyTorch's default).
IGNORE_INDEX: Final = -100

# --------------------------------------------------------------------------- #
# Model size
#
# One benchmark, three capacities. The task, the tokenizer, the tiers and the
# sequence layout are identical at every size; only the transformer's width and
# depth change, so a `small` run measures the same skill on a model that trains
# several times faster. A single run is one size throughout: the size is chosen
# by the operator before the container starts, and everything in this process —
# the reference model, the checkpoint contract, the scorer — reads it from here.
#
# Each size follows the ordinary shape conventions rather than anything clever:
# 64 channels per attention head, an MLP four times the model width, and a
# width-to-depth (aspect) ratio inside the 64-128 band that published models
# cluster in. Only width and depth are free; the rest is derived.
# --------------------------------------------------------------------------- #

#: Channels per attention head. 64 is the near-universal choice and is what the
#: fused attention kernels are tuned for.
HEAD_DIM: Final = 64
#: The MLP's inner width, as a multiple of ``d_model``.
FFN_MULTIPLIER: Final = 4


@dataclass(frozen=True, slots=True)
class ModelSize:
    """One rung of the benchmark's capacity ladder.

    Args:
        name: The ``--size`` value that selects it.
        d_model: Residual stream width. Must be a multiple of :data:`HEAD_DIM`.
        num_layers: Number of pre-norm encoder blocks.
    """

    name: str
    d_model: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.d_model % HEAD_DIM:
            raise ValueError(
                f"size {self.name!r}: d_model {self.d_model} is not a multiple of {HEAD_DIM}"
            )

    @property
    def num_heads(self) -> int:
        """Attention heads, at :data:`HEAD_DIM` channels each."""
        return self.d_model // HEAD_DIM

    @property
    def d_ff(self) -> int:
        """Inner width of the position-wise MLP."""
        return self.d_model * FFN_MULTIPLIER

    @property
    def aspect_ratio(self) -> float:
        """``d_model / num_layers`` — the shape knob Kaplan et al. found loss flat in."""
        return self.d_model / self.num_layers


#: The ladder, roughly a factor of 2-3 in parameters per step (3.2M / 9.0M / 19.1M).
SIZES: Final[Mapping[str, ModelSize]] = MappingProxyType(
    {
        size.name: size
        for size in (
            ModelSize(name="small", d_model=256, num_layers=4),
            ModelSize(name="medium", d_model=384, num_layers=5),
            ModelSize(name="large", d_model=512, num_layers=6),
        )
    }
)

#: What a run gets when the operator does not ask for a size. This is the
#: original — and only — benchmark model, so every historical run is a `large`.
DEFAULT_SIZE_NAME: Final = "large"
SIZE_ENV: Final = "ENIGMA_BENCH_SIZE"


def resolve_size(name: str | None) -> ModelSize:
    """Look up a size by name, falling back to :data:`DEFAULT_SIZE_NAME`.

    Raises:
        ValueError: if ``name`` is not one of :data:`SIZES`.
    """
    key = (name or "").strip().lower() or DEFAULT_SIZE_NAME
    size = SIZES.get(key)
    if size is None:
        raise ValueError(f"unknown model size {name!r}; expected one of {', '.join(SIZES)}")
    return size


def active_size() -> ModelSize:
    """The size this process is configured for, read from ``ENIGMA_BENCH_SIZE``."""
    return resolve_size(os.environ.get(SIZE_ENV))


# --------------------------------------------------------------------------- #
# Model architecture — fixed for the life of a run
#
# Bound once, at import, from the active size. The architecture is not the
# agent's to change; it is the operator's to choose, before the run starts.
# --------------------------------------------------------------------------- #

SIZE: Final = active_size()
SIZE_NAME: Final = SIZE.name
D_MODEL: Final = SIZE.d_model
NUM_LAYERS: Final = SIZE.num_layers
NUM_HEADS: Final = SIZE.num_heads
D_FF: Final = SIZE.d_ff
LAYER_NORM_EPS: Final = 1e-5
#: Learned absolute position embeddings, one per sequence slot.
MAX_POSITIONS: Final = SEQ_LEN


def architecture(size: ModelSize = SIZE) -> dict[str, object]:
    """The architecture as a plain dict, for logging and for the MCP description."""
    return {
        "kind": "encoder-only transformer (bidirectional, no causal mask)",
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "d_model": size.d_model,
        "num_layers": size.num_layers,
        "num_heads": size.num_heads,
        "d_ff": size.d_ff,
        "activation": "gelu",
        "norm": "pre-layernorm",
        "layer_norm_eps": LAYER_NORM_EPS,
        "position_embedding": "learned absolute",
        "num_classes": NUM_CLASSES,
        "attention_bias": False,
        "ffn_bias": True,
    }


def parameter_shapes(size: ModelSize = SIZE) -> dict[str, tuple[int, ...]]:
    """Every parameter the reference model owns, name -> shape.

    A submitted checkpoint must match this map exactly. Derived here rather than
    from a live ``nn.Module`` so the contract can be checked without importing
    torch.

    Args:
        size: Defaults to the size this process runs at. Pass another rung of
            :data:`SIZES` to inspect a size the process is not configured for.
    """
    d_model, d_ff = size.d_model, size.d_ff
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (VOCAB_SIZE, d_model),
        "position_embedding.weight": (MAX_POSITIONS, d_model),
        "embedding_norm.weight": (d_model,),
        "embedding_norm.bias": (d_model,),
    }
    for layer in range(size.num_layers):
        prefix = f"layers.{layer}"
        shapes.update(
            {
                f"{prefix}.attention_norm.weight": (d_model,),
                f"{prefix}.attention_norm.bias": (d_model,),
                f"{prefix}.attention.qkv.weight": (3 * d_model, d_model),
                f"{prefix}.attention.out.weight": (d_model, d_model),
                f"{prefix}.ffn_norm.weight": (d_model,),
                f"{prefix}.ffn_norm.bias": (d_model,),
                f"{prefix}.ffn.up.weight": (d_ff, d_model),
                f"{prefix}.ffn.up.bias": (d_ff,),
                f"{prefix}.ffn.down.weight": (d_model, d_ff),
                f"{prefix}.ffn.down.bias": (d_model,),
            }
        )
    shapes.update(
        {
            "final_norm.weight": (d_model,),
            "final_norm.bias": (d_model,),
            "head.weight": (NUM_CLASSES, d_model),
            "head.bias": (NUM_CLASSES,),
        }
    )
    return shapes


def parameter_count(size: ModelSize = SIZE) -> int:
    """Total number of trainable parameters at ``size``."""
    total = 0
    for shape in parameter_shapes(size).values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    return total


def architecture_fingerprint(size: ModelSize = SIZE) -> str:
    """A short hash over the parameter contract, embedded in every logged run.

    Two sizes never share a fingerprint — their shapes differ — so a checkpoint
    trained under one size cannot be passed off as another.
    """
    payload = json.dumps(
        {
            "spec_version": SPEC_VERSION,
            "shapes": {name: list(shape) for name, shape in sorted(parameter_shapes(size).items())},
            "architecture": architecture(size),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
