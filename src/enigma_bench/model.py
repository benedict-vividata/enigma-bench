"""The fixed benchmark architecture.

A bidirectional transformer encoder: **no causal mask**, every position attends
to every other, and a 26-way classifier reads out a plaintext letter at each
slot. Shapes and parameter names are pinned by :mod:`enigma_bench.spec` so any
checkpoint loads into any copy of this class.

Its width and depth come from the run's model size — 3.2M parameters at
``small``, 9.0M at ``medium``, 19.1M at ``large`` (the default). That is bound
once, at import, from ``ENIGMA_BENCH_SIZE``; see :class:`enigma_bench.spec.ModelSize`.

You may not change the architecture. You may change everything about how it is
trained: data mixture, curriculum, optimiser, schedule, precision, augmentation,
auxiliary losses on the header slots, initialisation, and so on.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .spec import (
    D_FF,
    D_MODEL,
    HEAD_DIM,
    IGNORE_INDEX,
    LAYER_NORM_EPS,
    MAX_POSITIONS,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_LAYERS,
    SEQ_LEN,
    VOCAB_SIZE,
    parameter_shapes,
)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention. Deliberately unmasked."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = dropout

    def forward(self, hidden: Tensor) -> Tensor:
        """Attend over the whole sequence.

        Args:
            hidden: ``(batch, seq, d_model)`` activations.

        Returns:
            ``(batch, seq, d_model)`` attention output.
        """
        batch, seq, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        shape = (batch, seq, NUM_HEADS, HEAD_DIM)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq, D_MODEL)
        return self.out(attended)


class FeedForward(nn.Module):
    """Position-wise GELU MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, D_FF)
        self.down = nn.Linear(D_FF, D_MODEL)

    def forward(self, hidden: Tensor) -> Tensor:
        """Apply the MLP to ``(batch, seq, d_model)`` activations."""
        return self.down(F.gelu(self.up(hidden)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer block: ``x + attn(ln(x))`` then ``x + ffn(ln(x))``."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.attention = SelfAttention(dropout=dropout)
        self.ffn_norm = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.ffn = FeedForward()
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        """Run one encoder block over ``(batch, seq, d_model)`` activations."""
        hidden = hidden + self.residual_dropout(self.attention(self.attention_norm(hidden)))
        return hidden + self.residual_dropout(self.ffn(self.ffn_norm(hidden)))


class EnigmaEncoder(nn.Module):
    """The benchmark model: 256 tokens in, 256 x 26 plaintext logits out.

    Args:
        dropout: Applied to residual branches and attention weights during
            training. Carries no parameters, so it does not affect the
            checkpoint contract — pick whatever value trains best.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Embedding(MAX_POSITIONS, D_MODEL)
        self.embedding_norm = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(EncoderLayer(dropout=dropout) for _ in range(NUM_LAYERS))
        self.final_norm = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.head = nn.Linear(D_MODEL, NUM_CLASSES)
        self.apply(self._init_weights)
        # Scale the residual projections down with depth (GPT-2 style) so the
        # pre-norm stack starts near-identity.
        for name, parameter in self.named_parameters():
            if name.endswith(("attention.out.weight", "ffn.down.weight")):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * NUM_LAYERS))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Score a batch of encoded examples.

        Args:
            input_ids: ``(batch, 256)`` long tensor of vocabulary ids.

        Returns:
            ``(batch, 256, 26)`` logits over plaintext letters ``a..z``. Only the
            text slots (36 onwards) are scored by the benchmark.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] != SEQ_LEN:
            raise ValueError(
                f"expected input_ids of shape (batch, {SEQ_LEN}), got {tuple(input_ids.shape)}"
            )
        positions = torch.arange(SEQ_LEN, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(self.final_norm(hidden))


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy over non-ignored positions.

    Args:
        logits: ``(batch, seq, 26)``.
        labels: ``(batch, seq)`` with :data:`~enigma_bench.spec.IGNORE_INDEX`
            at positions that should not contribute.
    """
    return F.cross_entropy(
        logits.reshape(-1, NUM_CLASSES), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )


def parameter_count(model: nn.Module) -> int:
    """Total number of parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters())


def assert_matches_spec(model: nn.Module) -> None:
    """Fail loudly if the module drifts from :func:`~enigma_bench.spec.parameter_shapes`."""
    actual = {name: tuple(param.shape) for name, param in model.named_parameters()}
    expected = parameter_shapes()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(actual) & set(expected) if actual[name] != expected[name]
        )
        raise AssertionError(
            f"model does not match the frozen spec: missing={missing} extra={extra} "
            f"mismatched={mismatched}"
        )
