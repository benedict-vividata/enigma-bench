"""The frozen architecture: the module and the declared contract must agree."""

from __future__ import annotations

import torch

from enigma_bench import spec
from enigma_bench.model import (
    EnigmaEncoder,
    assert_matches_spec,
    cross_entropy,
    parameter_count,
)


def test_vocabulary_and_layout_are_consistent() -> None:
    assert len(spec.TOKEN_NAMES) == spec.VOCAB_SIZE
    assert spec.HEADER_LEN + spec.TEXT_LEN == spec.SEQ_LEN
    assert spec.PLUGBOARD_SLOTS.stop == spec.HEADER_LEN
    assert spec.TOKEN_NAMES[spec.PAD_ID] == "<pad>"
    assert spec.TOKEN_NAMES[spec.MASK_ID] == "<mask>"
    assert spec.TOKEN_NAMES[spec.LETTER_OFFSET] == "a"


def test_fingerprint_is_stable_and_content_addressed() -> None:
    assert spec.architecture_fingerprint() == spec.architecture_fingerprint()
    assert len(spec.architecture_fingerprint()) == 16


def test_module_matches_the_declared_parameter_contract() -> None:
    model = EnigmaEncoder()
    assert_matches_spec(model)
    assert parameter_count(model) == spec.parameter_count()
    assert {name: tuple(p.shape) for name, p in model.named_parameters()} == spec.parameter_shapes()


def test_forward_shape_and_dtype() -> None:
    model = EnigmaEncoder().eval()
    tokens = torch.randint(0, spec.VOCAB_SIZE, (3, spec.SEQ_LEN))
    with torch.inference_mode():
        logits = model(tokens)
    assert logits.shape == (3, spec.SEQ_LEN, spec.NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_rejects_the_wrong_sequence_length() -> None:
    model = EnigmaEncoder().eval()
    try:
        model(torch.zeros((1, 8), dtype=torch.long))
    except ValueError as exc:
        assert "expected input_ids of shape" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("expected a ValueError for a short sequence")


def test_attention_is_bidirectional() -> None:
    """A change late in the sequence must reach early positions — no causal mask."""
    torch.manual_seed(0)
    model = EnigmaEncoder().eval()
    tokens = torch.full((1, spec.SEQ_LEN), spec.LETTER_OFFSET, dtype=torch.long)
    with torch.inference_mode():
        before = model(tokens)
        tokens[0, -1] = spec.LETTER_OFFSET + 7
        after = model(tokens)
    assert not torch.allclose(before[0, 0], after[0, 0], atol=1e-6)


def test_cross_entropy_ignores_masked_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, spec.SEQ_LEN, spec.NUM_CLASSES)
    labels = torch.full((2, spec.SEQ_LEN), spec.IGNORE_INDEX)
    labels[:, spec.HEADER_LEN] = 3
    loss = cross_entropy(logits, labels)
    direct = torch.nn.functional.cross_entropy(logits[:, spec.HEADER_LEN], torch.tensor([3, 3]))
    assert torch.allclose(loss, direct, atol=1e-6)


def test_dropout_does_not_change_the_checkpoint_contract() -> None:
    assert_matches_spec(EnigmaEncoder(dropout=0.3))
