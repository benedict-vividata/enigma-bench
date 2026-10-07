"""Checkpoint loading: the format contract an agent has to hit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from enigma_bench.checkpoint import (
    CheckpointError,
    describe_checkpoint,
    load_model,
    load_state_dict,
    save_checkpoint,
)
from enigma_bench.model import EnigmaEncoder
from enigma_bench.spec import parameter_shapes


def test_round_trip_through_save_and_load(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = EnigmaEncoder()
    path = save_checkpoint(model, tmp_path / "m.pt", metadata={"step": 10})
    loaded = load_state_dict(path)
    assert set(loaded) == set(parameter_shapes())
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded[name], tensor)


def test_bare_state_dict_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bare.pt"
    torch.save(EnigmaEncoder().state_dict(), path)
    assert set(load_state_dict(path)) == set(parameter_shapes())


def test_compile_and_ddp_prefixes_are_stripped(tmp_path: Path) -> None:
    state = EnigmaEncoder().state_dict()
    prefixed = {f"module._orig_mod.{name}": value for name, value in state.items()}
    path = tmp_path / "wrapped.pt"
    torch.save({"state_dict": prefixed}, path)
    assert set(load_state_dict(path)) == set(parameter_shapes())


def test_half_precision_weights_are_upcast(tmp_path: Path) -> None:
    state = {name: value.to(torch.bfloat16) for name, value in EnigmaEncoder().state_dict().items()}
    path = tmp_path / "bf16.pt"
    torch.save(state, path)
    loaded = load_state_dict(path)
    assert all(tensor.dtype is torch.float32 for tensor in loaded.values())


def test_non_tensor_entries_beside_the_weights_are_ignored(tmp_path: Path) -> None:
    payload = dict(EnigmaEncoder().state_dict())
    payload["training_step"] = 4200
    path = tmp_path / "mixed.pt"
    torch.save(payload, path)
    assert set(load_state_dict(path)) == set(parameter_shapes())


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no such file"):
        load_state_dict(tmp_path / "absent.pt")


def test_wrong_architecture_is_rejected_with_a_diff(tmp_path: Path) -> None:
    state = EnigmaEncoder().state_dict()
    del state["head.bias"]
    state["something.extra"] = torch.zeros(4)
    state["head.weight"] = torch.zeros(26, 8)
    path = tmp_path / "wrong.pt"
    torch.save(state, path)
    with pytest.raises(CheckpointError) as excinfo:
        load_state_dict(path)
    message = str(excinfo.value)
    assert "head.bias" in message
    assert "something.extra" in message
    assert "wrong shapes" in message


def test_a_non_dict_payload_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tensor.pt"
    torch.save(torch.zeros(3), path)
    with pytest.raises(CheckpointError, match="expected a dict"):
        load_state_dict(path)


def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a torch archive")
    with pytest.raises(CheckpointError, match="weights_only=True"):
        load_state_dict(path)


def test_describe_reports_validity_both_ways(tmp_path: Path) -> None:
    good = save_checkpoint(EnigmaEncoder(), tmp_path / "good.pt", metadata={"note": "hi"})
    report = describe_checkpoint(good)
    assert report["valid"] is True
    assert report["has_non_finite_weights"] is False
    assert report["metadata"] == {"note": "hi"}

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"nope")
    assert describe_checkpoint(bad)["valid"] is False


def test_load_model_returns_an_eval_mode_module(tmp_path: Path) -> None:
    path = save_checkpoint(EnigmaEncoder(), tmp_path / "m.pt")
    model = load_model(path)
    assert not model.training
