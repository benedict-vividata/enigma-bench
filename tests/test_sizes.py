"""The model-size ladder: shape conventions, continuity, and cross-size safety."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from enigma_bench import spec
from enigma_bench.checkpoint import CheckpointError, load_state_dict

#: Pinned so a shape change to any rung is a deliberate, visible edit.
EXPECTED_PARAMETERS = {"small": 3_237_914, "medium": 8_989_082, "large": 19_067_930}


def test_the_ladder_is_named_and_ordered() -> None:
    assert list(spec.SIZES) == ["small", "medium", "large"]
    widths = [size.d_model for size in spec.SIZES.values()]
    depths = [size.num_layers for size in spec.SIZES.values()]
    assert widths == sorted(widths) and depths == sorted(depths), "width and depth grow together"


@pytest.mark.parametrize("size", list(spec.SIZES.values()), ids=list(spec.SIZES))
def test_every_size_follows_the_usual_shape_conventions(size: spec.ModelSize) -> None:
    assert size.d_model % size.num_heads == 0
    assert size.d_model // size.num_heads == spec.HEAD_DIM == 64
    assert size.d_ff == 4 * size.d_model
    assert 64 <= size.aspect_ratio <= 128, "aspect ratio outside the band published models sit in"


@pytest.mark.parametrize("name", list(spec.SIZES), ids=list(spec.SIZES))
def test_parameter_counts_are_pinned(name: str) -> None:
    assert spec.parameter_count(spec.SIZES[name]) == EXPECTED_PARAMETERS[name]


def test_large_is_the_default_and_byte_identical_to_the_original_spec() -> None:
    """Every run logged before sizes existed was a `large`; keep it comparable."""
    assert spec.resolve_size(None) is spec.SIZES["large"]
    assert spec.DEFAULT_SIZE_NAME == "large"
    assert spec.architecture_fingerprint(spec.SIZES["large"]) == "eb4b9e50195ca17e"


def test_sizes_are_told_apart_by_their_fingerprints() -> None:
    fingerprints = {spec.architecture_fingerprint(size) for size in spec.SIZES.values()}
    assert len(fingerprints) == len(spec.SIZES)


def test_the_task_itself_is_identical_at_every_size() -> None:
    for size in spec.SIZES.values():
        architecture = spec.architecture(size)
        assert architecture["seq_len"] == spec.SEQ_LEN
        assert architecture["vocab_size"] == spec.VOCAB_SIZE
        assert architecture["num_classes"] == spec.NUM_CLASSES


def test_resolve_size_is_forgiving_about_spelling_but_not_about_names() -> None:
    assert spec.resolve_size("  Medium ") is spec.SIZES["medium"]
    assert spec.resolve_size("") is spec.SIZES["large"]
    assert spec.resolve_size("   ") is spec.SIZES["large"]
    with pytest.raises(ValueError, match="unknown model size 'tiny'"):
        spec.resolve_size("tiny")


def test_the_process_binds_one_size_from_the_environment() -> None:
    assert spec.SIZE is spec.resolve_size(spec.SIZE_NAME)
    assert spec.SIZE.d_model == spec.D_MODEL
    assert spec.SIZE.num_layers == spec.NUM_LAYERS
    assert spec.SIZE.num_heads == spec.NUM_HEADS
    assert spec.SIZE.d_ff == spec.D_FF


def test_a_checkpoint_from_another_size_is_refused_by_name(tmp_path: Path) -> None:
    """The failure an operator is most likely to hit: weights from the wrong run."""
    other = next(size for size in spec.SIZES.values() if size.name != spec.SIZE_NAME)
    path = tmp_path / "other-size.pt"
    torch.save(
        {"state_dict": {n: torch.zeros(s) for n, s in spec.parameter_shapes(other).items()}}, path
    )

    with pytest.raises(CheckpointError) as raised:
        load_state_dict(path)
    message = str(raised.value)
    assert f"complete '{other.name}' model" in message
    assert f"this run is '{spec.SIZE_NAME}'" in message
    assert "wrong shapes" not in message, "the size mismatch is the whole story; say only that"


SMALL_MODEL_CHECK = textwrap.dedent(
    """
    import torch
    from enigma_bench import spec
    from enigma_bench.model import EnigmaEncoder, assert_matches_spec, parameter_count

    assert spec.SIZE_NAME == "small", spec.SIZE_NAME
    model = EnigmaEncoder().eval()
    assert_matches_spec(model)
    assert parameter_count(model) == 3_237_914, parameter_count(model)
    with torch.inference_mode():
        logits = model(torch.zeros((2, spec.SEQ_LEN), dtype=torch.long))
    assert logits.shape == (2, spec.SEQ_LEN, spec.NUM_CLASSES), logits.shape
    print("ok")
    """
)


def test_a_smaller_model_really_builds_under_its_own_environment() -> None:
    """``model`` binds its shapes at import, so a second size needs its own interpreter."""
    completed = subprocess.run(  # noqa: S603 - a fixed script under this interpreter
        [sys.executable, "-c", SMALL_MODEL_CHECK],
        env={**os.environ, spec.SIZE_ENV: "small"},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# The operator's entry point: ./bench.sh run --size ...
# --------------------------------------------------------------------------- #

BENCH_DIR = Path(__file__).parents[1]
DOCKER_STUB = """#!/usr/bin/env bash
printf '%s' "${ENIGMA_BENCH_SIZE-}" > "$CAPTURE_SIZE"
"""


def _bench_run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``bench.sh run`` against a docker stub that records the size it was handed."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    stub = fake_bin / "docker"
    stub.write_text(DOCKER_STUB)
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_SIZE": str(tmp_path / "size"),
        "ENIGMA_BENCH_OUT": str(tmp_path / "out"),
    }
    env.pop(spec.SIZE_ENV, None)  # the default under test is bench.sh's, not the caller's
    return subprocess.run(  # noqa: S603 - the repo's own script, with a stubbed docker
        [str(BENCH_DIR / "bench.sh"), "run", *args],
        cwd=BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bench_run_hands_the_chosen_size_to_the_container(tmp_path: Path) -> None:
    result = _bench_run(tmp_path, "--size", "medium")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "size").read_text() == "medium"


def test_bench_run_defaults_to_large(tmp_path: Path) -> None:
    result = _bench_run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "size").read_text() == "large"


def test_bench_run_refuses_a_size_that_does_not_exist(tmp_path: Path) -> None:
    result = _bench_run(tmp_path, "--size", "enormous")
    assert result.returncode != 0
    assert "unknown --size 'enormous'" in result.stderr
    assert not (tmp_path / "size").exists(), "no container should have started"
