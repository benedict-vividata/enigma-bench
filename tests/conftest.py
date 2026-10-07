"""Shared fixtures: a tiny corpus, an isolated benchmark home, and a checkpoint."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from enigma_bench.corpus import Corpus

_VOCABULARY = (
    "the quick brown fox jumps over the lazy dog enigma machine cipher rotor plugboard "
    "bletchley park bombe reflector ringstellung grundstellung wehrmacht kriegsmarine "
    "message attack at dawn weather report nothing to report heil"
)
_WORDS = _VOCABULARY.split()


def _filler(word_count: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(word_count))


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    """A small in-memory corpus, big enough to draw non-overlapping windows from."""
    return Corpus.from_text(_filler(20_000, seed=11), name="test-corpus")


@pytest.fixture
def bench_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An isolated ENIGMA_BENCH_HOME with all three corpus splits written out."""
    home = tmp_path / "bench"
    corpus_dir = home / "corpus"
    corpus_dir.mkdir(parents=True)
    for index, (split, words) in enumerate((("train", 8_000), ("valid", 2_000), ("test", 4_000))):
        (corpus_dir / f"{split}.txt").write_text(_filler(words, seed=100 + index))
    monkeypatch.setenv("ENIGMA_BENCH_HOME", str(home))
    monkeypatch.delenv("ENIGMA_BENCH_CORPUS", raising=False)
    monkeypatch.delenv("ENIGMA_BENCH_RUNLOG", raising=False)
    monkeypatch.delenv("ENIGMA_BENCH_SUBMISSIONS", raising=False)
    monkeypatch.delenv("ENIGMA_BENCH_DEADLINE", raising=False)
    monkeypatch.delenv("ENIGMA_BENCH_MAX_EVALUATIONS", raising=False)
    monkeypatch.delenv("ENIGMA_BENCH_EVAL_SEED", raising=False)
    yield home


@pytest.fixture
def checkpoint(bench_home: Path) -> Path:
    """A freshly initialised, spec-conformant checkpoint."""
    import torch

    from enigma_bench.checkpoint import save_checkpoint
    from enigma_bench.model import EnigmaEncoder

    torch.manual_seed(0)
    path = bench_home / "submissions" / "model.pt"
    return save_checkpoint(EnigmaEncoder(), path, metadata={"note": "fixture"})


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio only."""
    return "asyncio"
