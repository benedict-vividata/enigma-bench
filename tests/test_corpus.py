"""Corpus cleaning, windowing and split loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from enigma_bench.corpus import Corpus, CorpusError, clean, load_manifest, load_split, split_path


def test_clean_keeps_only_letters_as_indices() -> None:
    assert list(clean("Ab, z!\n9")) == [0, 1, 25]


def test_windows_wrap_around_the_end_of_the_stream() -> None:
    corpus = Corpus.from_text("abcde")
    assert corpus.window(1, 3) == [1, 2, 3]
    assert corpus.window(3, 4) == [3, 4, 0, 1]
    assert corpus.window(8, 2) == corpus.window(3, 2)
    assert len(corpus) == 5


def test_window_validation() -> None:
    corpus = Corpus.from_text("abcde")
    with pytest.raises(ValueError, match="must be positive"):
        corpus.window(0, 0)
    with pytest.raises(CorpusError, match="fewer than the requested window"):
        corpus.window(0, 99)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(CorpusError, match="no a-z characters"):
        Corpus.from_text("1234 !!")


def test_content_hash_tracks_content() -> None:
    assert Corpus.from_text("hello").sha256 == Corpus.from_text("H E L L O!").sha256
    assert Corpus.from_text("hello").sha256 != Corpus.from_text("hellp").sha256


def test_load_split_reads_from_the_bench_home(bench_home: Path) -> None:
    corpus = load_split("test")
    assert len(corpus) > 1000
    assert split_path("test").parent == bench_home / "corpus"
    with pytest.raises(ValueError, match="unknown split"):
        split_path("holdout")


def test_missing_corpus_points_at_the_fetch_script(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=r"fetch_corpus\.py"):
        load_split("train", tmp_path)


def test_manifest_is_optional(bench_home: Path) -> None:
    assert load_manifest() == {}
    (bench_home / "corpus" / "manifest.json").write_text('{"source": "unit-test"}')
    assert load_manifest()["source"] == "unit-test"
