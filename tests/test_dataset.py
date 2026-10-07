"""Example construction and the determinism the eval relies on."""

from __future__ import annotations

import random

import pytest

from enigma_bench.corpus import Corpus
from enigma_bench.dataset import build_eval_set, build_example, example_seed, iter_training_examples
from enigma_bench.machine import EnigmaMachine
from enigma_bench.spec import HEADER_LEN, SEQ_LEN, TEXT_LEN
from enigma_bench.tiers import TIERS, TIERS_BY_ID
from enigma_bench.tokenizer import encode_input


@pytest.mark.parametrize("tier", TIERS, ids=[tier.id for tier in TIERS])
def test_example_ciphertext_really_is_the_plaintext_enciphered(tier, corpus: Corpus) -> None:
    example = build_example(tier, corpus, random.Random(5))
    assert len(example.plaintext) == TEXT_LEN
    assert len(example.input_ids) == SEQ_LEN
    expected = EnigmaMachine(example.settings).encrypt_indices(example.plaintext)
    assert list(example.ciphertext) == expected
    assert list(example.input_ids) == encode_input(
        example.settings, example.reveal, list(example.ciphertext)
    )


def test_eval_sets_are_deterministic_and_prefix_stable(corpus: Corpus) -> None:
    tier = TIERS_BY_ID["t5_wheel_order"]
    small = build_eval_set(tier, corpus, 4, base_seed=42)
    large = build_eval_set(tier, corpus, 12, base_seed=42)
    assert [e.input_ids for e in small] == [e.input_ids for e in large[:4]]
    assert build_eval_set(tier, corpus, 4, base_seed=42) == small


def test_a_different_seed_gives_different_examples(corpus: Corpus) -> None:
    tier = TIERS[0]
    a = build_eval_set(tier, corpus, 4, base_seed=1)
    b = build_eval_set(tier, corpus, 4, base_seed=2)
    assert [e.input_ids for e in a] != [e.input_ids for e in b]


def test_example_seeds_are_distinct_across_tiers_and_indices() -> None:
    seeds = {example_seed(tier.id, 0, index) for tier in TIERS for index in range(20)}
    assert len(seeds) == len(TIERS) * 20


def test_labels_line_up_with_the_plaintext(corpus: Corpus) -> None:
    example = build_example(TIERS[0], corpus, random.Random(0))
    assert list(example.labels[HEADER_LEN:]) == list(example.plaintext)
    assert example.plaintext_str().isupper()
    assert len(example.ciphertext_str()) == TEXT_LEN


def test_build_eval_set_rejects_a_non_positive_count(corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        build_eval_set(TIERS[0], corpus, 0, base_seed=1)


def test_training_stream_mixes_tiers(corpus: Corpus) -> None:
    stream = iter_training_examples(TIERS, corpus, random.Random(9))
    seen = {next(stream).tier_id for _ in range(200)}
    assert len(seen) > 1
    with pytest.raises(ValueError, match="at least one tier"):
        next(iter_training_examples((), corpus, random.Random(0)))
