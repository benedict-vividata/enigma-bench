"""Metric plumbing, checked against models whose answers we know exactly."""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch import Tensor, nn

from enigma_bench.corpus import Corpus
from enigma_bench.dataset import build_eval_set
from enigma_bench.evaluate import (
    DEFAULT_SEED,
    baseline_accuracy,
    evaluate_model,
    evaluate_tier,
)
from enigma_bench.spec import HEADER_LEN, NUM_CLASSES, SEQ_LEN
from enigma_bench.tiers import TIERS, TIERS_BY_ID

TIER = TIERS_BY_ID["t0_full"]
SAMPLES = 12


class ConstantModel(nn.Module):
    """Always predicts the same letter, whatever the input."""

    def __init__(self, letter: int) -> None:
        super().__init__()
        self.letter = letter

    def forward(self, input_ids: Tensor) -> Tensor:
        logits = torch.zeros(input_ids.shape[0], SEQ_LEN, NUM_CLASSES)
        logits[..., self.letter] = 10.0
        return logits


class OracleModel(nn.Module):
    """Looks the answer up by input sequence — a stand-in for a perfect model."""

    def __init__(self, examples: list) -> None:
        super().__init__()
        self.table = {example.input_ids: example.plaintext for example in examples}

    def forward(self, input_ids: Tensor) -> Tensor:
        logits = torch.zeros(input_ids.shape[0], SEQ_LEN, NUM_CLASSES)
        for row, tokens in enumerate(input_ids.tolist()):
            for offset, letter in enumerate(self.table[tuple(tokens)]):
                logits[row, HEADER_LEN + offset, letter] = 10.0
        return logits


def test_a_perfect_model_scores_a_hundred(corpus: Corpus) -> None:
    examples = build_eval_set(TIER, corpus, SAMPLES, DEFAULT_SEED)
    result = evaluate_tier(OracleModel(examples), TIER, corpus, samples=SAMPLES, batch_size=5)
    assert result.letter_accuracy == 1.0
    assert result.exact_match == 1.0
    assert all(value == 1.0 for value in result.accuracy_by_position)
    assert result.preview["plaintext"] == result.preview["predicted"]


def test_a_constant_model_scores_the_letter_frequency(corpus: Corpus) -> None:
    examples = build_eval_set(TIER, corpus, SAMPLES, DEFAULT_SEED)
    counts: Counter[int] = Counter()
    for example in examples:
        counts.update(example.plaintext)
    letter = counts.most_common(1)[0][0]
    expected = counts[letter] / sum(counts.values())

    result = evaluate_tier(ConstantModel(letter), TIER, corpus, samples=SAMPLES, batch_size=4)
    assert result.letter_accuracy == expected
    assert result.exact_match == 0.0
    assert result.mean_hidden_slots == 0.0, "t0 reveals the whole key"


def test_the_naive_baseline_scores_zero(corpus: Corpus) -> None:
    """Always guessing the corpus's most common letter earns a score of ~0."""
    counts: Counter[int] = Counter(corpus.letters)
    letter = counts.most_common(1)[0][0]
    baseline = baseline_accuracy(corpus)

    per_tier = []
    for tier in TIERS:
        tier_counts: Counter[int] = Counter()
        for example in build_eval_set(tier, corpus, SAMPLES, DEFAULT_SEED):
            tier_counts.update(example.plaintext)
        per_tier.append(tier_counts[letter] / sum(tier_counts.values()))
    expected = 100.0 * (sum(per_tier) / len(per_tier) - baseline) / (1.0 - baseline)

    result = evaluate_model(ConstantModel(letter), corpus, samples_per_tier=SAMPLES, batch_size=6)
    assert result.baseline_accuracy == pytest.approx(baseline, abs=1e-5)
    assert result.score == pytest.approx(expected, abs=1e-3)
    assert abs(result.score) < 15.0, "letter statistics alone must land near zero"


def test_batching_does_not_change_the_score(corpus: Corpus) -> None:
    model = ConstantModel(4)
    scores = {
        batch_size: evaluate_tier(
            model, TIER, corpus, samples=SAMPLES, batch_size=batch_size
        ).letter_accuracy
        for batch_size in (1, 5, SAMPLES, SAMPLES * 2)
    }
    assert len(set(scores.values())) == 1


def test_full_ladder_aggregates_and_grades(corpus: Corpus) -> None:
    examples = {tier.id: build_eval_set(tier, corpus, SAMPLES, DEFAULT_SEED) for tier in TIERS}
    model = OracleModel([example for group in examples.values() for example in group])
    result = evaluate_model(model, corpus, samples_per_tier=SAMPLES, batch_size=6)

    assert result.score == 100.0
    assert result.raw_score == 100.0
    assert result.grade == len(TIERS) - 1
    assert result.grade_label == TIERS[-1].id
    assert [tier.tier_id for tier in result.per_tier] == [tier.id for tier in TIERS]
    assert result.corpus["sha256"] == corpus.sha256
    assert "score 100.00/100" in result.format_table()
    assert result.to_dict()["per_tier"][0]["tier_id"] == TIERS[0].id


def test_hidden_slot_counts_grow_up_the_ladder(corpus: Corpus) -> None:
    model = ConstantModel(0)
    hidden = [
        evaluate_tier(model, tier, corpus, samples=SAMPLES, batch_size=6).mean_hidden_slots
        for tier in TIERS
    ]
    assert hidden == sorted(hidden)
    assert hidden[-1] == 36.0, "the blind tier masks every header slot"
