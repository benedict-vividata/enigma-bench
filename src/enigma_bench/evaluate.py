"""The scorer.

Builds a deterministic evaluation set per tier, runs the model over it, and
reports letter accuracy, exact-match rate and loss. The headline
:attr:`EvaluationResult.score` is the weighted mean letter accuracy across the
ladder, rescaled so that the **naive baseline** — always guessing the corpus's
most common letter, roughly 12% on English — scores 0 and a perfect model
scores 100. :attr:`EvaluationResult.raw_score` keeps the unrescaled weighted
mean accuracy x100. :attr:`EvaluationResult.grade` is the highest tier cleared
at :data:`~enigma_bench.tiers.GRADE_THRESHOLD` (raw letter accuracy), which is
the number worth quoting when comparing agents.

Chance is 1/26 = 3.85%. A score at or below 0 means the model has learned
nothing beyond the plaintext's letter statistics.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoint import load_model
from .corpus import Corpus
from .dataset import Example, build_eval_set
from .machine import indices_to_letters
from .model import cross_entropy
from .spec import (
    HEADER_LEN,
    IGNORE_INDEX,
    SIZE_NAME,
    SPEC_VERSION,
    TEXT_LEN,
    architecture_fingerprint,
)
from .tiers import CHANCE_ACCURACY, GRADE_THRESHOLD, TIERS, Tier, grade

DEFAULT_SAMPLES_PER_TIER = 256


def baseline_accuracy(corpus: Corpus) -> float:
    """Letter accuracy of the naive baseline: always guess the corpus's most common letter."""
    counts = Counter(corpus.letters)
    return max(counts.values()) / len(corpus)


DEFAULT_SEED = 20260820
DEFAULT_BATCH_SIZE = 64
POSITION_BUCKET = 20
PREVIEW_LETTERS = 64


@dataclass(frozen=True, slots=True)
class TierResult:
    """Scores for one rung of the ladder."""

    tier_id: str
    name: str
    samples: int
    letter_accuracy: float
    exact_match: float
    mean_loss: float
    accuracy_by_position: list[float]
    mean_hidden_slots: float
    preview: dict[str, str] = field(default_factory=dict)

    def cleared(self, threshold: float = GRADE_THRESHOLD) -> bool:
        """Whether this tier counts as passed."""
        return self.letter_accuracy >= threshold


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The full outcome of one ``evaluate_checkpoint`` call."""

    score: float
    raw_score: float
    baseline_accuracy: float
    grade: int
    grade_label: str
    per_tier: list[TierResult]
    samples_per_tier: int
    seed: int
    device: str
    batch_size: int
    elapsed_seconds: float
    spec_version: int
    model_size: str
    architecture_fingerprint: str
    chance_accuracy: float
    corpus: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable view, used by the CLI and the run log."""
        payload = asdict(self)
        payload["per_tier"] = [asdict(result) for result in self.per_tier]
        return payload

    def accuracy_by_tier(self) -> dict[str, float]:
        """Convenience map from tier id to letter accuracy."""
        return {result.tier_id: result.letter_accuracy for result in self.per_tier}

    def format_table(self) -> str:
        """A fixed-width summary suitable for a terminal or a tool response."""
        lines = [
            f"{'tier':<20} {'samples':>7} {'letter acc':>11} {'exact':>8} {'loss':>7}  status",
            "-" * 68,
        ]
        for result in self.per_tier:
            status = "PASS" if result.cleared() else "    "
            lines.append(
                f"{result.tier_id:<20} {result.samples:>7} {result.letter_accuracy:>10.2%} "
                f"{result.exact_match:>7.2%} {result.mean_loss:>7.3f}  {status}"
            )
        lines.append("-" * 68)
        lines.append(
            f"score {self.score:.2f}/100 vs the {self.baseline_accuracy:.2%} "
            f"most-common-letter baseline (raw accuracy {self.raw_score:.2f})   "
            f"grade {self.grade_label}"
        )
        return "\n".join(lines)


def _batch_tensors(examples: list[Example], device: str) -> tuple[Tensor, Tensor]:
    input_ids = torch.tensor([example.input_ids for example in examples], dtype=torch.long)
    labels = torch.tensor([example.labels for example in examples], dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def _reveal_summary(example: Example) -> str:
    """Compact ``RWGP plug=<26 bits>`` description of what the header showed."""
    reveal = example.reveal
    if isinstance(reveal.ground_setting, bool):
        ground = "P" if reveal.ground_setting else "-"
    else:
        ground = "".join("P" if flag else "-" for flag in reveal.ground_setting_flags())
    shown = (
        "".join(
            letter if flag else "-"
            for letter, flag in (
                ("R", reveal.reflector),
                ("W", reveal.rotors),
                ("G", reveal.ring_settings),
            )
        )
        + ground
    )
    return f"{shown} plug=" + "".join("1" if flag else "0" for flag in reveal.plugboard)


def _preview(example: Example, predicted: Tensor) -> dict[str, str]:
    """A short side-by-side of truth and prediction, for eyeballing a run."""
    letters = predicted[HEADER_LEN : HEADER_LEN + PREVIEW_LETTERS].tolist()
    return {
        "settings": str(example.settings),
        "reveal": _reveal_summary(example),
        "ciphertext": example.ciphertext_str()[:PREVIEW_LETTERS],
        "plaintext": example.plaintext_str()[:PREVIEW_LETTERS],
        "predicted": indices_to_letters(letters),
    }


@torch.inference_mode()
def evaluate_tier(
    model: nn.Module,
    tier: Tier,
    corpus: Corpus,
    *,
    samples: int = DEFAULT_SAMPLES_PER_TIER,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_preview: bool = True,
) -> TierResult:
    """Score ``model`` on one tier's deterministic evaluation set."""
    examples = build_eval_set(tier, corpus, samples, seed)

    correct = 0
    counted = 0
    exact = 0
    loss_total = 0.0
    loss_batches = 0
    bucket_correct = [0] * ((TEXT_LEN + POSITION_BUCKET - 1) // POSITION_BUCKET)
    bucket_total = [0] * len(bucket_correct)
    preview: dict[str, str] = {}

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        input_ids, labels = _batch_tensors(chunk, device)
        logits = model(input_ids)
        loss_total += float(cross_entropy(logits, labels))
        loss_batches += 1

        predicted = logits.argmax(dim=-1)
        scored = labels != IGNORE_INDEX
        hits = (predicted == labels) & scored
        correct += int(hits.sum())
        counted += int(scored.sum())
        exact += int((hits.sum(dim=1) == scored.sum(dim=1)).sum())

        text_hits = hits[:, HEADER_LEN:]
        text_scored = scored[:, HEADER_LEN:]
        for index in range(len(bucket_correct)):
            span = slice(index * POSITION_BUCKET, (index + 1) * POSITION_BUCKET)
            bucket_correct[index] += int(text_hits[:, span].sum())
            bucket_total[index] += int(text_scored[:, span].sum())

        if include_preview and not preview:
            preview = _preview(chunk[0], predicted[0].cpu())

    return TierResult(
        tier_id=tier.id,
        name=tier.name,
        samples=len(examples),
        letter_accuracy=correct / counted if counted else 0.0,
        exact_match=exact / len(examples),
        mean_loss=loss_total / loss_batches if loss_batches else float("nan"),
        accuracy_by_position=[
            (hit / total if total else 0.0)
            for hit, total in zip(bucket_correct, bucket_total, strict=True)
        ],
        mean_hidden_slots=sum(e.reveal.hidden_field_count() for e in examples) / len(examples),
        preview=preview,
    )


def evaluate_model(
    model: nn.Module,
    corpus: Corpus,
    *,
    tiers: tuple[Tier, ...] = TIERS,
    samples_per_tier: int = DEFAULT_SAMPLES_PER_TIER,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_preview: bool = True,
) -> EvaluationResult:
    """Score ``model`` across ``tiers`` and assemble the headline numbers."""
    started = time.monotonic()
    model.eval()
    results = [
        evaluate_tier(
            model,
            tier,
            corpus,
            samples=samples_per_tier,
            seed=seed,
            device=device,
            batch_size=batch_size,
            include_preview=include_preview,
        )
        for tier in tiers
    ]

    total_weight = sum(tier.weight for tier in tiers)
    weighted = sum(
        tier.weight * result.letter_accuracy for tier, result in zip(tiers, results, strict=True)
    )
    achieved = grade(
        {result.tier_id: result.letter_accuracy for result in results}, GRADE_THRESHOLD
    )
    baseline = baseline_accuracy(corpus)
    mean_accuracy = weighted / total_weight if total_weight else 0.0
    # A one-letter corpus would put the baseline at 1.0; guard the rescale.
    spread = 1.0 - baseline
    return EvaluationResult(
        score=round(100.0 * (mean_accuracy - baseline) / spread, 4) if spread > 0.0 else 0.0,
        raw_score=round(100.0 * mean_accuracy, 4),
        baseline_accuracy=round(baseline, 6),
        grade=achieved,
        grade_label=TIERS[achieved].id if achieved >= 0 else "none",
        per_tier=results,
        samples_per_tier=samples_per_tier,
        seed=seed,
        device=device,
        batch_size=batch_size,
        elapsed_seconds=round(time.monotonic() - started, 3),
        spec_version=SPEC_VERSION,
        model_size=SIZE_NAME,
        architecture_fingerprint=architecture_fingerprint(),
        chance_accuracy=CHANCE_ACCURACY,
        corpus={"name": corpus.name, "letters": len(corpus), "sha256": corpus.sha256},
    )


def evaluate_checkpoint_file(
    path: Path,
    corpus: Corpus,
    *,
    tiers: tuple[Tier, ...] = TIERS,
    samples_per_tier: int = DEFAULT_SAMPLES_PER_TIER,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_preview: bool = True,
) -> EvaluationResult:
    """Load a checkpoint into the fixed architecture and score it.

    Raises:
        CheckpointError: if the file does not match the architecture.
    """
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(path, device=resolved)
    return evaluate_model(
        model,
        corpus,
        tiers=tiers,
        samples_per_tier=samples_per_tier,
        seed=seed,
        device=resolved,
        batch_size=batch_size,
        include_preview=include_preview,
    )
