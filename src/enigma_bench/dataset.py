"""Turning a corpus plus a tier into scored examples.

Kept free of torch so the eval set can be built, inspected and unit-tested
without a GPU stack. Training code is expected to wrap :func:`build_example`
in whatever ``Dataset``/``DataLoader`` it likes — see ``task/starter/train.py``
for a worked example.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass

from .corpus import Corpus
from .machine import EnigmaMachine, EnigmaSettings, indices_to_letters
from .spec import SPEC_VERSION, TEXT_LEN
from .tiers import Tier
from .tokenizer import RevealMask, encode_input, header_labels, plaintext_labels


@dataclass(frozen=True, slots=True)
class Example:
    """One (input, target) pair plus the ground truth that produced it."""

    tier_id: str
    settings: EnigmaSettings
    reveal: RevealMask
    plaintext: tuple[int, ...]
    ciphertext: tuple[int, ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    aux_labels: tuple[int, ...]

    def plaintext_str(self) -> str:
        """The plaintext as uppercase letters."""
        return indices_to_letters(self.plaintext)

    def ciphertext_str(self) -> str:
        """The ciphertext as uppercase letters."""
        return indices_to_letters(self.ciphertext)


def build_example(
    tier: Tier, corpus: Corpus, rng: random.Random, *, text_len: int = TEXT_LEN
) -> Example:
    """Draw a plaintext window, a machine setup and a reveal mask, and encipher."""
    plaintext = corpus.random_window(rng, text_len)
    settings = tier.sample_settings(rng)
    reveal = tier.sample_reveal(settings, rng)
    ciphertext = EnigmaMachine(settings).encrypt_indices(plaintext)
    return Example(
        tier_id=tier.id,
        settings=settings,
        reveal=reveal,
        plaintext=tuple(plaintext),
        ciphertext=tuple(ciphertext),
        input_ids=tuple(encode_input(settings, reveal, ciphertext)),
        labels=tuple(plaintext_labels(plaintext)),
        aux_labels=tuple(header_labels(settings)),
    )


def example_seed(tier_id: str, base_seed: int, index: int) -> int:
    """A stable per-example seed.

    Derived from a hash rather than ``base_seed + index`` so that example *k* of
    a tier is identical no matter how many examples the caller asks for, and so
    that neighbouring tiers do not share correlated streams.
    """
    payload = f"enigma-bench/v{SPEC_VERSION}/{tier_id}/{base_seed}/{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_eval_set(
    tier: Tier, corpus: Corpus, count: int, base_seed: int, *, text_len: int = TEXT_LEN
) -> list[Example]:
    """The deterministic evaluation set for one tier.

    The same ``(tier, corpus, base_seed)`` always yields the same examples in the
    same order, and shrinking ``count`` yields a prefix of the larger set.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return [
        build_example(
            tier, corpus, random.Random(example_seed(tier.id, base_seed, index)), text_len=text_len
        )
        for index in range(count)
    ]


def iter_training_examples(
    tiers: tuple[Tier, ...],
    corpus: Corpus,
    rng: random.Random,
    *,
    weights: tuple[float, ...] | None = None,
    text_len: int = TEXT_LEN,
) -> Iterator[Example]:
    """An endless stream of examples, mixing tiers by ``weights``.

    Defaults to the tiers' own weights. Used by the starter trainer; a curriculum
    can simply change ``weights`` over time.
    """
    if not tiers:
        raise ValueError("need at least one tier")
    tier_weights = list(weights) if weights is not None else [tier.weight for tier in tiers]
    while True:
        tier = rng.choices(tiers, weights=tier_weights, k=1)[0]
        yield build_example(tier, corpus, rng, text_len=text_len)
