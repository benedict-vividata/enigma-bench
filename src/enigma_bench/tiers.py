"""The difficulty ladder.

Each tier fixes (a) how much of the Enigma setup the model is shown and (b) how
much plugboard there is to contend with. Tiers get strictly harder: ``t0``
hands over the whole key and only asks the model to *be* the machine, ``t8``
hands over nothing but ciphertext against a full ten-pair plugboard.

Every tier draws a fresh random machine setup per example, so a model cannot
memorise one key — it has to learn the mechanism (and, from ``t1`` up, to infer
the hidden part of the key from the ciphertext statistics).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from .machine import ALPHABET, ALPHABET_SIZE, BENCHMARK_ROTORS, EnigmaSettings
from .tokenizer import RevealMask


@dataclass(frozen=True, slots=True)
class Tier:
    """One rung of the ladder.

    Attributes:
        id: Stable identifier used in run logs and CLI flags.
        name: Short human label.
        description: What is hidden and why the rung is harder than the last.
        plugboard_pairs: Number of *Stecker* pairs on the machine.
        show_reflector: Whether slot 0 shows the reflector.
        show_rotors: Whether slots 1..3 show the rotor order.
        show_ring_settings: Whether slots 4..6 show the *Ringstellung*.
        show_ground_setting: Whether all slots 7..9 show the *Grundstellung*.
        ground_mask_options: Ground-slot index combinations to mask. One option
            is sampled uniformly when non-empty; indices are left/middle/right.
        plugboard_reveal: Fraction of letters whose plugboard image is shown.
            Revealed letters are chosen pair-wise, so a partially revealed
            plugboard never leaks half a pair.
        weight: Contribution to the headline score.
    """

    id: str
    name: str
    description: str
    plugboard_pairs: int
    show_reflector: bool
    show_rotors: bool
    show_ring_settings: bool
    show_ground_setting: bool
    plugboard_reveal: float
    weight: float = 1.0
    ground_mask_options: tuple[tuple[int, ...], ...] = ()

    def sample_settings(self, rng: random.Random) -> EnigmaSettings:
        """Draw a uniformly random machine setup consistent with this tier."""
        rotors = tuple(rng.sample(BENCHMARK_ROTORS, 3))
        return EnigmaSettings(
            rotors=(rotors[0], rotors[1], rotors[2]),
            reflector=rng.choice(("B", "C")),
            ring_settings=(
                rng.randrange(ALPHABET_SIZE),
                rng.randrange(ALPHABET_SIZE),
                rng.randrange(ALPHABET_SIZE),
            ),
            ground_setting=(
                rng.randrange(ALPHABET_SIZE),
                rng.randrange(ALPHABET_SIZE),
                rng.randrange(ALPHABET_SIZE),
            ),
            plugboard=_sample_plugboard(self.plugboard_pairs, rng),
        )

    def sample_reveal(self, settings: EnigmaSettings, rng: random.Random) -> RevealMask:
        """Draw the header reveal mask for one example of this tier."""
        ground_setting: bool | tuple[bool, bool, bool] = self.show_ground_setting
        if self.ground_mask_options:
            masked = set(rng.choice(self.ground_mask_options))
            ground_setting = tuple(index not in masked for index in range(3))
        return RevealMask(
            reflector=self.show_reflector,
            rotors=self.show_rotors,
            ring_settings=self.show_ring_settings,
            ground_setting=ground_setting,
            plugboard=_sample_plugboard_reveal(settings, self.plugboard_reveal, rng),
        )


def _sample_plugboard(pairs: int, rng: random.Random) -> tuple[tuple[str, str], ...]:
    if pairs == 0:
        return ()
    letters = list(ALPHABET)
    rng.shuffle(letters)
    return tuple((letters[2 * i], letters[2 * i + 1]) for i in range(pairs))


def _sample_plugboard_reveal(
    settings: EnigmaSettings, fraction: float, rng: random.Random
) -> tuple[bool, ...]:
    """Reveal roughly ``fraction`` of the 26 plugboard slots, pair by pair."""
    if fraction >= 1.0:
        return (True,) * ALPHABET_SIZE
    if fraction <= 0.0:
        return (False,) * ALPHABET_SIZE

    plugged = {ord(letter) - 65 for pair in settings.plugboard for letter in pair}
    groups: list[tuple[int, ...]] = [
        (ord(first) - 65, ord(second) - 65) for first, second in settings.plugboard
    ]
    groups.extend((i,) for i in range(ALPHABET_SIZE) if i not in plugged)
    rng.shuffle(groups)

    target = round(fraction * ALPHABET_SIZE)
    flags = [False] * ALPHABET_SIZE
    revealed = 0
    for group in groups:
        if revealed >= target:
            break
        for index in group:
            flags[index] = True
        revealed += len(group)
    return tuple(flags)


_FULL_PLUGBOARD_PAIRS: Final = 10

TIERS: Final[tuple[Tier, ...]] = (
    Tier(
        id="t0_full",
        name="Full key",
        description=(
            "Every setting is in the header and there is no plugboard. The model only has to "
            "implement the rotor machine — including ring offsets and the stepping schedule."
        ),
        plugboard_pairs=0,
        show_reflector=True,
        show_rotors=True,
        show_ring_settings=True,
        show_ground_setting=True,
        plugboard_reveal=1.0,
    ),
    Tier(
        id="t1_ground_one",
        name="One ground letter hidden",
        description=(
            "Rotor order, reflector and ring settings are given; exactly one of the left, "
            "middle or right starting-window letters is masked, with the three choices "
            "sampled equally. 26 candidate positions per message."
        ),
        plugboard_pairs=0,
        show_reflector=True,
        show_rotors=True,
        show_ring_settings=True,
        show_ground_setting=True,
        plugboard_reveal=1.0,
        ground_mask_options=((0,), (1,), (2,)),
    ),
    Tier(
        id="t2_ground_two",
        name="Two ground letters hidden",
        description=(
            "Rotor order, reflector and ring settings are given; exactly two of the left, "
            "middle or right starting-window letters are masked, with the three pair choices "
            "sampled equally. 26^2 candidate positions per message."
        ),
        plugboard_pairs=0,
        show_reflector=True,
        show_rotors=True,
        show_ring_settings=True,
        show_ground_setting=True,
        plugboard_reveal=1.0,
        ground_mask_options=((0, 1), (0, 2), (1, 2)),
    ),
    Tier(
        id="t3_ground",
        name="Ground setting hidden",
        description=(
            "Rotor order, reflector and ring settings are given; the three starting window "
            "letters are masked. 17,576 candidate positions per message."
        ),
        plugboard_pairs=0,
        show_reflector=True,
        show_rotors=True,
        show_ring_settings=True,
        show_ground_setting=False,
        plugboard_reveal=1.0,
    ),
    Tier(
        id="t4_rings",
        name="Rings and ground hidden",
        description=(
            "Rotor order and reflector are given; both Ringstellung and Grundstellung are "
            "masked, so rotor offsets and turnover timing are unknown."
        ),
        plugboard_pairs=0,
        show_reflector=True,
        show_rotors=True,
        show_ring_settings=False,
        show_ground_setting=False,
        plugboard_reveal=1.0,
    ),
    Tier(
        id="t5_wheel_order",
        name="Wheel order hidden",
        description=(
            "The whole rotor key is masked: reflector, wheel order, rings and ground. "
            "60 wheel orders x 2 reflectors x 26^6 offsets, no plugboard."
        ),
        plugboard_pairs=0,
        show_reflector=False,
        show_rotors=False,
        show_ring_settings=False,
        show_ground_setting=False,
        plugboard_reveal=1.0,
    ),
    Tier(
        id="t6_plugboard_given",
        name="Plugboard given",
        description=(
            "As t5 plus a ten-pair plugboard, but the plugboard wiring is shown in the header. "
            "Isolates 'can the model apply a known Stecker layer' from 'can it find one'."
        ),
        plugboard_pairs=_FULL_PLUGBOARD_PAIRS,
        show_reflector=False,
        show_rotors=False,
        show_ring_settings=False,
        show_ground_setting=False,
        plugboard_reveal=1.0,
    ),
    Tier(
        id="t7_plugboard_half",
        name="Plugboard half given",
        description=(
            "As t6 but only about half the plugboard slots are shown, pair by pair. Partial "
            "credit rung between a known and an unknown Stecker."
        ),
        plugboard_pairs=_FULL_PLUGBOARD_PAIRS,
        show_reflector=False,
        show_rotors=False,
        show_ring_settings=False,
        show_ground_setting=False,
        plugboard_reveal=0.5,
    ),
    Tier(
        id="t8_blind",
        name="Blind",
        description=(
            "Ciphertext only, ten-pair plugboard, nothing in the header. The historical "
            "problem: roughly 1.6e20 keys, from 220 letters of ciphertext."
        ),
        plugboard_pairs=_FULL_PLUGBOARD_PAIRS,
        show_reflector=False,
        show_rotors=False,
        show_ring_settings=False,
        show_ground_setting=False,
        plugboard_reveal=0.0,
    ),
)

TIERS_BY_ID: Final[dict[str, Tier]] = {tier.id: tier for tier in TIERS}

#: A model is graded at the highest tier it clears at this letter accuracy.
GRADE_THRESHOLD: Final = 0.90

#: Letter accuracy of a uniform random guesser.
CHANCE_ACCURACY: Final = 1.0 / ALPHABET_SIZE


def resolve_tiers(ids: list[str] | None) -> tuple[Tier, ...]:
    """Look up tiers by id, defaulting to the full ladder.

    Raises:
        KeyError: if an id is not one of :data:`TIERS`.
    """
    if not ids:
        return TIERS
    unknown = [tier_id for tier_id in ids if tier_id not in TIERS_BY_ID]
    if unknown:
        raise KeyError(f"unknown tier id(s) {unknown}; known: {sorted(TIERS_BY_ID)}")
    return tuple(TIERS_BY_ID[tier_id] for tier_id in ids)


def grade(per_tier_accuracy: dict[str, float], threshold: float = GRADE_THRESHOLD) -> int:
    """Highest tier index cleared at ``threshold``, or ``-1`` if none.

    Tiers must be cleared in order: a fluke pass on ``t4`` after failing ``t2``
    does not count.
    """
    achieved = -1
    for index, tier in enumerate(TIERS):
        accuracy = per_tier_accuracy.get(tier.id)
        if accuracy is None or accuracy < threshold:
            break
        achieved = index
    return achieved
