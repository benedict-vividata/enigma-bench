"""Turning a machine setup plus a ciphertext into the model's 256 token ids.

The layout is documented in :mod:`enigma_bench.spec`. This module is the only
place that knows how a header slot maps to a vocabulary id, so both the eval
harness and any training script should go through it rather than hand-rolling
the encoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from .machine import ALPHABET_SIZE, ROTOR_NAMES, EnigmaSettings
from .spec import (
    GROUND_SLOTS,
    HEADER_LEN,
    IGNORE_INDEX,
    LETTER_OFFSET,
    MASK_ID,
    PAD_ID,
    PLUGBOARD_SLOTS,
    REFLECTOR_OFFSET,
    REFLECTOR_SLOT,
    RING_SLOTS,
    ROTOR_OFFSET,
    ROTOR_SLOTS,
    SEQ_LEN,
    TEXT_LEN,
    TOKEN_NAMES,
    VOCAB_SIZE,
)

_ROTOR_TO_ID: Final[dict[str, int]] = {name: ROTOR_OFFSET + i for i, name in enumerate(ROTOR_NAMES)}
_REFLECTOR_TO_ID: Final[dict[str, int]] = {"B": REFLECTOR_OFFSET, "C": REFLECTOR_OFFSET + 1}

type GroundSettingReveal = bool | tuple[bool, bool, bool]


def letter_id(index: int) -> int:
    """Vocabulary id for letter index ``0..25``."""
    return LETTER_OFFSET + index


def rotor_id(name: str) -> int:
    """Vocabulary id for a rotor name such as ``"III"``."""
    return _ROTOR_TO_ID[name]


def reflector_id(name: str) -> int:
    """Vocabulary id for reflector ``"B"`` or ``"C"``."""
    return _REFLECTOR_TO_ID[name]


@dataclass(frozen=True, slots=True)
class RevealMask:
    """Which parts of the machine setup the model gets to see in the header.

    ``plugboard`` holds one flag per letter ``a..z``: ``True`` means slot
    ``10 + i`` shows the letter that ``i`` is steckered to (itself when
    unplugged), ``False`` means the slot is ``<mask>``.
    """

    reflector: bool = True
    rotors: bool = True
    ring_settings: bool = True
    ground_setting: GroundSettingReveal = True
    plugboard: tuple[bool, ...] = (True,) * ALPHABET_SIZE

    def __post_init__(self) -> None:
        if len(self.plugboard) != ALPHABET_SIZE:
            raise ValueError(
                f"plugboard reveal mask must have {ALPHABET_SIZE} flags, got {len(self.plugboard)}"
            )
        if isinstance(self.ground_setting, tuple) and (
            len(self.ground_setting) != 3
            or any(not isinstance(flag, bool) for flag in self.ground_setting)
        ):
            raise ValueError("ground setting reveal mask must have three boolean flags")

    def ground_setting_flags(self) -> tuple[bool, bool, bool]:
        """Return the reveal flag for each ground-setting slot."""
        if isinstance(self.ground_setting, bool):
            return (self.ground_setting,) * 3
        return self.ground_setting

    @classmethod
    def everything(cls) -> RevealMask:
        """Nothing hidden: the header fully describes the machine."""
        return cls()

    @classmethod
    def nothing(cls) -> RevealMask:
        """Every header field masked: the model sees ciphertext only."""
        return cls(
            reflector=False,
            rotors=False,
            ring_settings=False,
            ground_setting=False,
            plugboard=(False,) * ALPHABET_SIZE,
        )

    def with_plugboard(self, flags: Iterable[bool]) -> RevealMask:
        """Copy with a new per-letter plugboard reveal pattern."""
        return replace(self, plugboard=tuple(flags))

    def hidden_field_count(self) -> int:
        """Number of masked header slots — a crude difficulty proxy for logging."""
        hidden = 0
        hidden += 0 if self.reflector else 1
        hidden += 0 if self.rotors else 3
        hidden += 0 if self.ring_settings else 3
        hidden += sum(not flag for flag in self.ground_setting_flags())
        hidden += sum(1 for flag in self.plugboard if not flag)
        return hidden

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly summary, with the plugboard collapsed to a bit string."""
        return {
            "reflector": self.reflector,
            "rotors": self.rotors,
            "ring_settings": self.ring_settings,
            "ground_setting": (
                self.ground_setting
                if isinstance(self.ground_setting, bool)
                else self.ground_setting_flags()
            ),
            "plugboard": "".join("1" if flag else "0" for flag in self.plugboard),
            "hidden_slots": self.hidden_field_count(),
        }


def encode_header(settings: EnigmaSettings, reveal: RevealMask) -> list[int]:
    """Build the 36 header token ids for ``settings`` under ``reveal``."""
    header = [MASK_ID] * HEADER_LEN

    if reveal.reflector:
        header[REFLECTOR_SLOT] = reflector_id(settings.reflector)
    if reveal.rotors:
        header[ROTOR_SLOTS] = [rotor_id(name) for name in settings.rotors]
    if reveal.ring_settings:
        header[RING_SLOTS] = [letter_id(value) for value in settings.ring_settings]
    header[GROUND_SLOTS] = [
        letter_id(value) if show else MASK_ID
        for value, show in zip(settings.ground_setting, reveal.ground_setting_flags(), strict=True)
    ]

    permutation = settings.plugboard_permutation()
    header[PLUGBOARD_SLOTS] = [
        letter_id(permutation[i]) if reveal.plugboard[i] else MASK_ID for i in range(ALPHABET_SIZE)
    ]
    return header


def encode_text(letters: Sequence[int]) -> list[int]:
    """Encode up to :data:`TEXT_LEN` letter indices, right-padded with ``<pad>``."""
    if len(letters) > TEXT_LEN:
        raise ValueError(f"text of {len(letters)} letters exceeds TEXT_LEN={TEXT_LEN}")
    body = [letter_id(index) for index in letters]
    return body + [PAD_ID] * (TEXT_LEN - len(body))


def encode_input(
    settings: EnigmaSettings, reveal: RevealMask, ciphertext: Sequence[int]
) -> list[int]:
    """The full 256-token model input for one example."""
    return encode_header(settings, reveal) + encode_text(ciphertext)


def plaintext_labels(plaintext: Sequence[int]) -> list[int]:
    """Scored targets: the plaintext letter at each text slot, ignored elsewhere."""
    if len(plaintext) > TEXT_LEN:
        raise ValueError(f"plaintext of {len(plaintext)} letters exceeds TEXT_LEN={TEXT_LEN}")
    labels = [IGNORE_INDEX] * SEQ_LEN
    for offset, letter in enumerate(plaintext):
        labels[HEADER_LEN + offset] = letter
    return labels


def header_labels(settings: EnigmaSettings) -> list[int]:
    """Unscored auxiliary targets: the letter-valued header slots.

    Slots 4..35 (ring settings, ground setting, plugboard) are letters, so a
    training script can ask the model to recover them from the ciphertext even
    when they are masked in the input. The reflector and rotor-order slots are
    not letters and stay :data:`IGNORE_INDEX`.

    The benchmark never scores these positions; they exist purely as an
    optional auxiliary loss.
    """
    labels = [IGNORE_INDEX] * SEQ_LEN
    labels[RING_SLOTS] = list(settings.ring_settings)
    labels[GROUND_SLOTS] = list(settings.ground_setting)
    labels[PLUGBOARD_SLOTS] = list(settings.plugboard_permutation())
    return labels


def decode_tokens(token_ids: Iterable[int]) -> str:
    """Render token ids as readable names — for debugging and error messages."""
    parts = []
    for token in token_ids:
        parts.append(TOKEN_NAMES[token] if 0 <= token < VOCAB_SIZE else f"<bad:{token}>")
    return " ".join(parts)


def describe_input(token_ids: Sequence[int]) -> str:
    """A one-block human summary of an encoded example."""
    if len(token_ids) != SEQ_LEN:
        raise ValueError(f"expected {SEQ_LEN} tokens, got {len(token_ids)}")
    body = "".join(
        TOKEN_NAMES[token] if token != PAD_ID else "." for token in token_ids[HEADER_LEN:]
    )
    return (
        f"reflector : {decode_tokens(token_ids[REFLECTOR_SLOT : REFLECTOR_SLOT + 1])}\n"
        f"rotors    : {decode_tokens(token_ids[ROTOR_SLOTS])}\n"
        f"rings     : {decode_tokens(token_ids[RING_SLOTS])}\n"
        f"ground    : {decode_tokens(token_ids[GROUND_SLOTS])}\n"
        f"plugboard : {decode_tokens(token_ids[PLUGBOARD_SLOTS])}\n"
        f"ciphertext: {body}"
    )
