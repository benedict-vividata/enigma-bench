"""A clean, dependency-free Enigma I / M3 implementation.

This is the reference model of the cipher the benchmark is built on. It is
deliberately self-contained (no numpy, no torch) so it can be read top to
bottom, copied into a training script, or reimplemented in a vectorised form
by an agent that needs more throughput.

Signal path for one keypress::

    key -> plugboard -> right -> middle -> left -> reflector
                     <- right <- middle <- left <-          -> plugboard -> lamp

Stepping follows the real machine, including the *double-step anomaly*: when
the middle rotor sits on its own notch it steps itself and the left rotor on
the next keypress, so the middle rotor advances twice in two keypresses.

Conventions used throughout the package:

* Letters are integers ``0..25`` (``A``..``Z``); see :func:`letters_to_indices`.
* Rotor tuples are ordered **left, middle, right** — i.e. as printed on the
  machine, with the right-hand (fast) rotor last.
* Ring settings (*Ringstellung*) and ground settings (*Grundstellung*) are
  ``0..25``, so ``0`` means ring/position ``A``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

ALPHABET: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET_SIZE: Final = 26

# Wehrmacht/Kriegsmarine rotor wirings. I-V shipped with Enigma I; VI-VIII were
# Kriegsmarine additions and carry two notches each.
ROTOR_WIRINGS: Final[dict[str, str]] = {
    "I": "EKMFLGDQVZNTOWYHXUSPAIBRCJ",
    "II": "AJDKSIRUXBLHWTMCQGZNPYFVOE",
    "III": "BDFHJLCPRTXVZNYEIWGAKMUSQO",
    "IV": "ESOVPZJAYQUIRHXLNFTGKDCMWB",
    "V": "VZBRGITYUPSDNHLXAWMJQOFECK",
    "VI": "JPGVOUMFYQBENHZRDKASXLICTW",
    "VII": "NZJHGRCXMYSWBOUFAIVLPEKQDT",
    "VIII": "FKQHTLXOCBJSPDZRAMEWNIUYGV",
}

# The window letter(s) at which a rotor turns the rotor to its left over.
ROTOR_NOTCHES: Final[dict[str, str]] = {
    "I": "Q",
    "II": "E",
    "III": "V",
    "IV": "J",
    "V": "Z",
    "VI": "ZM",
    "VII": "ZM",
    "VIII": "ZM",
}

REFLECTOR_WIRINGS: Final[dict[str, str]] = {
    "B": "YRUHQSLDPXNGOKMIEBFZCWVJAT",
    "C": "FVPJIAOYEDRZXWGCTKUQSBNMHL",
}

ROTOR_NAMES: Final[tuple[str, ...]] = tuple(ROTOR_WIRINGS)
REFLECTOR_NAMES: Final[tuple[str, ...]] = tuple(REFLECTOR_WIRINGS)

#: The rotor set the benchmark draws from. Enigma I (army/air force) shipped
#: exactly these five; VI-VIII exist in :data:`ROTOR_WIRINGS` so a machine can
#: be constructed with them, but the benchmark never samples them.
BENCHMARK_ROTORS: Final[tuple[str, ...]] = ("I", "II", "III", "IV", "V")

#: Maximum number of plugboard pairs. 13 would pair every letter; the Wehrmacht
#: used 10 from 1939 onwards, which is what the benchmark uses at its top tiers.
MAX_PLUGBOARD_PAIRS: Final = 13


def letters_to_indices(text: str) -> list[int]:
    """Map ``A``..``Z`` (case-insensitive) to ``0``..``25``, dropping everything else."""
    return [ord(c) - 65 for c in text.upper() if "A" <= c <= "Z"]


def indices_to_letters(indices: Iterable[int]) -> str:
    """Inverse of :func:`letters_to_indices`."""
    return "".join(ALPHABET[i] for i in indices)


class EnigmaSettingsError(ValueError):
    """Raised when a settings combination could not exist on a real machine."""


@dataclass(frozen=True, slots=True)
class EnigmaSettings:
    """A complete machine setup: everything a code clerk read off the key sheet.

    Attributes:
        rotors: Rotor names ordered left, middle, right. Must be distinct.
        reflector: ``"B"`` or ``"C"``.
        ring_settings: *Ringstellung* per rotor, ``0..25``, left to right.
        ground_setting: *Grundstellung* — the window letters the operator dialled
            in before typing, ``0..25``, left to right.
        plugboard: *Steckerbrett* pairs, e.g. ``(("A", "B"), ("C", "D"))``. Each
            letter may appear at most once; order within a pair is irrelevant.
    """

    rotors: tuple[str, str, str]
    reflector: str
    ring_settings: tuple[int, int, int]
    ground_setting: tuple[int, int, int]
    plugboard: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.rotors) != 3:
            raise EnigmaSettingsError(f"expected 3 rotors, got {len(self.rotors)}")
        for name in self.rotors:
            if name not in ROTOR_WIRINGS:
                raise EnigmaSettingsError(f"unknown rotor {name!r}; known: {ROTOR_NAMES}")
        if len(set(self.rotors)) != 3:
            raise EnigmaSettingsError(f"a rotor cannot be fitted twice: {self.rotors}")
        if self.reflector not in REFLECTOR_WIRINGS:
            raise EnigmaSettingsError(
                f"unknown reflector {self.reflector!r}; known: {REFLECTOR_NAMES}"
            )
        for label, values in (
            ("ring_settings", self.ring_settings),
            ("ground_setting", self.ground_setting),
        ):
            if len(values) != 3:
                raise EnigmaSettingsError(f"{label} must have 3 entries, got {len(values)}")
            for value in values:
                if not isinstance(value, int) or not 0 <= value < ALPHABET_SIZE:
                    raise EnigmaSettingsError(f"{label} entries must be ints in 0..25, got {value}")
        self._validate_plugboard()

    def _validate_plugboard(self) -> None:
        if len(self.plugboard) > MAX_PLUGBOARD_PAIRS:
            raise EnigmaSettingsError(
                f"at most {MAX_PLUGBOARD_PAIRS} plugboard pairs, got {len(self.plugboard)}"
            )
        seen: set[str] = set()
        for pair in self.plugboard:
            if len(pair) != 2:
                raise EnigmaSettingsError(f"plugboard entry {pair!r} is not a pair")
            first, second = pair
            for letter in pair:
                if letter not in ALPHABET:
                    raise EnigmaSettingsError(f"plugboard letter {letter!r} is not A-Z uppercase")
            if first == second:
                raise EnigmaSettingsError(f"cannot plug {first!r} into itself")
            if first in seen or second in seen:
                raise EnigmaSettingsError(f"letter reused in plugboard pair {pair!r}")
            seen.update(pair)

    def plugboard_permutation(self) -> tuple[int, ...]:
        """The plugboard as a 26-entry involution over letter indices."""
        table = list(range(ALPHABET_SIZE))
        for first, second in self.plugboard:
            i, j = ord(first) - 65, ord(second) - 65
            table[i], table[j] = j, i
        return tuple(table)

    def to_dict(self) -> dict[str, object]:
        """A JSON-friendly representation, using letters for the 0..25 fields."""
        return {
            "rotors": list(self.rotors),
            "reflector": self.reflector,
            "ring_settings": indices_to_letters(self.ring_settings),
            "ground_setting": indices_to_letters(self.ground_setting),
            "plugboard": [f"{a}{b}" for a, b in self.plugboard],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EnigmaSettings:
        """Inverse of :meth:`to_dict`. Raises :class:`EnigmaSettingsError` on bad input."""
        try:
            rotors = tuple(str(name) for name in data["rotors"])  # type: ignore[union-attr]
            reflector = str(data["reflector"])
            rings = tuple(letters_to_indices(str(data["ring_settings"])))
            ground = tuple(letters_to_indices(str(data["ground_setting"])))
            pairs = tuple((str(p)[0], str(p)[1]) for p in data.get("plugboard", ()))  # type: ignore[union-attr]
        except (KeyError, TypeError, IndexError) as exc:
            raise EnigmaSettingsError(f"malformed settings dict: {data!r}") from exc
        if len(rotors) != 3 or len(rings) != 3 or len(ground) != 3:
            raise EnigmaSettingsError(f"malformed settings dict: {data!r}")
        return cls(
            rotors=(rotors[0], rotors[1], rotors[2]),
            reflector=reflector,
            ring_settings=(rings[0], rings[1], rings[2]),
            ground_setting=(ground[0], ground[1], ground[2]),
            plugboard=pairs,
        )

    def __str__(self) -> str:
        plugs = " ".join(f"{a}{b}" for a, b in self.plugboard) or "-"
        return (
            f"UKW-{self.reflector} | {' '.join(self.rotors)} | "
            f"ring {indices_to_letters(self.ring_settings)} | "
            f"pos {indices_to_letters(self.ground_setting)} | plugs {plugs}"
        )


def _inverse(wiring: str) -> tuple[int, ...]:
    inverse = [0] * ALPHABET_SIZE
    for i, char in enumerate(wiring):
        inverse[ord(char) - 65] = i
    return tuple(inverse)


def _offsets(mapping: Sequence[int]) -> tuple[int, ...]:
    """Store a permutation as per-index deltas so ring/position shifts cancel out.

    For a rotor at effective shift ``s`` the wiring lookup is
    ``(c + delta[(c + s) % 26]) % 26`` — the shift only enters the index, never
    the result, which is what makes the hot loop three adds and a lookup.
    """
    return tuple((value - i) % ALPHABET_SIZE for i, value in enumerate(mapping))


class EnigmaMachine:
    """A 3-rotor Enigma. Construct once per message; :meth:`encrypt` auto-resets.

    Encryption is an involution: with identical settings, encrypting the
    ciphertext returns the plaintext.
    """

    __slots__ = (
        "_backward",
        "_forward",
        "_notches",
        "_plugboard",
        "_positions",
        "_reflector",
        "settings",
    )

    def __init__(self, settings: EnigmaSettings) -> None:
        self.settings = settings
        # Index 0 = left, 1 = middle, 2 = right, matching EnigmaSettings.
        self._forward = tuple(
            _offsets([ord(c) - 65 for c in ROTOR_WIRINGS[name]]) for name in settings.rotors
        )
        self._backward = tuple(_offsets(_inverse(ROTOR_WIRINGS[name])) for name in settings.rotors)
        self._notches = tuple(
            frozenset(ord(c) - 65 for c in ROTOR_NOTCHES[name]) for name in settings.rotors
        )
        self._reflector = tuple(ord(c) - 65 for c in REFLECTOR_WIRINGS[settings.reflector])
        self._plugboard = settings.plugboard_permutation()
        self._positions = list(settings.ground_setting)

    @property
    def positions(self) -> tuple[int, int, int]:
        """Current window letters as indices, left to right."""
        left, middle, right = self._positions
        return (left, middle, right)

    def reset(self) -> None:
        """Return the rotors to the settings' ground position."""
        self._positions = list(self.settings.ground_setting)

    def step(self) -> None:
        """Advance the rotors one keypress, honouring the double-step anomaly."""
        positions = self._positions
        if positions[1] in self._notches[1]:
            # Middle rotor is on its own notch: it steps itself *and* the left
            # rotor. This is the double-step; the pawl engages both wheels.
            positions[0] = (positions[0] + 1) % ALPHABET_SIZE
            positions[1] = (positions[1] + 1) % ALPHABET_SIZE
        elif positions[2] in self._notches[2]:
            positions[1] = (positions[1] + 1) % ALPHABET_SIZE
        positions[2] = (positions[2] + 1) % ALPHABET_SIZE

    def encrypt_index(self, letter: int) -> int:
        """Step the rotors and encipher one letter index. Mutates rotor state."""
        self.step()
        left_shift = self._positions[0] - self.settings.ring_settings[0]
        middle_shift = self._positions[1] - self.settings.ring_settings[1]
        right_shift = self._positions[2] - self.settings.ring_settings[2]

        forward_left, forward_middle, forward_right = self._forward
        backward_left, backward_middle, backward_right = self._backward

        current = self._plugboard[letter]
        current = (current + forward_right[(current + right_shift) % 26]) % 26
        current = (current + forward_middle[(current + middle_shift) % 26]) % 26
        current = (current + forward_left[(current + left_shift) % 26]) % 26
        current = self._reflector[current]
        current = (current + backward_left[(current + left_shift) % 26]) % 26
        current = (current + backward_middle[(current + middle_shift) % 26]) % 26
        current = (current + backward_right[(current + right_shift) % 26]) % 26
        return self._plugboard[current]

    def encrypt_indices(self, letters: Sequence[int]) -> list[int]:
        """Encipher a sequence of letter indices from the ground setting."""
        self.reset()
        return [self.encrypt_index(letter) for letter in letters]

    def encrypt(self, text: str) -> str:
        """Encipher ``text`` from the ground setting, ignoring non-letters."""
        return indices_to_letters(self.encrypt_indices(letters_to_indices(text)))


def encrypt(settings: EnigmaSettings, text: str) -> str:
    """One-shot convenience wrapper around :class:`EnigmaMachine`."""
    return EnigmaMachine(settings).encrypt(text)
