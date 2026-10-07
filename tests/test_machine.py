"""The Enigma itself, checked against published behaviour and its invariants."""

from __future__ import annotations

import random

import pytest

from enigma_bench.machine import (
    ALPHABET,
    BENCHMARK_ROTORS,
    EnigmaMachine,
    EnigmaSettings,
    EnigmaSettingsError,
    encrypt,
    indices_to_letters,
    letters_to_indices,
)

BASE = EnigmaSettings(
    rotors=("I", "II", "III"), reflector="B", ring_settings=(0, 0, 0), ground_setting=(0, 0, 0)
)


def test_canonical_known_answer_vector() -> None:
    # The standard I-II-III / UKW-B / rings AAA / position AAA test vector.
    assert EnigmaMachine(BASE).encrypt("A" * 25) == "BDZGOWCXLTKSBTMCDLPBMUQOF"


def test_double_step_anomaly() -> None:
    # Starting at ADU the middle wheel steps on two consecutive keypresses.
    machine = EnigmaMachine(
        EnigmaSettings(
            rotors=("I", "II", "III"),
            reflector="B",
            ring_settings=(0, 0, 0),
            ground_setting=(0, 3, 20),
        )
    )
    observed = []
    for _ in range(4):
        machine.step()
        observed.append(indices_to_letters(machine.positions))
    assert observed == ["ADV", "AEW", "BFX", "BFY"]


def test_right_rotor_steps_every_keypress() -> None:
    machine = EnigmaMachine(BASE)
    for expected in range(1, 30):
        machine.step()
        assert machine.positions[2] == expected % 26


def test_encryption_is_reciprocal_and_never_fixes_a_letter() -> None:
    rng = random.Random(4)
    plaintext = "".join(rng.choice(ALPHABET) for _ in range(220))
    for _ in range(25):
        rotors = tuple(rng.sample(BENCHMARK_ROTORS, 3))
        letters = list(ALPHABET)
        rng.shuffle(letters)
        settings = EnigmaSettings(
            rotors=(rotors[0], rotors[1], rotors[2]),
            reflector=rng.choice(("B", "C")),
            ring_settings=(rng.randrange(26), rng.randrange(26), rng.randrange(26)),
            ground_setting=(rng.randrange(26), rng.randrange(26), rng.randrange(26)),
            plugboard=tuple((letters[2 * i], letters[2 * i + 1]) for i in range(10)),
        )
        ciphertext = encrypt(settings, plaintext)
        assert encrypt(settings, ciphertext) == plaintext
        assert all(a != b for a, b in zip(plaintext, ciphertext, strict=True))


def test_left_rotor_ring_and_position_shift_together_is_a_no_op() -> None:
    # The left wheel's notch drives nothing on a three-rotor machine, so only the
    # difference between its ring and its position can affect the output.
    plaintext = "ATTACKATDAWNSTOPMESSAGEENDS"
    reference = encrypt(BASE, plaintext)
    for shift in range(1, 26):
        shifted = EnigmaSettings(
            rotors=BASE.rotors,
            reflector=BASE.reflector,
            ring_settings=(shift, 0, 0),
            ground_setting=(shift, 0, 0),
        )
        assert encrypt(shifted, plaintext) == reference


def test_machine_resets_between_messages() -> None:
    machine = EnigmaMachine(BASE)
    first = machine.encrypt("HELLOWORLD")
    assert machine.encrypt("HELLOWORLD") == first


def test_non_letters_are_dropped() -> None:
    assert encrypt(BASE, "aaa aa!") == encrypt(BASE, "AAAAA")


def test_plugboard_permutation_is_an_involution() -> None:
    settings = EnigmaSettings(
        rotors=("IV", "V", "I"),
        reflector="C",
        ring_settings=(3, 14, 25),
        ground_setting=(1, 2, 3),
        plugboard=(("A", "M"), ("Q", "Z")),
    )
    table = settings.plugboard_permutation()
    assert len(table) == 26
    assert all(table[table[i]] == i for i in range(26))
    assert table[0] == 12 and table[12] == 0


def test_settings_roundtrip_through_dict() -> None:
    settings = EnigmaSettings(
        rotors=("V", "III", "II"),
        reflector="C",
        ring_settings=(0, 12, 25),
        ground_setting=(7, 7, 7),
        plugboard=(("B", "Q"), ("H", "X")),
    )
    assert EnigmaSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rotors": ("I", "I", "II")}, "twice"),
        ({"rotors": ("I", "II", "IX")}, "unknown rotor"),
        ({"reflector": "D"}, "unknown reflector"),
        ({"ring_settings": (0, 0, 26)}, "0..25"),
        ({"ground_setting": (0, 0)}, "3 entries"),
        ({"plugboard": (("A", "A"),)}, "into itself"),
        ({"plugboard": (("A", "B"), ("B", "C"))}, "reused"),
        ({"plugboard": (("a", "B"),)}, "uppercase"),
    ],
)
def test_impossible_settings_are_rejected(kwargs: dict, message: str) -> None:
    base = {
        "rotors": ("I", "II", "III"),
        "reflector": "B",
        "ring_settings": (0, 0, 0),
        "ground_setting": (0, 0, 0),
    }
    with pytest.raises(EnigmaSettingsError, match=message):
        EnigmaSettings(**{**base, **kwargs})


def test_letter_index_helpers_round_trip() -> None:
    assert letters_to_indices("Hello, World!") == [7, 4, 11, 11, 14, 22, 14, 17, 11, 3]
    assert indices_to_letters([0, 25]) == "AZ"
