"""Header/text encoding: the contract between the harness and any training script."""

from __future__ import annotations

import pytest

from enigma_bench.machine import EnigmaSettings
from enigma_bench.spec import (
    GROUND_SLOTS,
    HEADER_LEN,
    IGNORE_INDEX,
    MASK_ID,
    PAD_ID,
    PLUGBOARD_SLOTS,
    REFLECTOR_SLOT,
    RING_SLOTS,
    ROTOR_SLOTS,
    SEQ_LEN,
    TEXT_LEN,
)
from enigma_bench.tokenizer import (
    RevealMask,
    decode_tokens,
    describe_input,
    encode_header,
    encode_input,
    encode_text,
    header_labels,
    letter_id,
    plaintext_labels,
    reflector_id,
    rotor_id,
)

SETTINGS = EnigmaSettings(
    rotors=("IV", "I", "III"),
    reflector="C",
    ring_settings=(0, 13, 25),
    ground_setting=(2, 4, 6),
    plugboard=(("A", "Z"), ("D", "Q")),
)


def test_fully_revealed_header_describes_the_machine() -> None:
    header = encode_header(SETTINGS, RevealMask.everything())
    assert len(header) == HEADER_LEN
    assert header[REFLECTOR_SLOT] == reflector_id("C")
    assert header[ROTOR_SLOTS] == [rotor_id(name) for name in ("IV", "I", "III")]
    assert header[RING_SLOTS] == [letter_id(0), letter_id(13), letter_id(25)]
    assert header[GROUND_SLOTS] == [letter_id(2), letter_id(4), letter_id(6)]
    plugboard = header[PLUGBOARD_SLOTS]
    assert plugboard[0] == letter_id(25)  # A -> Z
    assert plugboard[25] == letter_id(0)  # Z -> A
    assert plugboard[1] == letter_id(1)  # B unplugged, maps to itself


def test_fully_hidden_header_is_all_mask() -> None:
    header = encode_header(SETTINGS, RevealMask.nothing())
    assert header == [MASK_ID] * HEADER_LEN


def test_partial_reveal_masks_only_the_hidden_fields() -> None:
    reveal = RevealMask(reflector=True, rotors=False, ring_settings=False, ground_setting=True)
    header = encode_header(SETTINGS, reveal)
    assert header[REFLECTOR_SLOT] != MASK_ID
    assert header[ROTOR_SLOTS] == [MASK_ID] * 3
    assert header[RING_SLOTS] == [MASK_ID] * 3
    assert header[GROUND_SLOTS] != [MASK_ID] * 3
    assert reveal.hidden_field_count() == 6


def test_plugboard_reveal_flags_are_honoured_per_letter() -> None:
    flags = [index % 2 == 0 for index in range(26)]
    header = encode_header(SETTINGS, RevealMask.everything().with_plugboard(flags))
    plugboard = header[PLUGBOARD_SLOTS]
    assert all(plugboard[i] != MASK_ID for i in range(0, 26, 2))
    assert all(plugboard[i] == MASK_ID for i in range(1, 26, 2))


def test_reveal_mask_rejects_a_wrong_length_plugboard() -> None:
    with pytest.raises(ValueError, match="26 flags"):
        RevealMask(plugboard=(True, False))


def test_short_text_is_padded_and_over_long_text_rejected() -> None:
    encoded = encode_text([0, 1, 2])
    assert len(encoded) == TEXT_LEN
    assert encoded[:3] == [letter_id(0), letter_id(1), letter_id(2)]
    assert set(encoded[3:]) == {PAD_ID}
    with pytest.raises(ValueError, match="exceeds TEXT_LEN"):
        encode_text([0] * (TEXT_LEN + 1))


def test_encode_input_is_header_then_text() -> None:
    ciphertext = list(range(26)) * 8
    encoded = encode_input(SETTINGS, RevealMask.everything(), ciphertext[:TEXT_LEN])
    assert len(encoded) == SEQ_LEN
    assert encoded[:HEADER_LEN] == encode_header(SETTINGS, RevealMask.everything())


def test_plaintext_labels_score_only_text_positions() -> None:
    labels = plaintext_labels([3, 4, 5])
    assert len(labels) == SEQ_LEN
    assert labels[:HEADER_LEN] == [IGNORE_INDEX] * HEADER_LEN
    assert labels[HEADER_LEN : HEADER_LEN + 3] == [3, 4, 5]
    assert labels[HEADER_LEN + 3] == IGNORE_INDEX


def test_header_labels_cover_the_letter_valued_slots_only() -> None:
    labels = header_labels(SETTINGS)
    assert labels[REFLECTOR_SLOT] == IGNORE_INDEX
    assert labels[ROTOR_SLOTS] == [IGNORE_INDEX] * 3
    assert labels[RING_SLOTS] == [0, 13, 25]
    assert labels[GROUND_SLOTS] == [2, 4, 6]
    assert labels[PLUGBOARD_SLOTS] == list(SETTINGS.plugboard_permutation())
    assert labels[HEADER_LEN:] == [IGNORE_INDEX] * TEXT_LEN


def test_decode_and_describe_are_readable() -> None:
    encoded = encode_input(SETTINGS, RevealMask.nothing(), [0, 1, 2])
    assert decode_tokens(encoded[:2]) == "<mask> <mask>"
    rendered = describe_input(encoded)
    assert "reflector : <mask>" in rendered
    assert "ciphertext: abc" in rendered
    with pytest.raises(ValueError, match="expected 256 tokens"):
        describe_input([0, 1])


def test_decode_flags_out_of_range_tokens() -> None:
    assert decode_tokens([999]) == "<bad:999>"
