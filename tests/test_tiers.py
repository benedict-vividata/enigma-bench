"""The difficulty ladder: sampling, monotonicity and grading."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from enigma_bench.machine import ALPHABET_SIZE
from enigma_bench.tiers import GRADE_THRESHOLD, TIERS, TIERS_BY_ID, grade, resolve_tiers


def test_tier_ids_are_unique_and_ordered() -> None:
    ids = [tier.id for tier in TIERS]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids), "tier ids are t0.. t8 so lexical order is ladder order"


def test_ladder_never_reveals_more_as_it_gets_harder() -> None:
    revealed = [
        (
            tier.show_reflector,
            tier.show_rotors,
            tier.show_ring_settings,
            tier.show_ground_setting,
            tier.plugboard_reveal,
        )
        for tier in TIERS
    ]
    for earlier, later in pairwise(revealed):
        assert all(bool(a) >= bool(b) for a, b in zip(earlier, later, strict=True))


def test_hidden_slot_count_is_non_decreasing_up_the_ladder() -> None:
    rng = random.Random(0)
    hidden = []
    for tier in TIERS:
        settings = tier.sample_settings(rng)
        hidden.append(tier.sample_reveal(settings, rng).hidden_field_count())
    assert hidden == sorted(hidden)


@pytest.mark.parametrize("tier", TIERS, ids=[tier.id for tier in TIERS])
def test_sampled_settings_are_valid_and_match_the_tier(tier) -> None:
    rng = random.Random(7)
    for _ in range(20):
        settings = tier.sample_settings(rng)
        assert len(set(settings.rotors)) == 3
        assert settings.reflector in {"B", "C"}
        assert len(settings.plugboard) == tier.plugboard_pairs


def test_partial_plugboard_reveal_never_splits_a_pair() -> None:
    tier = TIERS_BY_ID["t7_plugboard_half"]
    rng = random.Random(3)
    for _ in range(50):
        settings = tier.sample_settings(rng)
        flags = tier.sample_reveal(settings, rng).plugboard
        for first, second in settings.plugboard:
            assert flags[ord(first) - 65] == flags[ord(second) - 65]
        revealed = sum(flags)
        assert 0 < revealed < ALPHABET_SIZE


def test_partial_ground_tiers_sample_each_mask_pattern() -> None:
    rng = random.Random(11)
    settings = TIERS_BY_ID["t1_ground_one"].sample_settings(rng)

    one = TIERS_BY_ID["t1_ground_one"]
    one_patterns = {
        tuple(not flag for flag in one.sample_reveal(settings, rng).ground_setting_flags())
        for _ in range(600)
    }
    assert one_patterns == {
        (True, False, False),
        (False, True, False),
        (False, False, True),
    }

    two = TIERS_BY_ID["t2_ground_two"]
    two_patterns = {
        tuple(not flag for flag in two.sample_reveal(settings, rng).ground_setting_flags())
        for _ in range(600)
    }
    assert two_patterns == {
        (True, True, False),
        (True, False, True),
        (False, True, True),
    }


def test_partial_ground_reveals_count_hidden_slots(corpus) -> None:
    from enigma_bench.dataset import build_example

    one = build_example(TIERS_BY_ID["t1_ground_one"], corpus, random.Random(2))
    two = build_example(TIERS_BY_ID["t2_ground_two"], corpus, random.Random(3))
    assert one.reveal.hidden_field_count() == 1
    assert two.reveal.hidden_field_count() == 2


def test_grade_requires_clearing_tiers_in_order() -> None:
    passing = {tier.id: 1.0 for tier in TIERS}
    assert grade(passing) == len(TIERS) - 1

    partial = dict(passing)
    partial[TIERS[2].id] = GRADE_THRESHOLD - 0.01
    assert grade(partial) == 1

    # A fluke pass higher up the ladder does not count once a rung is failed.
    skipped = {TIERS[0].id: 1.0, TIERS[1].id: 0.1, TIERS[5].id: 1.0}
    assert grade(skipped) == 0
    assert grade({}) == -1


def test_resolve_tiers_defaults_to_the_whole_ladder() -> None:
    assert resolve_tiers(None) == TIERS
    assert resolve_tiers([]) == TIERS
    assert resolve_tiers(["t4_rings", "t0_full"]) == (
        TIERS_BY_ID["t4_rings"],
        TIERS_BY_ID["t0_full"],
    )
    with pytest.raises(KeyError, match="unknown tier"):
        resolve_tiers(["t9_nope"])
