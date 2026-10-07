"""The scored-evaluation path: logging, refusals, deadline and quota."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from enigma_bench import runlog
from enigma_bench.harness import file_sha256, resolve_checkpoint, run_evaluation, time_budget
from enigma_bench.paths import now, runlog_path

FAST = {"tier_ids": ["t0_full"], "samples_per_tier": 8, "batch_size": 8, "device": "cpu"}


def test_a_successful_evaluation_is_scored_and_logged(bench_home: Path, checkpoint: Path) -> None:
    outcome = run_evaluation(checkpoint, label="first", notes="untrained", **FAST)
    assert outcome["ok"] is True

    result = outcome["result"]
    assert -100.0 <= result.score <= 100.0
    assert 0.0 <= result.raw_score <= 100.0
    assert 0.0 < result.baseline_accuracy < 1.0
    assert result.grade == -1, "untrained weights cannot clear a tier"
    assert [tier.tier_id for tier in result.per_tier] == ["t0_full"]

    records = runlog.read_records(runlog_path())
    assert len(records) == 1
    assert records[0]["label"] == "first"
    assert records[0]["notes"] == "untrained"
    assert records[0]["partial"] is True
    assert records[0]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert records[0]["model_size"] == "large", "the log must say which model was scored"
    assert "submitted_via" not in records[0]
    assert runlog.verify_chain(runlog_path()) == (True, [])


def test_extra_fields_are_merged_into_the_record(bench_home: Path, checkpoint: Path) -> None:
    run_evaluation(checkpoint, extra={"submitted_via": "unit-test"}, **FAST)
    assert runlog.read_records(runlog_path())[0]["submitted_via"] == "unit-test"


def test_untrained_weights_score_around_chance(bench_home: Path, checkpoint: Path) -> None:
    result = run_evaluation(checkpoint, **FAST)["result"]
    accuracy = result.per_tier[0].letter_accuracy
    assert 0.0 <= accuracy < 0.15, "an untrained model must sit near 1/26"
    assert result.per_tier[0].exact_match == 0.0
    assert len(result.per_tier[0].accuracy_by_position) == 11
    assert result.per_tier[0].preview["plaintext"] != result.per_tier[0].preview["ciphertext"]


def test_a_bare_filename_resolves_inside_the_submissions_directory(
    bench_home: Path, checkpoint: Path
) -> None:
    assert resolve_checkpoint(checkpoint.name) == checkpoint
    outcome = run_evaluation(checkpoint.name, **FAST)
    assert outcome["ok"] is True


def test_a_missing_checkpoint_is_refused_and_still_logged(bench_home: Path) -> None:
    outcome = run_evaluation("nowhere.pt", **FAST)
    assert outcome["ok"] is False
    assert outcome["status"] == "checkpoint_missing"
    assert runlog.read_records(runlog_path())[0]["status"] == "checkpoint_missing"


def test_a_malformed_checkpoint_is_refused(bench_home: Path) -> None:
    junk = bench_home / "submissions" / "junk.pt"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"not a checkpoint")
    outcome = run_evaluation(junk, **FAST)
    assert outcome["status"] == "invalid_checkpoint"


def test_an_unknown_tier_is_refused(bench_home: Path, checkpoint: Path) -> None:
    outcome = run_evaluation(checkpoint, tier_ids=["t9_nope"], samples_per_tier=8, device="cpu")
    assert outcome["status"] == "bad_request"


def test_submissions_after_the_deadline_are_refused(
    bench_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_DEADLINE", (now() - timedelta(minutes=1)).isoformat())
    outcome = run_evaluation(checkpoint, **FAST)
    assert outcome["status"] == "deadline_exceeded"
    assert time_budget()["expired"] is True


def test_the_evaluation_quota_is_enforced(
    bench_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_MAX_EVALUATIONS", "2")
    assert run_evaluation(checkpoint, **FAST)["ok"] is True
    assert run_evaluation(checkpoint, **FAST)["ok"] is True
    third = run_evaluation(checkpoint, **FAST)
    assert third["status"] == "quota_exceeded"
    assert time_budget()["evaluations_remaining"] == 0


def test_an_unlimited_quota_is_configurable(
    bench_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_MAX_EVALUATIONS", "0")
    assert time_budget()["evaluations_allowed"] is None
    monkeypatch.setenv("ENIGMA_BENCH_MAX_EVALUATIONS", "not-a-number")
    assert time_budget()["evaluations_allowed"] == 100


def test_the_evaluation_seed_changes_the_held_out_examples(
    bench_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_EVAL_SEED", "1")
    first = run_evaluation(checkpoint, **FAST)["result"].per_tier[0].preview["ciphertext"]
    monkeypatch.setenv("ENIGMA_BENCH_EVAL_SEED", "2")
    second = run_evaluation(checkpoint, **FAST)["result"].per_tier[0].preview["ciphertext"]
    assert first != second


def test_previews_can_be_switched_off(
    bench_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_PREVIEW", "0")
    result = run_evaluation(checkpoint, **FAST)["result"]
    assert result.per_tier[0].preview == {}, "held-out plaintext must not leak into the result"
    assert runlog.read_records(runlog_path())[0]["per_tier"][0]["preview"] == {}
