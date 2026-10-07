"""The append-only run log and its tamper-evident hash chain."""

from __future__ import annotations

import json
from pathlib import Path

from enigma_bench import runlog


def test_append_stamps_sequence_timestamp_and_chain(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    first = runlog.append_record({"run_id": "a", "status": "ok", "score": 1.0}, log)
    second = runlog.append_record({"run_id": "b", "status": "ok", "score": 2.0}, log)

    assert first["sequence"] == 0
    assert first["previous_hash"] == runlog.GENESIS_HASH
    assert second["sequence"] == 1
    assert second["previous_hash"] == first["record_hash"]
    assert first["logged_at"] <= second["logged_at"]
    assert runlog.verify_chain(log) == (True, [])


def test_missing_log_reads_as_empty(tmp_path: Path) -> None:
    assert runlog.read_records(tmp_path / "nothing.jsonl") == []
    assert runlog.verify_chain(tmp_path / "nothing.jsonl") == (True, [])


def test_editing_a_record_breaks_the_chain(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    runlog.append_record({"run_id": "a", "status": "ok", "score": 1.0}, log)
    runlog.append_record({"run_id": "b", "status": "ok", "score": 2.0}, log)

    records = runlog.read_records(log)
    records[0]["score"] = 99.0
    log.write_text("".join(json.dumps(record) + "\n" for record in records))

    ok, problems = runlog.verify_chain(log)
    assert not ok
    assert any("does not match its contents" in problem for problem in problems)


def test_deleting_a_record_breaks_the_chain(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    for index in range(3):
        runlog.append_record({"run_id": str(index), "status": "ok"}, log)
    records = runlog.read_records(log)
    del records[1]
    log.write_text("".join(json.dumps(record) + "\n" for record in records))

    ok, problems = runlog.verify_chain(log)
    assert not ok
    assert any("sequence" in problem for problem in problems)


def test_best_record_ignores_failures_and_unscored_entries(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    runlog.append_record({"run_id": "a", "status": "ok", "score": 10.0}, log)
    runlog.append_record({"run_id": "b", "status": "invalid_checkpoint"}, log)
    runlog.append_record({"run_id": "c", "status": "ok", "score": 42.0}, log)
    runlog.append_record({"run_id": "d", "status": "ok", "score": 3.0}, log)

    best = runlog.best_record(runlog.read_records(log))
    assert best is not None
    assert best["run_id"] == "c"
    assert runlog.best_record([]) is None


def test_summarise_keeps_only_the_headline_fields(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    runlog.append_record({"run_id": "a", "status": "ok", "score": 1.0, "bulk": [1] * 100}, log)
    summary = runlog.summarise(runlog.read_records(log))[0]
    assert "bulk" not in summary
    assert summary["run_id"] == "a"


def test_export_writes_atomically(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    runlog.append_record({"run_id": "a", "status": "ok"}, log)
    destination = runlog.export_jsonl(runlog.read_records(log), tmp_path / "out" / "copy.jsonl")
    assert destination.is_file()
    assert len(destination.read_text().splitlines()) == 1
    assert not list(destination.parent.glob("*.tmp"))
