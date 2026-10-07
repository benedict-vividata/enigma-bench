"""The operator CLI — the same paths the MCP server uses, without an agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enigma_bench.cli import main

FAST = [
    "--tiers", "t0_full",
    "--samples-per-tier", "8",
    "--batch-size", "8",
    "--device", "cpu",
]  # fmt: skip


def test_describe_prints_the_spec_and_ladder(
    bench_home: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["describe"]) == 0
    out = capsys.readouterr().out
    assert "19,067,930" in out
    assert "model size: large" in out
    assert "t8_blind" in out
    assert "encoder-only transformer" in out


def test_sizes_prints_the_whole_ladder(capsys: pytest.CaptureFixture) -> None:
    assert main(["sizes"]) == 0
    out = capsys.readouterr().out
    assert "small" in out and "medium" in out and "large*" in out
    assert "3,237,914" in out and "19,067,930" in out


def test_selftest_passes(capsys: pytest.CaptureFixture) -> None:
    assert main(["selftest"]) == 0
    assert "ok" in capsys.readouterr().out


def test_sample_renders_one_encoded_example(
    bench_home: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["sample", "--tier", "t8_blind", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "reflector : <mask>" in out
    assert "ciphertext:" in out
    assert "plaintext :" in out


def test_random_checkpoint_then_validate_then_evaluate(
    bench_home: Path, capsys: pytest.CaptureFixture
) -> None:
    path = bench_home / "submissions" / "cli.pt"
    assert main(["random-checkpoint", str(path)]) == 0
    capsys.readouterr()

    assert main(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["evaluate", str(path), "--label", "cli", *FAST]) == 0
    out = capsys.readouterr().out
    assert "letter acc" in out
    assert "logged as" in out


def test_evaluate_json_output(bench_home: Path, capsys: pytest.CaptureFixture) -> None:
    path = bench_home / "submissions" / "cli.pt"
    main(["random-checkpoint", str(path)])
    capsys.readouterr()
    assert main(["evaluate", str(path), "--json", *FAST]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["per_tier"][0]["tier_id"] == "t0_full"
    assert payload["chance_accuracy"] == pytest.approx(1 / 26)


def test_evaluate_reports_a_missing_checkpoint(
    bench_home: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["evaluate", "absent.pt", *FAST]) == 1
    assert "checkpoint_missing" in capsys.readouterr().err


def test_validate_exits_non_zero_for_a_bad_file(
    bench_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"nope")
    assert main(["validate", str(junk)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_runs_and_report_summarise_the_log(bench_home: Path, capsys: pytest.CaptureFixture) -> None:
    path = bench_home / "submissions" / "cli.pt"
    main(["random-checkpoint", str(path)])
    # A full-ladder run (no --tiers) so the report has a comparable best submission.
    main(
        [
            "evaluate",
            str(path),
            "--label",
            "one",
            "--samples-per-tier",
            "8",
            "--batch-size",
            "8",
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()

    assert main(["runs", "--verify"]) == 0
    listing = capsys.readouterr().out
    assert "chain: intact" in listing
    assert "one" in listing

    assert main(["runs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["label"] == "one"

    report_path = bench_home / "report.md"
    assert main(["report", "--out", str(report_path)]) == 0
    capsys.readouterr()
    report = report_path.read_text()
    assert "# Enigma Bench run report" in report
    assert "chain integrity: intact" in report
    assert "| `t0_full` |" in report


def test_runs_on_an_empty_log(bench_home: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["runs"]) == 0
    assert "no runs logged" in capsys.readouterr().out


def test_report_flags_a_broken_chain(bench_home: Path, capsys: pytest.CaptureFixture) -> None:
    log = bench_home / "runs" / "runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"sequence": 3, "status": "ok", "record_hash": "x"}) + "\n")
    assert main(["report"]) == 0
    assert "BROKEN" in capsys.readouterr().out
    assert main(["runs", "--verify"]) == 1


def test_budget_reports_the_quota(bench_home: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["budget"]) == 0
    assert json.loads(capsys.readouterr().out)["evaluations_allowed"] == 100


def test_unknown_tier_is_a_parser_error(bench_home: Path) -> None:
    with pytest.raises(SystemExit):
        main(["sample", "--tier", "t9_nope"])
