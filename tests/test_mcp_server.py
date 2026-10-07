"""The tool surface the agent actually sees, exercised through an MCP client."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from enigma_bench.mcp_server import server
from enigma_bench.paths import now
from enigma_bench.tiers import TIERS

FAST: dict[str, Any] = {
    "tiers": ["t0_full"],
    "samples_per_tier": 8,
    "batch_size": 8,
    "device": "cpu",
}


async def _call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    async with Client(server) as client:
        result = await client.call_tool(name, arguments or {})
    assert result.structured_content is not None
    return result.structured_content


@pytest.mark.anyio
async def test_every_tool_is_advertised() -> None:
    async with Client(server) as client:
        listing = await client.list_tools()
    assert {tool.name for tool in listing.tools} == {
        "describe_benchmark",
        "time_remaining",
        "validate_checkpoint",
        "evaluate_checkpoint",
        "list_runs",
        "get_run",
    }


@pytest.mark.anyio
async def test_describe_benchmark_covers_the_whole_contract(bench_home: Path) -> None:
    described = await _call("describe_benchmark")
    assert described["architecture"]["seq_len"] == 256
    assert described["parameter_count"] == 19_067_930
    assert described["model_size"] == "large"
    assert [tier["id"] for tier in described["tiers"]] == [tier.id for tier in TIERS]
    assert "state_dict" in described["checkpoint"]["format"]
    assert described["paths"]["submissions_dir"].endswith("submissions")


@pytest.mark.anyio
async def test_validate_is_free_and_does_not_log(bench_home: Path, checkpoint: Path) -> None:
    report = await _call("validate_checkpoint", {"checkpoint_path": str(checkpoint)})
    assert report["valid"] is True
    assert (await _call("list_runs"))["total"] == 0


@pytest.mark.anyio
async def test_evaluate_scores_logs_and_reports_the_budget(
    bench_home: Path, checkpoint: Path
) -> None:
    response = await _call(
        "evaluate_checkpoint", {"checkpoint_path": str(checkpoint), "label": "run-1", **FAST}
    )
    assert response["ok"] is True
    assert response["grade_label"] == "none"
    assert response["per_tier"][0]["tier_id"] == "t0_full"
    assert response["per_tier"][0]["cleared"] is False
    assert response["budget"]["evaluations_used"] == 1
    assert "letter acc" in response["summary"]

    listing = await _call("list_runs", {"limit": 5})
    assert listing["total"] == 1
    assert listing["best"]["label"] == "run-1"

    fetched = await _call("get_run", {"run_id": response["run_id"]})
    assert fetched["found"] is True
    assert fetched["record"]["sequence"] == response["sequence"]

    by_sequence = await _call("get_run", {"run_id": str(response["sequence"])})
    assert by_sequence["found"] is True


@pytest.mark.anyio
async def test_unknown_run_is_reported_not_raised(bench_home: Path) -> None:
    assert (await _call("get_run", {"run_id": "nope"}))["found"] is False


@pytest.mark.anyio
async def test_a_broken_submission_comes_back_as_a_refusal(bench_home: Path) -> None:
    response = await _call("evaluate_checkpoint", {"checkpoint_path": "missing.pt", **FAST})
    assert response["ok"] is False
    assert response["status"] == "checkpoint_missing"
    assert (await _call("list_runs"))["total"] == 1, "refusals are logged too"


@pytest.mark.anyio
async def test_the_deadline_closes_the_evaluate_tool(
    bench_home: Path, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENIGMA_BENCH_DEADLINE", (now() - timedelta(seconds=1)).isoformat())
    assert (await _call("time_remaining"))["expired"] is True
    response = await _call("evaluate_checkpoint", {"checkpoint_path": str(checkpoint), **FAST})
    assert response["status"] == "deadline_exceeded"


@pytest.mark.anyio
async def test_samples_per_tier_is_bounded(bench_home: Path, checkpoint: Path) -> None:
    async with Client(server) as client:
        result = await client.call_tool(
            "evaluate_checkpoint", {"checkpoint_path": str(checkpoint), "samples_per_tier": 99_999}
        )
    assert result.is_error
