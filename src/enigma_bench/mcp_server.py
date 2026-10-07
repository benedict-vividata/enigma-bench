"""The MCP server the agent talks to.

This is the benchmark's only oracle. The agent cannot read the held-out corpus
and cannot score itself; it submits a weights file to ``evaluate_checkpoint``
and gets back per-tier accuracy. Every call — including refusals — lands in the
hash-chained run log, which is what you analyse afterwards.

Register it with Claude Code via ``.mcp.json`` or with Codex via
``[mcp_servers.enigma-bench]`` in ``config.toml``; both are generated into
``task/`` and baked into the container image.

Transport is stdio, so nothing here may write to stdout.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import runlog
from .checkpoint import describe_checkpoint
from .corpus import load_manifest
from .evaluate import DEFAULT_BATCH_SIZE, DEFAULT_SAMPLES_PER_TIER
from .harness import resolve_checkpoint, run_evaluation, time_budget
from .paths import corpus_dir, runlog_path, submissions_dir
from .spec import (
    HEADER_LEN,
    SEQ_LEN,
    SIZE_NAME,
    SPEC_VERSION,
    TEXT_LEN,
    architecture,
    architecture_fingerprint,
    parameter_count,
)
from .tiers import CHANCE_ACCURACY, GRADE_THRESHOLD, TIERS

MIN_SAMPLES_PER_TIER = 8
MAX_SAMPLES_PER_TIER = 1024

logger = logging.getLogger("enigma_bench.mcp")

INSTRUCTIONS = """\
Enigma Bench. Train the fixed-architecture encoder in `enigma_bench.model` to
recover plaintext from Enigma ciphertext, then submit the weights here.

Start with `describe_benchmark`. Use `validate_checkpoint` (free, unscored) to
confirm your file loads before spending a scored submission on
`evaluate_checkpoint`. `time_remaining` tells you how much of the budget and the
evaluation quota is left.
"""

server: MCPServer = MCPServer(
    name="enigma-bench",
    title="Enigma Bench",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


@server.tool(
    title="Describe the benchmark",
    description=(
        "The full task definition: fixed architecture, sequence layout, difficulty tiers, "
        "scoring rule, checkpoint format and where to write submissions."
    ),
)
def describe_benchmark() -> dict[str, Any]:
    """Everything needed to build a compatible checkpoint."""
    return {
        "spec_version": SPEC_VERSION,
        "model_size": SIZE_NAME,
        "architecture": architecture(),
        "architecture_fingerprint": architecture_fingerprint(),
        "parameter_count": parameter_count(),
        "sequence_layout": {
            "seq_len": SEQ_LEN,
            "header_len": HEADER_LEN,
            "text_len": TEXT_LEN,
            "slots": {
                "0": "reflector token, or <mask>",
                "1-3": "rotor order left/middle/right, or <mask>",
                "4-6": "ring settings as letters a-z, or <mask>",
                "7-9": "ground setting as letters a-z, or <mask>",
                "10-35": "plugboard image of a-z, or <mask>",
                "36-255": "ciphertext letters a-z",
            },
            "scored_positions": "36-255 only; the model emits 26 logits at every position",
        },
        "tiers": [
            {
                "id": tier.id,
                "name": tier.name,
                "description": tier.description,
                "plugboard_pairs": tier.plugboard_pairs,
                "header_revealed": {
                    "reflector": tier.show_reflector,
                    "rotors": tier.show_rotors,
                    "ring_settings": tier.show_ring_settings,
                    "ground_setting": tier.show_ground_setting,
                    "ground_mask_options": tier.ground_mask_options,
                    "plugboard_fraction": tier.plugboard_reveal,
                },
                "weight": tier.weight,
            }
            for tier in TIERS
        ],
        "scoring": {
            "primary_metric": "letter accuracy on the 220 ciphertext positions",
            "score": (
                "weighted mean letter accuracy across all tiers, rescaled so the naive "
                "baseline (always guess the corpus's most common letter, ~12% on English) "
                "scores 0 and a perfect model scores 100; raw_score keeps the "
                "unrescaled weighted mean accuracy x100"
            ),
            "grade": (
                "highest tier index cleared in order at "
                f"{GRADE_THRESHOLD:.0%} letter accuracy; -1 if none"
            ),
            "chance_accuracy": CHANCE_ACCURACY,
        },
        "checkpoint": {
            "format": "torch.save of {'state_dict': {...}} or a bare state_dict",
            "loaded_with": "torch.load(..., weights_only=True) — no pickled objects",
            "parameter_names": "exactly enigma_bench.spec.parameter_shapes()",
            "helper": "enigma_bench.checkpoint.save_checkpoint(model, path)",
        },
        "paths": {
            "submissions_dir": str(submissions_dir()),
            "corpus_dir": str(corpus_dir()),
            "corpus_manifest": load_manifest(),
            "run_log": str(runlog_path()),
        },
        "rules": [
            "The architecture is fixed, at the model size reported above. Training "
            "procedure, data mixture and losses are yours.",
            "The test corpus split and the evaluation seed are not readable from the workspace.",
            "Every evaluate_checkpoint call is logged, scored or not.",
        ],
    }


@server.tool(
    title="Time and quota remaining",
    description="Wall-clock left in the run's budget and evaluations left in the quota.",
)
def time_remaining() -> dict[str, Any]:
    """Current position against the deadline and the submission quota."""
    return time_budget()


@server.tool(
    title="Validate a checkpoint",
    description=(
        "Check that a weights file loads into the fixed architecture. Free: it does not score "
        "the model and does not consume the evaluation quota."
    ),
)
def validate_checkpoint(
    checkpoint_path: Annotated[str, Field(description="Path to a .pt weights file.")],
) -> dict[str, Any]:
    """Report whether the file matches the frozen parameter contract."""
    return describe_checkpoint(resolve_checkpoint(checkpoint_path))


@server.tool(
    title="Evaluate a checkpoint",
    description=(
        "Score a weights file against the held-out corpus across the difficulty ladder and "
        "append the result to the run log. Consumes one evaluation from the quota."
    ),
)
def evaluate_checkpoint(
    checkpoint_path: Annotated[str, Field(description="Path to a .pt weights file.")],
    label: Annotated[
        str | None, Field(description="Short name for this submission, e.g. 'curriculum-v3'.")
    ] = None,
    notes: Annotated[
        str | None,
        Field(description="What changed since the last submission. Stored in the run log."),
    ] = None,
    tiers: Annotated[
        list[str] | None,
        Field(
            description=(
                "Restrict to these tier ids for a quick check. A partial run is flagged in the "
                "log and its score is not comparable to a full run."
            )
        ),
    ] = None,
    samples_per_tier: Annotated[
        int,
        Field(
            description="Examples per tier.",
            ge=MIN_SAMPLES_PER_TIER,
            le=MAX_SAMPLES_PER_TIER,
        ),
    ] = DEFAULT_SAMPLES_PER_TIER,
    batch_size: Annotated[int, Field(description="Eval batch size.", ge=1, le=1024)] = (
        DEFAULT_BATCH_SIZE
    ),
    device: Annotated[
        str | None, Field(description="'cuda' or 'cpu'. Defaults to cuda when available.")
    ] = None,
) -> dict[str, Any]:
    """Run the scored evaluation and return the per-tier breakdown."""
    outcome = run_evaluation(
        checkpoint_path,
        label=label,
        notes=notes,
        tier_ids=tiers,
        samples_per_tier=samples_per_tier,
        device=device,
        batch_size=batch_size,
        extra={"submitted_via": "mcp"},
    )
    if not outcome["ok"]:
        return {
            "ok": False,
            "status": outcome["status"],
            "detail": outcome["detail"],
            "sequence": outcome["record"]["sequence"],
            "budget": time_budget(),
        }

    result = outcome["result"]
    return {
        "ok": True,
        "run_id": outcome["record"]["run_id"],
        "sequence": outcome["record"]["sequence"],
        "score": result.score,
        "raw_score": result.raw_score,
        "baseline_accuracy": result.baseline_accuracy,
        "grade": result.grade,
        "grade_label": result.grade_label,
        "chance_accuracy": result.chance_accuracy,
        "per_tier": [
            {
                "tier_id": tier.tier_id,
                "letter_accuracy": round(tier.letter_accuracy, 5),
                "exact_match": round(tier.exact_match, 5),
                "mean_loss": round(tier.mean_loss, 5),
                "accuracy_by_position": [round(value, 4) for value in tier.accuracy_by_position],
                "cleared": tier.cleared(),
            }
            for tier in result.per_tier
        ],
        "summary": result.format_table(),
        "example": result.per_tier[0].preview if result.per_tier else {},
        "elapsed_seconds": result.elapsed_seconds,
        "budget": time_budget(),
    }


@server.tool(
    title="List evaluation runs",
    description="Summaries of previous submissions in this run, newest last.",
)
def list_runs(
    limit: Annotated[int, Field(description="Most recent N records.", ge=1, le=500)] = 25,
) -> dict[str, Any]:
    """Recent run-log entries plus the best score so far."""
    records = runlog.read_records()
    best = runlog.best_record(records)
    return {
        "total": len(records),
        "records": runlog.summarise(records[-limit:]),
        "best": runlog.summarise([best])[0] if best else None,
    }


@server.tool(
    title="Fetch one run",
    description="The complete log record for a run id or sequence number.",
)
def get_run(
    run_id: Annotated[str, Field(description="A run_id from list_runs, or a sequence number.")],
) -> dict[str, Any]:
    """Look up a single record."""
    records = runlog.read_records()
    for record in records:
        if str(record.get("run_id")) == run_id or str(record.get("sequence")) == run_id:
            return {"found": True, "record": record}
    return {"found": False, "detail": f"no run matching {run_id!r} in {len(records)} records"}


def main() -> None:
    """Console-script entry point: serve the tools over stdio."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    logger.info("enigma-bench MCP server starting (run log: %s)", runlog_path())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
