"""The scored-evaluation entry point shared by the CLI and the MCP server.

One function, :func:`run_evaluation`, owns the whole side-effecting path:
resolve and hash the checkpoint, enforce the time budget and the evaluation
quota, score it, and append an immutable record to the run log. Both front ends
call it so a CLI-triggered eval and an agent-triggered eval produce byte-identical
records.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from . import runlog
from .checkpoint import CheckpointError
from .corpus import CorpusError, load_split
from .evaluate import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLES_PER_TIER,
    DEFAULT_SEED,
    evaluate_checkpoint_file,
)
from .paths import deadline, now, preview_enabled, runlog_path, submissions_dir
from .spec import SIZE_NAME, SPEC_VERSION, architecture_fingerprint
from .tiers import GRADE_THRESHOLD, resolve_tiers

#: ``0`` disables the quota. The default keeps an agent from hill-climbing the
#: held-out set by brute-force resubmission over a long run.
MAX_EVALUATIONS_ENV = "ENIGMA_BENCH_MAX_EVALUATIONS"
DEFAULT_MAX_EVALUATIONS = 100

EVAL_SEED_ENV = "ENIGMA_BENCH_EVAL_SEED"


def max_evaluations() -> int:
    """Evaluation quota for this run; ``0`` means unlimited."""
    raw = os.environ.get(MAX_EVALUATIONS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_EVALUATIONS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_EVALUATIONS


def eval_seed() -> int:
    """Base seed for the evaluation sets.

    The harness operator may randomise this per benchmark run so that two agents
    are compared on independent draws; within a run it is constant, so scores are
    directly comparable across submissions.
    """
    raw = os.environ.get(EVAL_SEED_ENV, "").strip()
    try:
        return int(raw) if raw else DEFAULT_SEED
    except ValueError:
        return DEFAULT_SEED


def file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file — checkpoints are ~76 MiB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def time_budget(log_path: Path | None = None) -> dict[str, Any]:
    """Where the run stands against its wall-clock budget and evaluation quota."""
    records = runlog.read_records(log_path or runlog_path())
    ends_at = deadline()
    remaining = (ends_at - now()).total_seconds() if ends_at else None
    quota = max_evaluations()
    return {
        "now": now().isoformat(),
        "deadline": ends_at.isoformat() if ends_at else None,
        "seconds_remaining": round(remaining, 1) if remaining is not None else None,
        "expired": bool(ends_at and remaining is not None and remaining <= 0),
        "evaluations_used": len(records),
        "evaluations_allowed": quota or None,
        "evaluations_remaining": max(0, quota - len(records)) if quota else None,
    }


def _elapsed_since_first_run(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return 0.0
    try:
        first = datetime.fromisoformat(str(records[0]["logged_at"]))
    except (KeyError, ValueError):
        return None
    return round((now() - first).total_seconds(), 1)


def _rejection(
    reason: str, detail: str, *, label: str | None, checkpoint: Path, log_path: Path
) -> dict[str, Any]:
    record = runlog.append_record(
        {
            "run_id": uuid.uuid4().hex[:12],
            "status": reason,
            "detail": detail,
            "label": label,
            "checkpoint_path": str(checkpoint),
            "spec_version": SPEC_VERSION,
            "model_size": SIZE_NAME,
            "architecture_fingerprint": architecture_fingerprint(),
        },
        log_path,
    )
    return {"ok": False, "status": reason, "detail": detail, "record": record}


def resolve_checkpoint(raw: str) -> Path:
    """Resolve a user-supplied checkpoint path, preferring the submissions directory.

    A bare filename is looked up in :func:`~enigma_bench.paths.submissions_dir`
    so agents can submit ``"best.pt"`` without knowing the mount layout.
    """
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        in_submissions = submissions_dir() / candidate
        if in_submissions.exists():
            return in_submissions.resolve()
    return candidate.resolve()


def run_evaluation(  # noqa: PLR0911 - each refusal is its own guard clause
    checkpoint: str | Path,
    *,
    label: str | None = None,
    notes: str | None = None,
    tier_ids: list[str] | None = None,
    samples_per_tier: int = DEFAULT_SAMPLES_PER_TIER,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    log_path: Path | None = None,
    corpus_root: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a checkpoint and append the outcome to the run log.

    Every call writes exactly one log record, including refusals — a run's
    history shows failed submissions and quota/deadline hits as well as scores.

    Returns:
        ``{"ok": True, "result": ..., "record": ...}`` on success, or
        ``{"ok": False, "status": ..., "detail": ..., "record": ...}``.
    """
    log = log_path or runlog_path()
    path = resolve_checkpoint(str(checkpoint))

    budget = time_budget(log)
    if budget["expired"]:
        return _rejection(
            "deadline_exceeded",
            f"the time budget ended at {budget['deadline']}; no further submissions are scored",
            label=label,
            checkpoint=path,
            log_path=log,
        )
    if budget["evaluations_remaining"] == 0:
        return _rejection(
            "quota_exceeded",
            f"the evaluation quota of {budget['evaluations_allowed']} submissions is used up",
            label=label,
            checkpoint=path,
            log_path=log,
        )
    if not path.is_file():
        return _rejection(
            "checkpoint_missing",
            f"{path} does not exist. Write the weights first, then submit the path.",
            label=label,
            checkpoint=path,
            log_path=log,
        )

    try:
        tiers = resolve_tiers(tier_ids)
    except KeyError as exc:
        return _rejection("bad_request", str(exc), label=label, checkpoint=path, log_path=log)

    try:
        result = evaluate_checkpoint_file(
            path,
            load_split("test", corpus_root),
            tiers=tiers,
            samples_per_tier=samples_per_tier,
            seed=eval_seed(),
            device=device,
            batch_size=batch_size,
            include_preview=preview_enabled(),
        )
    except CheckpointError as exc:
        return _rejection(
            "invalid_checkpoint", str(exc), label=label, checkpoint=path, log_path=log
        )
    except CorpusError as exc:
        return _rejection(
            "corpus_unavailable", str(exc), label=label, checkpoint=path, log_path=log
        )

    previous = runlog.read_records(log)
    record = runlog.append_record(
        {
            "run_id": uuid.uuid4().hex[:12],
            "status": "ok",
            "label": label,
            "notes": notes,
            "checkpoint_path": str(path),
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "grade_threshold": GRADE_THRESHOLD,
            "partial": bool(tier_ids),
            "elapsed_since_first_run_seconds": _elapsed_since_first_run(previous),
            "seconds_remaining": budget["seconds_remaining"],
            **result.to_dict(),
            **dict(extra or {}),
        },
        log,
    )
    return {"ok": True, "result": result, "record": record}
