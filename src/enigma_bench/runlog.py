"""The append-only, hash-chained log of scored evaluations.

Every call to the ``evaluate_checkpoint`` tool appends one JSON object per line.
Each record carries the SHA-256 of the previous record, so an after-the-fact
edit or deletion anywhere in the file is detectable with
:func:`verify_chain` — the log is the benchmark's audit trail, and it is what
you analyse afterwards to compare agents.

The chain proves *tamper-evidence*, not tamper-*proofness*: an agent that can
write the log can rewrite it end to end. Hardening that is the container's job
(the log lives outside the agent's writable tree; see ``container/``), and the
definitive copy is the one the host collects when the run finishes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from .paths import now, runlog_path

GENESIS_HASH: Final = "0" * 64

_CHAIN_FIELDS: Final = ("record_hash",)


def _canonical(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in _CHAIN_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(record: Mapping[str, Any]) -> str:
    """SHA-256 over the record's canonical JSON, excluding the hash field itself."""
    return hashlib.sha256(_canonical(record).encode()).hexdigest()


def read_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Every record in the log, oldest first. Missing log means no records."""
    return list(iter_records(path))


def iter_records(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream records from the log, skipping blank lines."""
    target = path or runlog_path()
    if not target.is_file():
        return
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def append_record(record: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Append one record, stamping sequence number, timestamp and chain hash.

    Returns the stored record, including the fields this function added.
    """
    target = path or runlog_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    previous = read_records(target)
    stored: dict[str, Any] = {
        "sequence": len(previous),
        "logged_at": now().isoformat(),
        "previous_hash": previous[-1]["record_hash"] if previous else GENESIS_HASH,
        **dict(record),
    }
    stored["record_hash"] = compute_hash(stored)

    # O_APPEND write of a single line: concurrent evaluations cannot interleave
    # partial lines, though they can still race on the sequence number.
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stored, default=str) + "\n")
    return stored


def verify_chain(path: Path | None = None) -> tuple[bool, list[str]]:
    """Check sequence numbers and hash links across the whole log.

    Returns:
        ``(ok, problems)`` where ``problems`` lists every inconsistency found.
    """
    problems: list[str] = []
    expected_previous = GENESIS_HASH
    for index, record in enumerate(iter_records(path)):
        if record.get("sequence") != index:
            problems.append(
                f"record {index}: sequence is {record.get('sequence')!r}, expected {index}"
            )
        if record.get("previous_hash") != expected_previous:
            problems.append(f"record {index}: previous_hash does not match record {index - 1}")
        recomputed = compute_hash(record)
        if record.get("record_hash") != recomputed:
            problems.append(f"record {index}: record_hash does not match its contents")
        expected_previous = record.get("record_hash", GENESIS_HASH)
    return (not problems, problems)


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce full records to the handful of fields worth listing."""
    return [
        {
            "sequence": record.get("sequence"),
            "run_id": record.get("run_id"),
            "logged_at": record.get("logged_at"),
            "label": record.get("label"),
            "status": record.get("status"),
            "score": record.get("score"),
            "grade": record.get("grade"),
            "elapsed_since_first_run_seconds": record.get("elapsed_since_first_run_seconds"),
        }
        for record in records
    ]


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-scoring successful record, or ``None`` if there is none."""
    scored = [
        record
        for record in records
        if record.get("status") == "ok" and isinstance(record.get("score"), int | float)
    ]
    return max(scored, key=lambda record: record["score"]) if scored else None


def export_jsonl(records: list[dict[str, Any]], destination: Path) -> Path:
    """Write records to ``destination`` atomically — used by the report command."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".tmp")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")
        staging.replace(destination)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return destination
