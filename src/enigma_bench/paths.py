"""Where the benchmark keeps its data, on the host and inside the container.

Every path is overridable by an environment variable so the same code runs from
a checkout (``ENIGMA_BENCH_HOME`` unset) and from the hardened image, where the
corpus and the run log live outside the agent's writable workspace.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

#: Root for everything the harness owns. The container image sets this to
#: ``/srv/enigma-bench``; locally it defaults to ``./.enigma-bench``.
HOME_ENV: Final = "ENIGMA_BENCH_HOME"
CORPUS_ENV: Final = "ENIGMA_BENCH_CORPUS"
RUNLOG_ENV: Final = "ENIGMA_BENCH_RUNLOG"
SUBMISSIONS_ENV: Final = "ENIGMA_BENCH_SUBMISSIONS"
DEADLINE_ENV: Final = "ENIGMA_BENCH_DEADLINE"
PREVIEW_ENV: Final = "ENIGMA_BENCH_PREVIEW"


def home() -> Path:
    """Root directory for corpus, run log and submissions."""
    return Path(os.environ.get(HOME_ENV, Path.cwd() / ".enigma-bench"))


def corpus_dir() -> Path:
    """Directory holding ``train.txt``, ``valid.txt``, ``test.txt`` and ``manifest.json``."""
    override = os.environ.get(CORPUS_ENV)
    return Path(override) if override else home() / "corpus"


def runlog_path() -> Path:
    """Append-only JSONL log of every scored evaluation."""
    override = os.environ.get(RUNLOG_ENV)
    return Path(override) if override else home() / "runs" / "runs.jsonl"


def submissions_dir() -> Path:
    """Drop directory the agent writes checkpoints into."""
    override = os.environ.get(SUBMISSIONS_ENV)
    return Path(override) if override else home() / "submissions"


def deadline() -> datetime | None:
    """The end of the agent's time budget, if the harness set one.

    Returns:
        A timezone-aware UTC datetime, or ``None`` when ``ENIGMA_BENCH_DEADLINE``
        is unset or unparseable.
    """
    raw = os.environ.get(DEADLINE_ENV, "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def preview_enabled() -> bool:
    """Whether evaluation results may quote held-out plaintext.

    Previews are a side-by-side of true and predicted plaintext for one example
    per tier — invaluable for debugging, and a slow leak of the held-out set
    across many submissions. Set ``ENIGMA_BENCH_PREVIEW=0`` for a run where the
    score has to be defensible; the operator can still see previews by scoring
    the checkpoint themselves afterwards.
    """
    return os.environ.get(PREVIEW_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def now() -> datetime:
    """Current UTC time — one definition so run logs are internally consistent."""
    return datetime.now(UTC)
