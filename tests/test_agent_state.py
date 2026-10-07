"""Tests for keeping credentials while isolating coding-agent run state."""

from __future__ import annotations

import subprocess
from pathlib import Path

BENCH_DIR = Path(__file__).parents[1]
RESET_SCRIPT = BENCH_DIR / "container" / "reset-agent-state.sh"


def _run_reset(codex_home: Path, claude_home: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [str(RESET_SCRIPT), str(codex_home), str(claude_home)],
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def test_reset_keeps_only_agent_credentials(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    claude_home = tmp_path / "claude"
    codex_home.mkdir()
    claude_home.mkdir()

    (codex_home / "auth.json").write_text("codex-token")
    (codex_home / "config.toml").write_text("untrusted = true")
    (codex_home / "history.jsonl").write_text("prior prompt")
    (codex_home / "sessions").mkdir()
    (codex_home / "sessions" / "prior.jsonl").write_text("prior session")

    (claude_home / ".credentials.json").write_text("claude-token")
    (claude_home / "history.jsonl").write_text("prior prompt")
    (claude_home / "projects").mkdir()
    (claude_home / "projects" / "prior.jsonl").write_text("prior session")
    (claude_home / "settings.json").symlink_to(tmp_path / "outside")

    result = _run_reset(codex_home, claude_home)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in codex_home.iterdir()} == {"auth.json"}
    assert (codex_home / "auth.json").read_text() == "codex-token"
    assert {path.name for path in claude_home.iterdir()} == {".credentials.json"}
    assert (claude_home / ".credentials.json").read_text() == "claude-token"


def test_reset_works_without_stored_credentials(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex"
    claude_home = tmp_path / "claude"
    codex_home.mkdir()
    claude_home.mkdir()
    (codex_home / "sessions").mkdir()
    (claude_home / "projects").mkdir()

    result = _run_reset(codex_home, claude_home)

    assert result.returncode == 0, result.stderr
    assert list(codex_home.iterdir()) == []
    assert list(claude_home.iterdir()) == []


def test_reset_refuses_broad_home_path(tmp_path: Path) -> None:
    claude_home = tmp_path / "claude"
    claude_home.mkdir()

    result = _run_reset(Path("/"), claude_home)

    assert result.returncode == 64
    assert "refusing unsafe Codex home" in result.stderr
