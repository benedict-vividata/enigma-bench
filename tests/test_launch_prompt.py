"""Tests for passing operator prompt text into agent startup."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

BENCH_DIR = Path(__file__).parents[1]
DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\0' "$@" > "$CAPTURE_ARGS"
printf '%s' "${ENIGMA_BENCH_PROMPT-}" > "$CAPTURE_ENV"
"""
CAT_STUB = """#!/usr/bin/env bash
if [[ "$1" == /workspace/TASK.md ]]; then
  printf 'Base task from benchmark'
else
  exec /usr/bin/cat "$@"
fi
"""
CLAUDE_STUB = """#!/usr/bin/env bash
printf '%s' "${!#}" > "$CAPTURE_PROMPT"
"""
CODEX_STUB = """#!/usr/bin/env bash
if [[ "${2-}" == resume ]]; then
  exit 1
fi
printf '%s' "${!#}" > "$CAPTURE_PROMPT"
printf '{"type":"thread.started","thread_id":"test-thread"}\\n'
"""


def _executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_bench_run_passes_multiline_prompt_as_compose_override(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_path = tmp_path / "docker.args"
    env_path = tmp_path / "prompt.env"
    _executable(
        fake_bin / "docker",
        DOCKER_STUB,
    )

    prompt = "Focus on t0 first.\nThen test rotor generalisation."
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_ARGS": str(args_path),
        "CAPTURE_ENV": str(env_path),
        "ENIGMA_BENCH_OUT": str(tmp_path / "out"),
    }
    result = subprocess.run(  # noqa: S603
        [
            str(BENCH_DIR / "bench.sh"),
            "run",
            "--agent",
            "claude",
            "--model",
            "claude-fable-5",
            "--hours",
            "12",
            "--prompt",
            prompt,
        ],
        cwd=BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = args_path.read_bytes().split(b"\0")[:-1]
    assert b"-e" in args
    assert f"ENIGMA_BENCH_PROMPT={prompt}".encode() in args
    assert env_path.read_text() == ""


def test_run_agent_appends_prompt_for_claude(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "claude.prompt"
    _executable(
        fake_bin / "cat",
        CAT_STUB,
    )
    _executable(
        fake_bin / "claude",
        CLAUDE_STUB,
    )
    credentials = tmp_path / "claude"
    credentials.mkdir()
    (credentials / ".credentials.json").write_text("{}")

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CLAUDE_CONFIG_DIR": str(credentials),
        "ENIGMA_BENCH_HOME": str(tmp_path / "home"),
        "ENIGMA_BENCH_AGENT": "claude",
        "ENIGMA_BENCH_MODEL": "claude-fable-5",
        "ENIGMA_BENCH_PROMPT": "Operator note\nwith two lines",
        "CAPTURE_PROMPT": str(capture),
    }
    result = subprocess.run(  # noqa: S603
        [str(BENCH_DIR / "container" / "run-agent.sh")],
        cwd=BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text() == (
        "Base task from benchmark\n\nAdditional operator prompt:\nOperator note\nwith two lines"
    )


def test_run_agent_appends_prompt_for_codex(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "codex.prompt"
    _executable(
        fake_bin / "cat",
        CAT_STUB,
    )
    _executable(
        fake_bin / "codex",
        CODEX_STUB,
    )
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}")

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CODEX_HOME": str(codex_home),
        "ENIGMA_BENCH_HOME": str(tmp_path / "home"),
        "ENIGMA_BENCH_AGENT": "codex",
        "ENIGMA_BENCH_MODEL": "gpt-test",
        "ENIGMA_BENCH_PROMPT": "Operator note for Codex",
        "CAPTURE_PROMPT": str(capture),
    }
    result = subprocess.run(  # noqa: S603
        [str(BENCH_DIR / "container" / "run-agent.sh")],
        cwd=BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert capture.read_text() == (
        "Base task from benchmark\n\nAdditional operator prompt:\nOperator note for Codex"
    )
