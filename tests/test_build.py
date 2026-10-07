"""Tests for build-time Codex version resolution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

BENCH_DIR = Path(__file__).parents[1]


def _executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\0' "$@" > "$CAPTURE_ARGS"
"""


NPM_STUB = """#!/usr/bin/env bash
if [[ "$1" == view && "$2" == '@openai/codex@latest' && "$3" == version && "$4" == --silent ]]; then
  printf '0.153.4\\n'
else
  printf 'unexpected npm arguments: %s\\n' "$*" >&2
  exit 1
fi
"""


UV_STUB = """#!/usr/bin/env bash
exit 0
"""


def _run_build(tmp_path: Path, *, codex_version: str | None = None) -> tuple[list[bytes], str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_path = tmp_path / "docker.args"
    _executable(fake_bin / "docker", DOCKER_STUB)
    _executable(fake_bin / "npm", NPM_STUB)
    _executable(fake_bin / "uv", UV_STUB)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_ARGS": str(args_path),
        "ENIGMA_BENCH_OUT": str(tmp_path / "out"),
    }
    if codex_version is not None:
        env["CODEX_VERSION"] = codex_version
    else:
        env.pop("CODEX_VERSION", None)

    result = subprocess.run(  # noqa: S603 - the repo's own script with stubbed tools
        [str(BENCH_DIR / "bench.sh"), "build"],
        cwd=BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return args_path.read_bytes().split(b"\0")[:-1], result.stdout


def test_default_build_passes_resolved_codex_version_as_build_arg(tmp_path: Path) -> None:
    args, stdout = _run_build(tmp_path)

    assert b"--build-arg" in args
    index = args.index(b"--build-arg")
    assert args[index + 1] == b"CODEX_VERSION=0.153.4"
    assert "@openai/codex@latest resolves to 0.153.4" in stdout


def test_explicit_codex_version_skips_latest_lookup_and_build_arg(tmp_path: Path) -> None:
    args, stdout = _run_build(tmp_path, codex_version="0.152.0")

    assert b"--build-arg" not in args
    assert "resolves to" not in stdout
