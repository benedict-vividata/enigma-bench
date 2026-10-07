#!/usr/bin/env bash
#
# Enigma Bench, end to end.
#
#   ./bench.sh test                        lint + unit tests on this machine
#   ./bench.sh corpus                      build the corpus locally (optional)
#   ./bench.sh build                       build the task image
#   ./bench.sh run --agent codex --model <model> --hours 6
#   ./bench.sh run --size medium           smaller benchmark model (small|medium|large)
#   ./bench.sh run-bg --agent codex --model <model> --effort high --hours 6 \
#       --prompt 'Prioritise experiments that improve t0 generalisation.'
#   ./bench.sh sizes                       print the model-size ladder (no Docker needed)
#   ./bench.sh shell                       interactive container, same isolation
#   ./bench.sh runs                        list logged evaluations
#   ./bench.sh report [--out FILE]         markdown summary of the run log
#   ./bench.sh clean                       remove the image and local artefacts
#
# `test` and `corpus` need only uv. Everything else needs Docker with the NVIDIA
# Container Toolkit; see container/README.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
readonly COMPOSE_FILE=container/docker-compose.yml

# Absolute, so it means the same thing here and inside the compose file (whose
# relative paths resolve against container/).
OUT_DIR="${ENIGMA_BENCH_OUT:-out}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
export ENIGMA_BENCH_OUT="$OUT_DIR"
readonly OUT_DIR
readonly RUN_PID_FILE="$OUT_DIR/bench-run.pid"
readonly RUN_STDOUT_LOG="$OUT_DIR/bench-run.log"

die() { printf 'bench: %s\n' "$*" >&2; exit 1; }

# The benchmark model's capacity. One size per run, fixed before the container
# starts: it decides the architecture the agent trains and the scorer loads, so
# a checkpoint from one size is not scoreable at another.
readonly SIZES="small medium large"
check_size() {
    local candidate
    for candidate in $SIZES; do
        [[ "$1" == "$candidate" ]] && return 0
    done
    die "unknown --size '$1'; expected one of ${SIZES// /, }"
}

compose() {
    command -v docker >/dev/null || die "docker is not installed"
    docker compose -f "$COMPOSE_FILE" "$@"
}

require_uv() { command -v uv >/dev/null || die "uv is not installed: https://docs.astral.sh/uv/"; }

latest_codex_version() {
    command -v npm >/dev/null || die "npm is required to resolve @openai/codex@latest before a build"

    local version
    version="$(npm view @openai/codex@latest version --silent 2>/dev/null)" ||
        die "could not resolve @openai/codex@latest; set CODEX_VERSION explicitly to build offline"
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]] ||
        die "npm returned an invalid @openai/codex version: '$version'"
    printf '%s\n' "$version"
}

cmd_test() {
    require_uv
    uv sync --quiet
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest "$@"
}

cmd_corpus() {
    require_uv
    local out status
    out="${1:-corpus}"

    # A manifest is written last by fetch_corpus.py, so this is an atomic-ish
    # completion marker.  It also makes repeated builds cheap and safe.
    if [[ -s "$out/train.txt" && -s "$out/valid.txt" && \
          -s "$out/test.txt" && -s "$out/manifest.json" ]]; then
        printf 'bench: corpus already present in %s\n' "$out"
        return 0
    fi

    # datasets/aiohttp can leave a native worker thread behind after a complete
    # download.  Bound that shutdown problem, then accept the result only when
    # the fetcher left a complete corpus behind.
    set +e
    timeout --signal=TERM --kill-after=15 10m \
        uv run --no-project --python 3.12 --with "datasets>=4,<6" \
            python scripts/fetch_corpus.py --out "$out" "${@:2}"
    status=$?
    set -e

    if [[ -s "$out/train.txt" && -s "$out/valid.txt" && \
          -s "$out/test.txt" && -s "$out/manifest.json" ]]; then
        if [[ "$status" -ne 0 ]]; then
            printf 'bench: fetch exited %s after writing a complete corpus; continuing\n' \
                "$status" >&2
        fi
        return 0
    fi
    return "$status"
}

cmd_build() {
    cmd_corpus

    # Docker cannot tell that the registry's `latest` tag changed, so a cached
    # `npm install @openai/codex@latest` layer can silently contain an old CLI.
    # Resolve the tag to its immutable version before the build; the build arg
    # changes, and therefore invalidates the npm layer, only when the package's
    # latest version changes. An explicit CODEX_VERSION remains fully pinned.
    local -a build_args=("$@")
    if [[ -z "${CODEX_VERSION:-}" || "$CODEX_VERSION" == "latest" ]]; then
        local codex_version
        codex_version="$(latest_codex_version)"
        build_args+=(--build-arg "CODEX_VERSION=$codex_version")
        printf 'bench: @openai/codex@latest resolves to %s\n' "$codex_version"
    fi
    compose build "${build_args[@]}"
}

cmd_run() {
    local agent model effort hours prompt size
    agent="${ENIGMA_BENCH_AGENT:-claude}"
    model="${ENIGMA_BENCH_MODEL:-${ENIGMA_BENCH_CODEX_MODEL:-gpt-5.6-luna}}"
    effort="${ENIGMA_BENCH_EFFORT:-}"
    hours="${ENIGMA_BENCH_HOURS:-6}"
    prompt="${ENIGMA_BENCH_PROMPT:-}"
    size="${ENIGMA_BENCH_SIZE:-large}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --agent) agent="$2"; shift 2 ;;
            --model) model="$2"; shift 2 ;;
            --effort) effort="$2"; shift 2 ;;
            --hours) hours="$2"; shift 2 ;;
            --prompt) prompt="$2"; shift 2 ;;
            --size) size="$2"; shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    check_size "$size"
    mkdir -p "$OUT_DIR"/{runs,submissions,transcripts}
    printf 'bench: %s (%s) on the %s model for %s hours; log -> %s/runs/runs.jsonl\n' \
        "$agent" "$model" "$size" "$hours" "$OUT_DIR"
    ENIGMA_BENCH_AGENT="$agent" ENIGMA_BENCH_MODEL="$model" ENIGMA_BENCH_EFFORT="$effort" \
    ENIGMA_BENCH_HOURS="$hours" ENIGMA_BENCH_SIZE="$size" \
        compose run --rm --service-ports -e "ENIGMA_BENCH_PROMPT=$prompt" bench run
}

cmd_run_bg() {
    local pid
    mkdir -p "$OUT_DIR"
    if [[ -r "$RUN_PID_FILE" ]]; then
        pid="$(<"$RUN_PID_FILE")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && \
           ps -p "$pid" -o args= 2>/dev/null | grep -q '[b]ench.sh run'; then
            die "a detached benchmark is already running (pid $pid)"
        fi
    fi
    rm -f "$RUN_PID_FILE"
    nohup "$PWD/bench.sh" run "$@" >"$RUN_STDOUT_LOG" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$RUN_PID_FILE"
    printf 'Started benchmark in background.\nPID: %s\nLog: %s\n\n' "$!" "$RUN_STDOUT_LOG"
    printf 'Watch it with: ./watch-run.sh dashboard\n'
}

cmd_sizes() {
    require_uv
    uv sync --quiet
    uv run enigma-bench sizes
}

cmd_shell() {
    mkdir -p "$OUT_DIR"/{runs,submissions,transcripts}
    compose run --rm bench shell
}

cmd_runs()   { compose run --rm bench score runs --verify "$@"; }
cmd_report() { compose run --rm bench score report "$@"; }

cmd_clean() {
    compose down --remove-orphans 2>/dev/null || true
    docker image rm "enigma-bench:${ENIGMA_BENCH_TAG:-latest}" 2>/dev/null || true
    rm -rf .venv .pytest_cache .ruff_cache
    printf 'bench: left %s in place; delete it by hand if you mean to\n' "$OUT_DIR"
}

case "${1:-}" in
    test)   shift; cmd_test "$@" ;;
    corpus) shift; cmd_corpus "$@" ;;
    build)  shift; cmd_build "$@" ;;
    run)    shift; cmd_run "$@" ;;
    run-bg) shift; cmd_run_bg "$@" ;;
    sizes)  shift; cmd_sizes ;;
    shell)  shift; cmd_shell ;;
    runs)   shift; cmd_runs "$@" ;;
    report) shift; cmd_report "$@" ;;
    clean)  shift; cmd_clean ;;
    ""|-h|--help)
        awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
        ;;
    *) die "unknown command '$1'; try --help" ;;
esac
