#!/usr/bin/env bash
#
# Container entrypoint. Runs as root: fixes up mounted volumes, stamps the
# deadline, locks down egress, then drops to the unprivileged `agent` user for
# the actual run.
#
#   run    (default) start the coding agent against TASK.md
#   shell  drop into a shell as `agent` — for developing the harness itself
#   score  run one evaluation as `bench` and exit, for smoke-testing the image
set -euo pipefail

MODE="${1:-run}"
HOME_DIR="${ENIGMA_BENCH_HOME:-/srv/enigma-bench}"
HOURS="${ENIGMA_BENCH_HOURS:-6}"
SIZE="${ENIGMA_BENCH_SIZE:-large}"

log() { printf '[enigma-bench] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Deadline. Written to disk as well as exported, because sudo scrubs the
# environment when the agent starts the MCP server as `bench`.
# ---------------------------------------------------------------------------
DEADLINE="$(date -u -d "+${HOURS} hours" +%Y-%m-%dT%H:%M:%S+00:00)"
install -d -o root -g root -m 0755 "$HOME_DIR"
cat > "$HOME_DIR/env" <<ENVEOF
ENIGMA_BENCH_HOME=${HOME_DIR}
ENIGMA_BENCH_DEADLINE=${DEADLINE}
ENIGMA_BENCH_SIZE=${SIZE}
ENIGMA_BENCH_EFFORT=${ENIGMA_BENCH_EFFORT:-}
ENIGMA_BENCH_MAX_EVALUATIONS=${ENIGMA_BENCH_MAX_EVALUATIONS:-100}
ENIGMA_BENCH_EVAL_SEED=${ENIGMA_BENCH_EVAL_SEED:-20260820}
ENIGMA_BENCH_PREVIEW=${ENIGMA_BENCH_PREVIEW:-1}
ENVEOF
chmod 0644 "$HOME_DIR/env"
export ENIGMA_BENCH_DEADLINE="$DEADLINE"
# The model size has to reach the agent's own training processes as well as the
# scorer, which reads it back from the file above after sudo scrubs the
# environment. Both halves of the run must agree or no checkpoint will load.
export ENIGMA_BENCH_SIZE="$SIZE"
log "budget: ${HOURS}h, ending at ${DEADLINE}"
log "model size: ${SIZE}"

# Mounted volumes come up owned by root; hand them to the right user.
install -d -o bench -g bench -m 0700 "$HOME_DIR/runs"
install -d -o agent -g bench -m 2750 "$HOME_DIR/submissions"
install -d -o agent -g agent -m 0755 "$HOME_DIR/out"
install -d -o agent -g agent -m 0755 /home/agent/.codex /home/agent/.claude

# The named volumes exist to retain login tokens, but both CLIs also write
# conversations, memories, prompt history and other entrant-controlled state to
# these directories. Start each benchmark run from clean agent state while
# preserving the two credential files. Shell mode remains persistent so it can
# be used to authenticate; the next run cleans anything else it left behind.
if [[ "$MODE" == "run" ]]; then
    /opt/enigma-bench/bin/reset-agent-state.sh \
        /home/agent/.codex /home/agent/.claude
fi

# A host-side command can create the bind-mounted log as root before the
# container starts. Repair its ownership without truncating an existing audit
# trail; the scorer must be able to append as `bench` (UID 2000).
RUN_LOG="$HOME_DIR/runs/runs.jsonl"
create_run_log() {
    runuser -u bench -- touch "$RUN_LOG"
    runuser -u bench -- chmod 0640 "$RUN_LOG"
}

# One `run` container = one benchmark run: it stamps a fresh deadline above, so
# it also starts a fresh run log. Archive a previous run's log instead of
# appending to it — the evaluation quota, the elapsed-since-first-submission
# clock and the hash chain are all derived from the log and must be per-run.
# `shell` and `score` containers operate on the current run and never rotate.
if [[ "$MODE" == "run" && -s "$RUN_LOG" ]]; then
    stamp="$(date -u -r "$RUN_LOG" +%Y%m%dT%H%M%SZ 2>/dev/null || date -u +%Y%m%dT%H%M%SZ)"
    archive="$HOME_DIR/runs/runs-${stamp}.jsonl"
    while [[ -e "$archive" ]]; do archive="${archive%.jsonl}-1.jsonl"; done
    if mv "$RUN_LOG" "$archive"; then
        log "archived previous run log to ${archive}"
    else
        log "could not archive the previous run log at ${RUN_LOG}"
        log "fix ownership/ACLs on the host or move it aside before retrying"
        exit 73
    fi
fi
if [[ ! -e "$RUN_LOG" ]]; then
    create_run_log
elif runuser -u bench -- test -w "$RUN_LOG"; then
    # Rootless Docker may not allow chown, but an already-writable file is
    # perfectly usable and should be left alone.
    true
elif chown bench:bench "$RUN_LOG" 2>/dev/null; then
    chmod 0640 "$RUN_LOG"
elif [[ ! -s "$RUN_LOG" ]]; then
    # A host-side watcher may have created an empty file with the wrong owner.
    # Removing only an empty file avoids losing an existing audit trail, then
    # lets the bench user create the replacement under either Docker mode.
    rm -f "$RUN_LOG"
    create_run_log
else
    log "run log is not writable by bench and cannot be repaired in this container"
    log "fix ownership/ACLs on the host or remove only the empty log before retrying"
    exit 73
fi

# A named volume hides the image copy, so seed the benchmark config after a run
# reset (or on the first shell) without overwriting Codex credentials.
if [[ ! -e /home/agent/.codex/config.toml ]]; then
    cp /workspace/codex-config.toml /home/agent/.codex/config.toml
    chown agent:agent /home/agent/.codex/config.toml
    chmod 0644 /home/agent/.codex/config.toml
fi

# ---------------------------------------------------------------------------
# Egress lockdown. Requires --cap-add=NET_ADMIN --cap-add=NET_RAW; set
# ENIGMA_BENCH_FIREWALL=0 only when an outer network policy already applies.
# ---------------------------------------------------------------------------
if [[ "${ENIGMA_BENCH_FIREWALL:-1}" == "1" ]]; then
    /opt/enigma-bench/bin/init-firewall.sh
else
    log "WARNING firewall disabled; the task is not offline"
fi

case "$MODE" in
    shell)
        exec runuser -u agent -- /bin/bash -l
        ;;
    score)
        shift
        exec runuser -u bench -- /opt/enigma-bench/bin/bench-cli "$@"
        ;;
    run)
        log "starting agent '${ENIGMA_BENCH_AGENT:-claude}'"
        # A hard wall-clock stop, five minutes after the scored deadline, so a
        # wedged agent cannot hold the container open forever.
        hard_stop="$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 + 300 }')"
        exec timeout --signal=INT --kill-after=120 "$hard_stop" \
            runuser -u agent -- /opt/enigma-bench/bin/run-agent.sh
        ;;
    *)
        log "unknown mode '${MODE}'; expected run, shell or score"
        exit 64
        ;;
esac
