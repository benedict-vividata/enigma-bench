#!/usr/bin/env bash
#
# Starts the coding agent, as `agent`, with the task brief as its prompt.
#
# Both CLIs run fully autonomously: the container *is* the sandbox, so their own
# permission prompts and sandboxes are bypassed. Nothing here can escape the
# container's filesystem or its egress allowlist.
set -euo pipefail

AGENT="${ENIGMA_BENCH_AGENT:-claude}"
MODEL="${ENIGMA_BENCH_MODEL:-${ENIGMA_BENCH_CODEX_MODEL:-gpt-5.6-luna}}"
EFFORT="${ENIGMA_BENCH_EFFORT:-}"
HOME_DIR="${ENIGMA_BENCH_HOME:-/srv/enigma-bench}"
OUT_DIR="${HOME_DIR}/out"
TASK=/workspace/TASK.md
TASK_PROMPT="$(cat "$TASK")"
if [[ -n "${ENIGMA_BENCH_PROMPT:-}" ]]; then
    TASK_PROMPT+=$'\n\nAdditional operator prompt:\n'
    TASK_PROMPT+="${ENIGMA_BENCH_PROMPT}"
fi

log() { printf '[run-agent] %s\n' "$*" >&2; }

if [[ -r "${HOME_DIR}/env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${HOME_DIR}/env"
    set +a
fi

mkdir -p "$OUT_DIR"
STARTED="$(date -u +%Y%m%dT%H%M%SZ)"
TRANSCRIPT="${OUT_DIR}/${AGENT}-${STARTED}.jsonl"

case "$AGENT" in
    claude)
        if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" && ! -r "${CLAUDE_CONFIG_DIR}/.credentials.json" ]]; then
            log "no Claude credentials: set CLAUDE_CODE_OAUTH_TOKEN (from 'claude setup-token')"
            log "or mount a host ~/.claude at ${CLAUDE_CONFIG_DIR}. See container/README.md."
            exit 78
        fi
        # The prompt is positional, so it goes last.
        CLAUDE_ARGS=(--model "$MODEL")
        if [[ -n "$EFFORT" ]]; then
            CLAUDE_ARGS+=(--effort "$EFFORT")
        fi
        exec claude "${CLAUDE_ARGS[@]}" \
            --print \
            --dangerously-skip-permissions \
            --mcp-config /workspace/.mcp.json \
            --strict-mcp-config \
            --output-format stream-json \
            --verbose \
            "$TASK_PROMPT" \
            > "$TRANSCRIPT"
        ;;
    codex)
        if [[ ! -r "${CODEX_HOME}/auth.json" ]]; then
            log "no Codex credentials: mount a host ~/.codex/auth.json at ${CODEX_HOME}/auth.json"
            log "or run 'codex login --device-auth' in a 'shell' container. See container/README.md."
            exit 78
        fi

        # Codex can end a turn after submitting a checkpoint or score even
        # though the benchmark is still running. Keep the same session alive
        # by explicitly resuming it after each clean turn completion. This is
        # deliberately an in-process loop: the outer timeout remains the
        # authority that stops the benchmark.
        CODEX_ARGS=(
            --model "$MODEL"
            --dangerously-bypass-approvals-and-sandbox
            --skip-git-repo-check
            --json
        )
        if [[ -n "$EFFORT" ]]; then
            CODEX_ARGS+=(--config "model_reasoning_effort=\"${EFFORT}\"")
        fi
        KEEP_GOING_PROMPT='Keep on going. You are not finished. First inspect any active background jobs, checkpoints, or processes you started and continue or monitor them as appropriate; do not start redundant work. Keep running experiments, improving the official score, and submitting improvements until the external benchmark harness stops you. Do not end your turn voluntarily because you have a result, checkpoint, plateau, or apparent completion.'

        : > "$TRANSCRIPT"
        if codex exec "${CODEX_ARGS[@]}" --cd /workspace "$TASK_PROMPT" >> "$TRANSCRIPT"; then
            :
        else
            status=$?
            log "Codex exited before a clean turn completion (status ${status}); not resuming"
            exit "$status"
        fi

        THREAD_ID="$(jq -r 'select(.type == "thread.started") | .thread_id' "$TRANSCRIPT" | head -n 1)"
        if [[ -z "$THREAD_ID" || "$THREAD_ID" == "null" ]]; then
            log "could not identify the Codex session; not attempting continuation"
            exit 70
        fi

        while true; do
            log "Codex turn completed; sending continuation prompt"
            if codex exec resume "${CODEX_ARGS[@]}" "$THREAD_ID" "$KEEP_GOING_PROMPT" >> "$TRANSCRIPT"; then
                :
            else
                status=$?
                log "Codex continuation exited (status ${status}); stopping"
                exit "$status"
            fi
        done
        ;;
    *)
        log "unknown agent '${AGENT}'; expected 'claude' or 'codex'"
        exit 64
        ;;
esac
