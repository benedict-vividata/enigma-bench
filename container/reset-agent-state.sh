#!/usr/bin/env bash
#
# Remove coding-agent state that must not cross benchmark-run boundaries while
# retaining the subscription credentials stored in each agent's config home.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: reset-agent-state.sh CODEX_HOME CLAUDE_CONFIG_DIR\n' >&2
    exit 64
fi
readonly CODEX_DIR="$1"
readonly CLAUDE_DIR="$2"

log() { printf '[reset-agent-state] %s\n' "$*" >&2; }

validate_home() {
    local name="$1" path="$2"
    case "$path" in
        ""|/|/home|/home/agent)
            log "refusing unsafe ${name} home: '${path}'"
            exit 64
            ;;
    esac
    if [[ -L "$path" || ! -d "$path" ]]; then
        log "${name} home is not a directory: '${path}'"
        exit 72
    fi
}

remove_except() {
    local directory="$1" keep_name="$2" entry
    while IFS= read -r -d '' entry; do
        if [[ "${entry##*/}" != "$keep_name" ]]; then
            rm -rf -- "$entry"
        fi
    done < <(find "$directory" -mindepth 1 -maxdepth 1 -print0)
}

validate_home Codex "$CODEX_DIR"
validate_home Claude "$CLAUDE_DIR"

# auth.json and .credentials.json contain refreshable subscription tokens. All
# other entries can include transcripts, prompt history, memories, tool output,
# snapshots, caches, or mutable configuration from an earlier entrant.
remove_except "$CODEX_DIR" auth.json
remove_except "$CLAUDE_DIR" .credentials.json
