#!/usr/bin/env bash
# Live views for an Enigma Bench run.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${ENIGMA_BENCH_WATCH_INTERVAL:-1}"
INTERVAL_MS=$((INTERVAL * 1000))
OUT_DIR="${ENIGMA_BENCH_OUT:-out}"
TRANSCRIPT_DIR="$OUT_DIR/transcripts"
RUN_LOG="$OUT_DIR/runs/runs.jsonl"
RUN_PID_FILE="$OUT_DIR/bench-run.pid"
RUN_STDOUT_LOG="$OUT_DIR/bench-run.log"

if [[ -t 1 ]]; then
    export WATCH_COLOR=1
    BLUE=$'\033[34;1m'
    CYAN=$'\033[36;1m'
    GREEN=$'\033[32;1m'
    MAGENTA=$'\033[35;1m'
    YELLOW=$'\033[33;1m'
    RESET=$'\033[0m'
else
    export WATCH_COLOR=0
    BLUE=''; CYAN=''; GREEN=''; MAGENTA=''; YELLOW=''; RESET=''
fi

usage() {
    cat <<'EOF'
Usage: ./watch-run.sh [view]

Views:
  dashboard    Refreshing overview of the run (default)
  transcript   Follow the newest Codex/Claude transcript
  runs         Follow scored evaluations
  container    Show container and GPU status once
  status       Show detached-run, container, transcript, and log status
  log          Follow the detached benchmark log
  stop         Stop the detached benchmark and its active container

Environment:
  ENIGMA_BENCH_OUT             Benchmark output directory (default: out)
  ENIGMA_BENCH_WATCH_INTERVAL Dashboard refresh interval in seconds (default: 1)

Examples:
  ./watch-run.sh
  ./watch-run.sh transcript
  ./bench.sh run-bg --agent codex --model gpt-5.6-luna --hours 1
  ./watch-run.sh status
  ENIGMA_BENCH_WATCH_INTERVAL=5 ./watch-run.sh dashboard
EOF
}

latest_transcript() {
    find "$TRANSCRIPT_DIR" -maxdepth 1 -type f \
        \( -name 'codex-*.jsonl' -o -name 'claude-*.jsonl' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed 's/^[^ ]* //' | head -1
}

latest_container() {
    docker ps --filter name=enigma-bench-bench-run --format '{{.ID}} {{.Names}}' 2>/dev/null | head -1
}

run_pid() {
    [[ -r "$RUN_PID_FILE" ]] || return 1
    local pid
    pid="$(<"$RUN_PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    ps -p "$pid" -o args= 2>/dev/null | grep -q '[b]ench.sh run'
}

show_status() {
    local pid container transcript
    printf '%sDETACHED BENCHMARK%s\n' "$BLUE" "$RESET"
    if run_pid; then
        pid="$(<"$RUN_PID_FILE")"
        printf 'running (pid %s)\n' "$pid"
        ps -p "$pid" -o pid=,etime=,args= 2>/dev/null || true
    else
        printf 'not running\n'
    fi
    printf 'log: %s%s\n\n' "$RUN_STDOUT_LOG" "$( [[ -f "$RUN_STDOUT_LOG" ]] && printf ' (present)' || true )"

    printf '%sCONTAINER%s\n' "$BLUE" "$RESET"
    container="$(latest_container)"
    printf '%s\n\n' "${container:-none}"

    printf '%sLATEST TRANSCRIPT%s\n' "$MAGENTA" "$RESET"
    transcript="$(latest_transcript)"
    printf '%s\n\n' "${transcript:-none}"

    printf '%sLATEST EVALUATION LOG%s\n%s\n' "$YELLOW" "$RESET" \
        "$RUN_LOG"
}

follow_run_log() {
    mkdir -p "$OUT_DIR"
    touch "$RUN_STDOUT_LOG"
    printf 'Following %s (Ctrl-C to stop)\n\n' "$RUN_STDOUT_LOG"
    tail -n 30 -F "$RUN_STDOUT_LOG"
}

stop_background() {
    local pid container
    if run_pid; then
        pid="$(<"$RUN_PID_FILE")"
        printf 'Stopping benchmark process %s...\n' "$pid"
        kill "$pid" 2>/dev/null || true
    else
        printf 'No detached benchmark process found.\n'
    fi
    container="$(latest_container)"
    if [[ -n "$container" ]]; then
        printf 'Stopping container %s...\n' "$container"
        docker stop "${container%% *}" >/dev/null 2>&1 || true
    fi
    rm -f "$RUN_PID_FILE"
}

pretty_file_tail() {
    local file="$1" lines="${2:-8}"
    [[ -f "$file" ]] || { printf '  waiting for %s\n' "$file"; return; }
    if command -v jq >/dev/null 2>&1; then
        tail -n "$lines" "$file" | format_events || tail -n "$lines" "$file"
    else
        tail -n "$lines" "$file"
    fi
}

# Turn Codex's JSONL event stream into short, human-readable blocks. Unknown
# event types are deliberately hidden; they tend to be protocol bookkeeping.
format_events() {
    jq --unbuffered -r '
        def color($code; $text):
          if ($ENV.WATCH_COLOR // "0") == "1" then
            "\u001b[" + $code + "m" + $text + "\u001b[0m"
          else $text end;
        def clip($n):
          if . == null then ""
          else (tostring | split("\n") |
            if length > $n then .[0:$n] + ["…"] else . end |
            join("\n"))
          end;
        def block($label; $body): "\n" + $label + "\n" + $body;
        if .type == "item.completed" and .item.type == "agent_message" then
          block((color("32;1"; "▶ AGENT")); (.item.text | clip(30)))
        elif .type == "item.started" and .item.type == "command_execution" then
          block((color("33;1"; "⌘ RUNNING")); ("$ " + (.item.command | clip(4))) )
        elif .type == "item.completed" and .item.type == "command_execution" then
          block((color("34;1"; "⌘ COMMAND"));
            ("$ " + (.item.command | clip(4)) +
             (if (.item.aggregated_output // "") != "" then
                "\n" + (.item.aggregated_output | clip(12))
              else "" end)))
        elif .type == "item.completed" and .item.type == "mcp_tool_call" then
          block((color("35;1"; "⚙ TOOL"));
            ((.item.tool // "unknown") + " [" + (.item.status // "completed") + "]" +
             (if .item.result.structured_content then
                "\n" + (.item.result.structured_content | tojson | clip(18))
              elif .item.result.content[0].text then
                "\n" + (.item.result.content[0].text | clip(18))
              else "" end)))
        elif .type == "turn.started" then
          color("36;1"; "\n── TURN STARTED ──")
        elif .type == "turn.completed" then
          color("36;1"; "\n── TURN COMPLETED ──")
        else empty end
    ' 2>/dev/null
}

show_container() {
    local container
    container="$(latest_container)"
    if [[ -z "$container" ]]; then
        printf 'Container: no active enigma-bench run\n'
        return
    fi

    printf 'Container: %s\n\n' "$container"
    docker stats --no-stream --format \
        'CPU {{.CPUPerc}} | memory {{.MemUsage}} ({{.MemPerc}}) | net {{.NetIO}} | pids {{.PIDs}}' \
        "${container%% *}" 2>/dev/null || true
    docker exec "${container%% *}" ps -eo pid,etime,%cpu,%mem,cmd --sort=-%cpu 2>/dev/null | head -8 || true

    if command -v nvidia-smi >/dev/null 2>&1; then
        printf '\nGPU:\n'
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader 2>/dev/null || true
    fi
}

view_transcript() {
    local file
    file="$(latest_transcript)"
    if [[ -z "$file" ]]; then
        printf 'No transcript found yet under %s\n' "$TRANSCRIPT_DIR" >&2
        exit 1
    fi
    printf 'Following %s (Ctrl-C to stop)\n\n' "$file"
    if command -v jq >/dev/null 2>&1; then
        tail -n 20 -F "$file" | format_events | fold -s -w "${COLUMNS:-110}"
    else
        tail -n 20 -F "$file"
    fi
}

view_runs() {
    printf 'Following %s (Ctrl-C to stop)\n\n' "$RUN_LOG"
    mkdir -p "$(dirname "$RUN_LOG")"
    if command -v jq >/dev/null 2>&1; then
        tail -n 0 -F "$RUN_LOG" | jq --unbuffered -C . 2>/dev/null
    else
        tail -n 20 -F "$RUN_LOG"
    fi
}

render_dashboard() {
    local transcript container
    transcript="$(latest_transcript)"
    container="$(latest_container)"

    printf '%sEnigma Bench dashboard%s — %s\n' "$CYAN" "$RESET" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf 'Output: %s    Refresh: %ss\n' "$OUT_DIR" "$INTERVAL"
    printf 'Press Ctrl-C to stop.\n\n'

    if [[ -n "$container" ]]; then
        printf '%sACTIVE CONTAINER%s\n%s\n\n' "$BLUE" "$RESET" "$container"
        docker stats --no-stream --format \
            'CPU {{.CPUPerc}} | memory {{.MemUsage}} ({{.MemPerc}}) | net {{.NetIO}} | pids {{.PIDs}}' \
            "${container%% *}" 2>/dev/null || true
    else
        printf '%sACTIVE CONTAINER%s\nnone\n' "$BLUE" "$RESET"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        printf '\n%sGPU%s\n' "$GREEN" "$RESET"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader 2>/dev/null || true
    fi

    printf '\n%sLATEST TRANSCRIPT%s: %s\n' "$MAGENTA" "$RESET" "${transcript:-waiting}"
    [[ -n "$transcript" ]] && pretty_file_tail "$transcript" 5 || true

    printf '\n%sLATEST EVALUATIONS%s\n' "$YELLOW" "$RESET"
    if [[ -f "$RUN_LOG" ]]; then
        pretty_file_tail "$RUN_LOG" 3
    else
        printf '  no evaluations logged yet\n'
    fi
}

view_dashboard() {
    local alt_screen=0
    local frame frame_started elapsed remaining
    if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
        tput smcup 2>/dev/null || true
        tput civis 2>/dev/null || true
        alt_screen=1
        restore_dashboard_terminal() {
            tput cnorm 2>/dev/null || true
            tput rmcup 2>/dev/null || true
        }
        stop_dashboard() {
            restore_dashboard_terminal
            exit 130
        }
        trap restore_dashboard_terminal EXIT
        trap stop_dashboard INT TERM
    fi
    while true; do
        frame_started="$(date +%s%3N)"
        # Build the complete frame before touching the terminal. The DEC
        # synchronized-update guard makes compatible terminals display the
        # refresh atomically instead of showing a blank intermediate frame.
        frame="$(render_dashboard)"
        if (( alt_screen )); then
            printf '\033[?2026h'
            tput cup 0 0 2>/dev/null || true
            printf '%s\n' "$frame"
            tput ed 2>/dev/null || true
            printf '\033[?2026l'
        else
            printf '\033[H\033[2J%s\n' "$frame"
        fi
        # Keep refreshes on a one-second cadence instead of sleeping after the
        # render and accidentally making each tick render-time + interval.
        elapsed=$(( $(date +%s%3N) - frame_started ))
        remaining=$((INTERVAL_MS - elapsed))
        if (( remaining > 0 )); then
            sleep "$(printf '%d.%03d' $((remaining / 1000)) $((remaining % 1000)))"
        fi
    done
}

case "${1:-dashboard}" in
    dashboard)  view_dashboard ;;
    transcript) view_transcript ;;
    runs)       view_runs ;;
    container)  show_container ;;
    status)     show_status ;;
    log|logs)   follow_run_log ;;
    stop)       stop_background ;;
    -h|--help|help) usage ;;
    *) printf 'Unknown view: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
esac
