# The benchmark container

An offline Ubuntu 24.04 container with an NVIDIA GPU passed through, a coding
agent inside it, and one narrow channel out: the `enigma-bench` MCP server.

## What is isolated from what

```
  host                                container (cap_add NET_ADMIN,NET_RAW)
 ┌──────────────┐        ┌──────────────────────────────────────────────────┐
 │ ./out/runs   │◀──────▶│ /srv/enigma-bench/runs        bench:bench 0700    │
 │ ./out/subs   │◀──────▶│ /srv/enigma-bench/submissions agent:bench 2750    │
 │ ./out/trans  │◀──────▶│ /srv/enigma-bench/out                             │
 └──────────────┘        │                                                  │
                         │ /srv/enigma-bench/corpus/test.txt bench:bench 0600│
                         │                                                  │
                         │  agent ──sudo -u bench──▶ enigma-bench-mcp        │
                         │    │                        (scores, logs)        │
                         │    └── /workspace, GPU, torch                     │
                         │                                                  │
                         │  egress: DROP, except the model-provider APIs     │
                         └──────────────────────────────────────────────────┘
```

Three properties, in decreasing order of how firmly they hold:

1. **The task is offline.** `init-firewall.sh` sets the OUTPUT policy to DROP
   and allows only the addresses in `allowed-domains.txt` — the Anthropic and
   OpenAI endpoints the agent CLIs need. No package index, no model hub, no
   general web. The agent has to _train_ a model, not fetch one.
2. **The held-out corpus is unreadable.** `test.txt` is mode 0600 and owned by
   `bench`; the agent runs as `agent`. The scorer runs as `bench` through one
   root-owned wrapper the agent may `sudo` to, and that wrapper speaks MCP and
   nothing else.
3. **The run log is tamper-evident.** It lives in a `bench`-only directory and
   every record carries the hash of its predecessor
   (`enigma-bench runs --verify`). The definitive copy is the one on the host
   bind mount, written as the run proceeds.

None of this survives a container escape, and a `--dangerously-skip-permissions`
agent with a kernel exploit is out of scope. It is designed to keep an honest
benchmark honest, not to contain an adversary.

## Prerequisites

- x86_64 Linux host with an NVIDIA GPU (developed against a 4090) and a recent
  driver.
- Docker Engine with the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed and configured.
- ~15 GB of disk for the image (PyTorch's CUDA wheels dominate) plus the ~100 MiB
  baked corpus.

Check the GPU reaches a container before anything else:

```sh
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## Authentication: use a subscription, not an API key

Both CLIs bill against a subscription when you sign them in as a person, and
against usage when you hand them an API key. For a multi-hour benchmark run you
want the former.

### Claude Code

Anthropic supports exactly this case. On a machine with a browser, signed in to
a Pro, Max, Team or Enterprise plan:

```sh
claude setup-token
```

That opens the normal browser authorisation flow and prints a one-year OAuth
token. It is not saved anywhere — copy it into `.env` as
`CLAUDE_CODE_OAUTH_TOKEN`. The token authenticates with your subscription and
can only make model requests.

Alternatively, use the Compose-provided persistent `claude-home` volume and
authenticate inside the isolated container:

```sh
./bench.sh shell
claude                         # follow Claude Code's login flow
exit
```

Later `run` containers reuse the credentials from that named volume. At the
start of each run, the volume is stripped back to `.credentials.json`; Claude
transcripts, prompt history, memory, snapshots, caches, settings and other
state from earlier runs are removed. This is the recommended setup when you do
not want to bind-mount Claude credentials from the host.

The alternative is to mount only your host credential file, which Claude Code
stores at `~/.claude/.credentials.json` on Linux:

```yaml
# docker-compose.override.yml
services:
  bench:
    volumes:
      - ${HOME}/.claude/.credentials.json:/home/agent/.claude/.credentials.json
```

Do not bind-mount the whole host `~/.claude` directory here. A benchmark run
intentionally deletes every entry in its Claude config directory except
`.credentials.json` to prevent state contamination between entrants.

Anthropic's own dev container documentation warns against mounting host secrets
into a container running an agent with permissions skipped. The token is the
better default; it is scoped to model requests and is trivially revocable.

### Codex

Codex has no `setup-token` equivalent. The recommended setup uses the
Docker-managed `codex-home` volume, so no host credential file is mounted:

```sh
./bench.sh shell
codex login --device-auth
exit
```

The credentials persist across the temporary shell and run containers. At the
start of each run, the volume is stripped back to `auth.json`; Codex sessions,
history, memories, caches, configuration and other state from earlier runs are
removed, then the benchmark's MCP configuration is seeded afresh.

If you use Claude Code, authenticate once in the same shell with `claude` (or
copy its credentials into the persistent `claude-home` volume). The existing
`CLAUDE_CODE_OAUTH_TOKEN` environment variable remains supported.

Note that the firewall allowlist includes `auth.openai.com` and `chatgpt.com`
precisely so this flow works.

## Running

From `enigma-bench/`:

```sh
./bench.sh build                       # resolves Codex latest; rebuilds its layer only when it changes
./bench.sh run --agent codex --model <model> --effort high --hours 6
./bench.sh run-bg --agent claude --model claude-opus-5 --effort max --hours 6 \
  --prompt 'Try to improve t0 generalisation early.'
./bench.sh report                      # markdown summary of ./container/out/runs
```

Each `run` container starts a fresh run log: an existing
`out/runs/runs.jsonl` is archived to `runs-<UTC stamp>.jsonl` (stamped with
the old log's last write) before the agent starts. The evaluation quota, the
elapsed-since-first-submission clock and the hash chain are all derived from
the log, so they reset with it — `runs.jsonl` is always the current (or most
recent) run, and earlier runs stay on the host as their archives. Verify an
archive with `ENIGMA_BENCH_RUNLOG=<path> enigma-bench runs --verify`.

Each run also starts with clean Claude Code and Codex state. In their Docker
named volumes, only the credential file required for login is retained across
run boundaries (`.credentials.json` for Claude Code and `auth.json` for
Codex); all other CLI home contents are deleted before the selected agent
starts. State remains available within a run, so Codex turn continuation still
resumes the active session normally.

Codex runs automatically resume the same session after each clean turn
completion and send a continuation prompt. This prevents a completed
checkpoint or score from ending the benchmark early; the container's external
time limit remains the stop mechanism. A Codex error, interrupt, or timeout is
not retried.

`./bench.sh shell` gives an interactive container with the same isolation, which
is the right place to debug the harness or run `codex login --device-auth`.

## Verifying the isolation

```sh
# inside a `./bench.sh shell` container, as `agent`:
curl -sS --max-time 5 https://pypi.org        # must fail
curl -sS --max-time 5 https://api.anthropic.com/v1/models   # must return 401
cat /srv/enigma-bench/corpus/test.txt         # must be Permission denied
cat /srv/enigma-bench/runs/runs.jsonl         # must be Permission denied
sudo -n -u bench /opt/enigma-bench/bin/enigma-bench-mcp </dev/null  # must start
```

## Knobs

| Variable                       | Default        | Effect                                                                                              |
| ------------------------------ | -------------- | --------------------------------------------------------------------------------------------------- |
| `ENIGMA_BENCH_AGENT`           | `claude`       | `claude` or `codex`                                                                                 |
| `ENIGMA_BENCH_MODEL`           | `gpt-5.6-luna` | Codex model; `--model` on `bench.sh run` takes precedence                                           |
| `ENIGMA_BENCH_CODEX_MODEL`     | `gpt-5.6-luna` | Codex model, used when `ENIGMA_BENCH_AGENT=codex`                                                   |
| `ENIGMA_BENCH_EFFORT`          | unset          | Optional reasoning effort; `--effort` on `run`/`run-bg` takes precedence                            |
| `ENIGMA_BENCH_PROMPT`          | unset          | Optional text appended to the task brief; `--prompt` on `run`/`run-bg` takes precedence             |
| `ENIGMA_BENCH_HOURS`           | `6`            | Budget. The scorer refuses submissions after it; the container is killed 5 minutes later            |
| `ENIGMA_BENCH_SIZE`            | `large`        | Benchmark model capacity: `small`, `medium` or `large`; `--size` on `run`/`run-bg` takes precedence |
| `ENIGMA_BENCH_MAX_EVALUATIONS` | `100`          | Scored submissions; `0` for unlimited                                                               |
| `ENIGMA_BENCH_EVAL_SEED`       | `20260820`     | Randomise per run so two agents face independent draws                                              |
| `ENIGMA_BENCH_PREVIEW`         | `1`            | `0` keeps held-out plaintext out of tool responses entirely                                         |
| `ENIGMA_BENCH_FIREWALL`        | `1`            | `0` skips the egress lockdown — only if an outer policy already applies                             |
| `CORPUS_SOURCE`                | WikiText-103   | Build arg; `dir:/mnt/corpus` to bake your own                                                       |

The default build resolves `@openai/codex@latest` to its concrete npm version
and passes that version to Docker. This makes Docker invalidate the Codex CLI
install layer when a new latest release appears, while keeping ordinary rebuilds
cacheable. To build with a specific version or without the registry lookup, set
`CODEX_VERSION`, for example `CODEX_VERSION=0.153.4 ./bench.sh build`.
