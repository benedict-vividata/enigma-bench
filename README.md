# Enigma Bench

A time-boxed benchmark for coding agents, built around a task that cannot be
looked up: **train a neural network to break Enigma**.

An agent — Claude Code or Codex, on your subscription — is dropped into an
offline Ubuntu container with a GPU and _N_ hours. It must train a
fixed-architecture transformer encoder to recover plaintext from Enigma
ciphertext, and submit weights to a scoring tool. Everything it does is logged.

## Why this task

Most agent benchmarks measure whether a model can write code that passes tests.
This one measures whether it can _do research_: form a hypothesis about a hard
sequence-modelling problem, build a data pipeline, run experiments against a
clock, read the results, and change its mind.

It has properties that are hard to get elsewhere:

- **Unbounded headroom.** Nobody has a reference solution to memorise. Chance is
  3.85%; the ceiling is 100%; the seven difficulty tiers spread real models
  across the range instead of bunching them at pass/fail.
- **No shortcuts.** The container is offline. There is no pretrained checkpoint
  to fetch and no answer key to read — the held-out plaintext lives under a
  different Unix user than the agent.
- **A real time/compute tradeoff.** Data generation, batch size, curriculum and
  when to stop training all matter, and the agent has to reason about them
  against a deadline it can query.
- **Graded, not binary.** Tier 0 ("here is the whole key, apply the machine")
  is reachable in an hour. Tier 6 ("220 letters, ten-pair plugboard, nothing
  else") is an open problem. The score says _where on that ladder_ a model got.

The idea and the crude first implementation come from an earlier set of
experiments training encoders on Enigma; this repository is the cleaned-up,
scorable version.

## Quick start

```sh
./bench.sh test                            # lint + the unit tests; no GPU needed
./bench.sh build                           # image: CUDA wheels + baked corpus
./bench.sh run --agent codex --model <model> --effort high --hours 6
./bench.sh run-bg --agent claude --model claude-opus-5 --effort max --hours 6 \
  --prompt 'Try to improve t0 generalisation early.'
./bench.sh run --size medium               # a smaller benchmark model (default: large)
./bench.sh report --out report.md          # what happened
./watch-run.sh                             # live run dashboard
```

`build` and `run` need Docker with the NVIDIA Container Toolkit on an x86_64
host with an NVIDIA GPU (developed against a 4090). Authentication uses your
Claude or ChatGPT **subscription**, not an API key — see
[`container/README.md`](container/README.md) for the token and credential-mount
options.

Nothing but `uv` is needed for `./bench.sh test`, and everything except the
container runs on CPU, so the harness itself is developable on a laptop.

### Authentication quickstart

The Compose setup keeps Codex and Claude credentials in Docker-managed named
volumes rather than bind-mounting credential files from the host:

```sh
./bench.sh shell
codex login --device-auth       # Codex
exit
```

The credentials persist across `shell` and `run` containers, but chats,
history, memories, configuration and other agent state do not cross run
boundaries. Each `run` strips both named volumes back to their credential file
before starting the selected agent. Claude’s `CLAUDE_CODE_OAUTH_TOKEN`
environment variable remains supported as an alternative. For Claude Code,
authenticate in the same persistent shell:

```sh
./bench.sh shell
claude                         # follow Claude Code's login flow
exit
```

While a run is active, `./watch-run.sh transcript` follows the newest agent
transcript, `./watch-run.sh runs` follows scored evaluations, and the default
dashboard combines transcript, container, and GPU status. Press `Ctrl-C` to
close a view without stopping the benchmark. The dashboard refreshes once per
second by default; set `ENIGMA_BENCH_WATCH_INTERVAL=2` for a slower view.

For a detached benchmark run, use `./bench.sh run-bg ...`; its PID and
console output are saved under `out/`. Use `./watch-run.sh status` to inspect
it, `./watch-run.sh log` to follow the launcher log, and `./watch-run.sh stop`
to stop it.

Use `--effort` to set the model’s reasoning effort explicitly. For example,
`--effort max` requests the deepest supported reasoning; omit it to use the
agent/model default.

Use `--prompt TEXT` to append an operator-supplied instruction to the task
brief when the agent starts. Quote the value when it contains spaces; multiline
values are supported. The option works with both `run` and `run-bg` and is
passed to Claude Code or Codex alongside the standard task brief.

## The task

### Input

Every example is 256 tokens: a 36-slot header describing the machine, then 220
letters of ciphertext.

```
idx      field                            vocabulary
------   ------------------------------   ---------------------------
0        reflector                        <ukw-B> | <ukw-C> | <mask>
1..3     wheel order (left, mid, right)   <rotor-I..VIII> | <mask>
4..6     ring settings (Ringstellung)     a..z | <mask>
7..9     ground setting (Grundstellung)   a..z | <mask>
10..35   plugboard image of a..z          a..z | <mask>
36..255  ciphertext                       a..z
```

The model emits 26 logits (`a`..`z`) at every position. Only positions 36..255
are scored, against the plaintext that produced the ciphertext. Slots 4..35 are
letters too, so a training script may supervise them as an auxiliary
"recover the key" objective — the benchmark never scores them.

### The ladder

Each tier draws a fresh random machine setup per example, so nothing can be
memorised; what changes is how much of the setup the header reveals.

| tier                 | reflector | wheels | rings | ground | plugboard | shown |
| -------------------- | --------- | ------ | ----- | ------ | --------- | ----- |
| `t0_full`            | ✓         | ✓      | ✓     | ✓      | none      | —     |
| `t1_ground_one`      | ✓         | ✓      | ✓     | 1/3    | none      | —     |
| `t2_ground_two`      | ✓         | ✓      | ✓     | 1/3    | none      | —     |
| `t3_ground`          | ✓         | ✓      | ✓     | ✗      | none      | —     |
| `t4_rings`           | ✓         | ✓      | ✗     | ✗      | none      | —     |
| `t5_wheel_order`     | ✗         | ✗      | ✗     | ✗      | none      | —     |
| `t6_plugboard_given` | ✗         | ✗      | ✗     | ✗      | 10 pairs  | all   |
| `t7_plugboard_half`  | ✗         | ✗      | ✗     | ✗      | 10 pairs  | ~half |
| `t8_blind`           | ✗         | ✗      | ✗     | ✗      | 10 pairs  | none  |

The `t1_ground_one` tier masks one of the left, middle or right ground letters,
with each choice sampled equally. `t2_ground_two` masks one of the three pairs
of ground letters equally. `t0` asks only "can you _be_ an Enigma" — ring
offsets, the stepping schedule and the double-step anomaly, learned from data.
`t8` is the historical problem:
about 1.6 × 10²⁰ keys, from 220 letters.

### Scoring

- **Letter accuracy** on the 220 scored positions, per tier. Chance is 1/26.
- **Score** — weighted mean letter accuracy across the ladder, rescaled so the
  naive baseline (always guess the corpus's most common letter, ~12% on
  English) scores 0 and a perfect model scores 100. The unrescaled weighted
  mean ×100 is reported alongside as `raw_score`.
- **Grade** — the highest tier cleared _in order_ at 90% raw letter accuracy,
  or `none`. This is the number worth quoting.

Also reported per tier: exact-match rate, mean loss, and accuracy bucketed by
position in the message (early positions behave differently — the rotors have
stepped less).

### The model

A pre-norm encoder stack: GELU MLPs, learned absolute positions, **no causal
mask**, a 26-way head. Defined in `enigma_bench.model` and pinned by
`enigma_bench.spec.parameter_shapes()`, whose SHA-256 fingerprint goes into
every logged run.

Its capacity is the operator's one architectural choice, made before the run
starts with `./bench.sh run --size` and fixed for the run's whole life:

| size     | `d_model` | layers | heads | `d_ff` | aspect | parameters |
| -------- | --------: | -----: | ----: | -----: | -----: | ---------: |
| `small`  |       256 |      4 |     4 |   1024 |     64 |  3,237,914 |
| `medium` |       384 |      5 |     6 |   1536 |   76.8 |  8,989,082 |
| `large`  |       512 |      6 |     8 |   2048 |   85.3 | 19,067,930 |

`large` is the default and the original benchmark model, so every run logged
before sizes existed is comparable with a `large` run today — its fingerprint
is unchanged. The ladder follows the ordinary conventions rather than anything
tuned: 64 channels per attention head, an MLP four times the model width, and a
width-to-depth ratio inside the 64–128 band published models cluster in. Only
width and depth vary, together; the task, the tokenizer, the tiers and the
256-token sequence layout are identical at all three, so scores across sizes
answer "how much capacity does this recipe need?" rather than "which benchmark
is this?".

Print the ladder with `./bench.sh sizes`. A checkpoint only loads at the size it
was trained under — the shapes differ, and the scorer says so by name.

Everything about _training_ is the agent's: mixture, curriculum, auxiliary
losses, optimiser, schedule, precision, augmentation, how to keep the GPU fed.

## How a run is kept honest

| Concern                | Mechanism                                                                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Downloading a solution | Container egress is default-DROP with an allowlist of just the two model-provider APIs                                                   |
| Reading the answer key | Held-out `test.txt` is mode 0600 under a different user; the scorer runs as that user through one root-owned MCP wrapper                 |
| Editing the score      | The run log lives in a directory the agent cannot read, and each record carries the hash of its predecessor                              |
| Overfitting the scorer | Scored submissions are quota-limited; the eval seed is per-run; `ENIGMA_BENCH_PREVIEW=0` removes held-out text from tool output entirely |
| Running over time      | The scorer refuses submissions past the deadline and logs the attempt; the container is killed 5 minutes later                           |

[`container/README.md`](container/README.md) documents each of these, including
what they do _not_ protect against.

## Layout

```
enigma-bench/
├── bench.sh                  one entry point for everything
├── src/enigma_bench/
│   ├── machine.py            the Enigma — dependency-free, readable, exact
│   ├── spec.py               the frozen architecture and sequence layout
│   ├── tokenizer.py          header + ciphertext -> token ids
│   ├── tiers.py              the difficulty ladder
│   ├── dataset.py            deterministic example construction
│   ├── model.py              the fixed encoder (torch)
│   ├── checkpoint.py         submission format and validation
│   ├── evaluate.py           the scorer
│   ├── runlog.py             append-only, hash-chained audit trail
│   ├── harness.py            deadline, quota, logging — shared by CLI and MCP
│   ├── mcp_server.py         the six tools the agent sees
│   └── cli.py                the operator's command line
├── container/                Dockerfile, compose, egress firewall, isolation
├── task/                     what the agent is given: TASK.md, MCP config, starter
├── scripts/fetch_corpus.py   builds the ~100 MiB corpus at image-build time
└── tests/                    126 tests, all CPU, all fast
```

## The corpus

~98 MiB of cleaned English from [WikiText-103
(raw)](https://huggingface.co/datasets/Salesforce/wikitext), lowercased and
stripped to `a-z` — which is what a wartime clerk typed. Its train / validation
/ test splits are article-disjoint, so the scorer's plaintext is genuinely
unseen rather than a neighbouring slice of the training stream.

`train.txt` and `valid.txt` are the agent's; `test.txt` is the scorer's alone.
Swap the source with `CORPUS_SOURCE=dir:/mnt/whatever` at build time.

## Analysing a run

The run log at `out/runs/runs.jsonl` is one JSON object per submission —
score, per-tier breakdown, position curve, checkpoint hash, model size,
elapsed time since the first submission, and the label and notes the agent
attached. The log is
per-run: starting a new `run` archives the previous log to
`out/runs/runs-<UTC stamp>.jsonl`, so `runs.jsonl` always describes the
current (or most recent) run and its quota and elapsed-time figures never mix
sessions.

```sh
./bench.sh runs                       # list, and verify the hash chain
./bench.sh report --out report.md     # best submission + full history
jq -c 'select(.status == "ok") | {t: .elapsed_since_first_run_seconds, s: .score}' \
    out/runs/runs.jsonl               # the learning curve of the *agent*
```

Comparing agents is a matter of comparing those curves: not only the final
score, but how many submissions it took, how long until the first working one,
and whether it kept improving or plateaued.

## Development

```sh
uv sync                  # torch resolves to CUDA 12.9 on x86_64 Linux, CPU elsewhere
uv run pytest            # 148 tests, ~23s on CPU
uv run ruff check .
uv run enigma-bench describe
uv run enigma-bench sizes       # the capacity ladder, with parameter counts
uv run enigma-bench selftest    # the Enigma against published test vectors
```

The harness is torch-free below `model.py`, so the machine, tokenizer, tiers,
dataset and run log can be exercised without a GPU stack at all.
