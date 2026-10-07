# Enigma Bench

You have a GPU, an offline container, and a fixed amount of wall-clock time.
Train a neural network to break Enigma.

## Objective

Given Enigma **ciphertext** — and, depending on the difficulty tier, some, all
or none of the machine's key settings — predict the **plaintext**, letter by
letter.

You will be scored across seven tiers, from "the whole key is in the header,
just apply the machine" to "220 letters of ciphertext against a ten-pair
plugboard and nothing else". Chance is 1/26 = 3.85% letter accuracy.

## Persistence and the clock

This is a time-boxed optimization run, not a one-shot task. It is imperative
that you keep working persistently until the harness stops you externally. Do
**not** voluntarily end the session, stop the run, or return a final response
after producing a checkpoint, receiving a score, completing an experiment, or
seeing a plateau. The highest official score obtained during the run is the
result that matters, so keep trying to improve on the current best and submit
stronger checkpoints as you find them. The last checkpoint is not automatically
the best checkpoint.

The wall-clock limit is dynamic. Call `time_remaining` at the start and
periodically throughout the run; it reports the actual deadline and remaining
budget. Treat that deadline as your planning horizon: reserve time for
validation and submissions, save checkpoints before risky experiments, and
balance exploration of new ideas against exploitation of the best method so
far. If the reported time reaches zero, you must keep working if the harness
still allows it—the external runner will terminate the process. The deadline
is not a signal to stop, and you must not stop early on your own for any
reason; only external termination ends the run.

## What is fixed, and what is yours

**Fixed — do not change:**

- The model architecture. A bidirectional transformer encoder (**no causal
  mask**), defined in `enigma_bench.model.EnigmaEncoder`. Its width and depth
  were chosen by the operator before this run started — `describe_benchmark`
  reports the size and the exact parameter count, and
  `enigma_bench.spec.parameter_shapes()` pins every name and shape. The scorer
  rejects anything that does not match exactly. Build the model by
  instantiating `EnigmaEncoder()`; never hard-code its dimensions.
- The sequence layout: 256 tokens = 36 header slots + 220 ciphertext letters.
- The tokenizer and the tier definitions.

**Yours — this is the whole game:**

- Data mixture and curriculum. Which tiers to train on, in what proportion, and
  how that changes over the run.
- Objective. The scored loss is cross-entropy on the plaintext, but you may add
  auxiliary losses. In particular the header slots 4..35 (ring settings, ground
  setting, plugboard) are _letters_, so you can ask the model to recover the key
  from the ciphertext as well as decode it — see
  `enigma_bench.tokenizer.header_labels`.
- Optimiser, schedule, batch size, precision, initialisation, augmentation,
  `torch.compile`, EMA, how you generate data fast enough to keep the GPU busy.
- How you spend the clock: one long run, or many short ablations.

Despite the eval pinning a fixed architecture, your _research_ does not have to
stay inside it. If it would help your understanding, experiment freely with
different architectures, scales and training setups along the way — probe
models, simplified variants, throwaway networks at other sizes, whatever
teaches you something about the problem. Only the checkpoint you submit must
match the frozen architecture exactly.

## The environment

- No internet. Everything you need is installed; nothing can be downloaded.
- Python and PyTorch are on `PATH` from `/opt/enigma-bench/.venv`. `torch`,
  `numpy`, `tqdm`, `matplotlib` and `einops` are available. `enigma_bench` is
  importable.
- `/workspace` is yours: write code, checkpoints and notes there.
- Corpus at `/srv/enigma-bench/corpus`:
  - `train.txt` — ~96 MiB of cleaned English, lowercase `a-z`, no spaces.
  - `valid.txt` — held out from training, **readable by you**. Use it to
    measure your own accuracy as often as you like.
  - `test.txt` — the scorer's plaintext. Not readable by you.
- Submit checkpoints by writing them to `/srv/enigma-bench/submissions/`.

## The model contract

```python
from enigma_bench.checkpoint import save_checkpoint
from enigma_bench.model import EnigmaEncoder

model = EnigmaEncoder(dropout=0.0)  # dropout is free; nothing else is
save_checkpoint(model, Path("/srv/enigma-bench/submissions/best.pt"))
```

`save_checkpoint` writes the format the scorer expects. A bare
`torch.save(model.state_dict(), path)` also works, as do checkpoints straight
out of `torch.compile` or DDP (the `_orig_mod.`/`module.` prefixes are
stripped) and bf16/fp16 weights (they are upcast). The file is loaded with
`weights_only=True`, so it may contain tensors and primitives only.

## Scoring

Call the `enigma-bench` MCP tools:

| Tool                    | Cost               | What it does                                 |
| ----------------------- | ------------------ | -------------------------------------------- |
| `describe_benchmark`    | free               | The full spec: layout, tiers, scoring, paths |
| `time_remaining`        | free               | Wall clock and evaluation quota left         |
| `validate_checkpoint`   | free               | Does this file load into the architecture?   |
| `evaluate_checkpoint`   | **one submission** | Scores it and logs the result                |
| `list_runs` / `get_run` | free               | Your submission history                      |

Every `evaluate_checkpoint` call is recorded, whether it scores or fails. The
quota is finite (100 by default) and there is a deadline — check
`time_remaining` before you plan a long run and continue checking it as you
schedule experiments.

Iterate against `valid.txt` locally. Spend scored submissions on things you
actually believe in.

**Headline score** = weighted mean letter accuracy across the seven tiers,
rescaled so that the naive baseline — always guessing the corpus's most common
letter, roughly 12% on English — scores 0 and a perfect model scores 100.
Matching the plaintext's letter statistics earns nothing; only decoding does.
The unrescaled weighted mean accuracy x100 is reported alongside as
`raw_score`.
**Grade** = the highest tier you clear _in order_ at 90% raw letter accuracy.

## Suggested first hour

1. `describe_benchmark`, then read `/opt/enigma-bench/src/enigma_bench/machine.py`.
   Understand ring settings and the double-step anomaly before you train anything.
2. Run `enigma-bench sample --tier t0_full --split valid` and
   `--tier t8_blind` to see what the model actually sees.
3. Run the baseline: `python /workspace/starter/train.py --minutes 20`, then
   `validate_checkpoint` on its output. That proves your loop end to end.
4. Only then decide what to actually build.

`/workspace/starter/train.py` is a deliberately plain baseline. Its docstring
lists what it does not do; that list is a reasonable place to start.

Good luck.
