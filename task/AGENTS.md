<!--
  This file is copied to /workspace/CLAUDE.md (and /workspace/AGENTS.md) inside
  the benchmark container: it is what the *entrant* reads, not guidance for
  anyone maintaining Enigma Bench itself. For that, see ../README.md.
-->

# Working in this container

You are an entrant in Enigma Bench. The task brief is `/workspace/TASK.md` —
read it first if it is not already in your context.

## Ground rules

- **The architecture is fixed.** `enigma_bench.model.EnigmaEncoder`, unmodified.
  Changing layer counts, widths or parameter names makes your checkpoint
  unloadable and wastes a submission. `validate_checkpoint` is free — use it.
- **Persistence is mandatory.** This is an optimization run, not a one-shot
  task. You must keep experimenting and trying to improve the best official
  score until the external harness stops you. Never voluntarily end the
  session after a checkpoint, score, experiment, or plateau; only external
  termination ends the run. The highest score obtained during the run matters,
  not necessarily the last checkpoint.
- **Use the dynamic clock.** Call `time_remaining` at startup and throughout
  the run. Use its deadline to plan the exploration/exploitation tradeoff,
  leaving time to validate and submit improvements. If the reported deadline
  passes, you must keep working if possible—the runner will stop you
  externally. Do not treat the deadline as permission to stop.
- **You are offline.** No package installs, no downloads, no external weights.
  If something is missing, work around it rather than trying to fetch it.
- **`/srv/enigma-bench/corpus/test.txt` is not yours to read**, and the run log
  is not yours to edit. Both are owned by another user; attempts are visible.
- **Scored submissions are finite.** Measure on `valid.txt` yourself; submit
  when you have a reason to believe the number moved.

## Where things are

| Path                                 | What                                               |
| ------------------------------------ | -------------------------------------------------- |
| `/workspace`                         | yours — code, notes, scratch checkpoints           |
| `/workspace/starter/train.py`        | a working baseline trainer to read and beat        |
| `/srv/enigma-bench/corpus`           | `train.txt`, `valid.txt` (yours), `test.txt` (not) |
| `/srv/enigma-bench/submissions`      | write checkpoints here to submit them              |
| `/opt/enigma-bench/src/enigma_bench` | the harness source — read it                       |

## Useful commands

```bash
enigma-bench describe                      # the frozen spec and the tier ladder
enigma-bench sample --tier t8_blind --split valid
enigma-bench selftest                      # sanity-check the Enigma itself
nvidia-smi                                 # what you are working with
```

## Habits that pay off here

- Keep a running `NOTES.md` of what you tried and what it scored. You will be
  interrupted by the deadline, not by finishing.
- Save a checkpoint on a timer, not only at the end.
- Watch GPU utilisation early. Data generation is pure Python; if the GPU is
  idle, fix the input pipeline before tuning anything else.
