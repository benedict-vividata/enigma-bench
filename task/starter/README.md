# Starter

`train.py` is a complete, working, deliberately unambitious trainer. It exists
so you can confirm the loop end to end in the first ten minutes:

```bash
python /workspace/starter/train.py --minutes 20 --out /srv/enigma-bench/submissions/baseline.pt
```

Then `validate_checkpoint` it, and `evaluate_checkpoint` it once if you want the
floor on the board.

Read its module docstring: it lists, in rough order of expected impact, the
things it does not do. That list is not a plan, but it is a decent map of the
space you are being asked to search.
