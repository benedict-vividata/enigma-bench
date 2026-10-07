#!/usr/bin/env python3
"""A working baseline trainer for Enigma Bench. Beat it.

This is deliberately plain: an even mix of all seven tiers, AdamW, cosine decay
with warmup, bf16 autocast, and a checkpoint every N steps. It converges on
``t0_full`` (apply a fully specified machine) and gets nowhere on the blind
tiers, which is exactly the gap you are being asked to close.

Things it does *not* do, in rough order of how much they are likely to matter:

* curriculum over tiers, or over ciphertext length
* auxiliary supervision on the header slots (recover the key, then the text) —
  see :func:`enigma_bench.tokenizer.header_labels`
* any thought about batch size, learning rate, warmup or weight decay
* torch.compile, fused optimisers, gradient accumulation, EMA of weights
* generating data on the GPU instead of in dataloader workers

Run it::

    uv run python task/starter/train.py --minutes 30 --out submissions/baseline.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from enigma_bench.checkpoint import save_checkpoint
from enigma_bench.corpus import Corpus, load_split
from enigma_bench.dataset import build_eval_set, build_example
from enigma_bench.model import EnigmaEncoder, cross_entropy
from enigma_bench.spec import HEADER_LEN, IGNORE_INDEX
from enigma_bench.tiers import TIERS


class EnigmaStream(IterableDataset):
    """Endless stream of examples, generated on the fly in dataloader workers.

    Enciphering is pure Python at roughly a million letters a second per worker,
    so a handful of workers keeps a 4090 fed. If the GPU starves, either raise
    ``--workers`` or precompute batches.
    """

    def __init__(self, corpus: Corpus, seed: int) -> None:
        self.corpus = corpus
        self.seed = seed

    def __iter__(self):  # noqa: ANN204 - torch's IterableDataset protocol
        worker = get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0) * 10_007)
        weights = [tier.weight for tier in TIERS]
        while True:
            tier = rng.choices(TIERS, weights=weights, k=1)[0]
            example = build_example(tier, self.corpus, rng)
            yield (
                torch.tensor(example.input_ids, dtype=torch.long),
                torch.tensor(example.labels, dtype=torch.long),
            )


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


@torch.inference_mode()
def validation_accuracy(
    model: torch.nn.Module, corpus: Corpus, device: str, samples: int = 64
) -> dict[str, float]:
    """Per-tier letter accuracy on the agent-visible ``valid`` split.

    Use this to iterate. The scored ``evaluate_checkpoint`` tool uses a corpus
    split you cannot read, so treat this as a proxy, not as the answer.
    """
    model.eval()
    accuracies: dict[str, float] = {}
    for tier in TIERS:
        examples = build_eval_set(tier, corpus, samples, base_seed=999)
        inputs = torch.tensor([e.input_ids for e in examples], dtype=torch.long).to(device)
        labels = torch.tensor([e.labels for e in examples], dtype=torch.long).to(device)
        predicted = model(inputs).argmax(dim=-1)
        scored = labels != IGNORE_INDEX
        accuracies[tier.id] = float(((predicted == labels) & scored).sum() / scored.sum())
    model.train()
    return accuracies


def main(argv: list[str] | None = None) -> int:
    """Train until the time budget runs out, checkpointing as we go."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=30.0, help="wall-clock training budget")
    parser.add_argument("--out", default="submissions/baseline.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200_000, help="upper bound for the schedule")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    train_corpus = load_split("train")
    valid_corpus = load_split("valid")
    print(f"device={device} train={len(train_corpus):,} letters valid={len(valid_corpus):,}")

    model = EnigmaEncoder(dropout=args.dropout).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    loader = DataLoader(
        EnigmaStream(train_corpus, seed=args.seed),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
    )

    out = Path(args.out)
    deadline = time.monotonic() + args.minutes * 60
    running = 0.0
    step = 0

    model.train()
    for inputs, labels in loader:
        if time.monotonic() > deadline or step >= args.steps:
            break
        for group in optimiser.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)

        batch_inputs = inputs.to(device, non_blocking=True)
        batch_labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = cross_entropy(model(batch_inputs), batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        running += float(loss.detach())
        step += 1
        if step % args.log_every == 0:
            remaining = (deadline - time.monotonic()) / 60
            print(
                f"step {step:>7} loss {running / args.log_every:.4f} "
                f"lr {optimiser.param_groups[0]['lr']:.2e} {remaining:5.1f} min left",
                flush=True,
            )
            running = 0.0
        if step % args.eval_every == 0:
            accuracies = validation_accuracy(model, valid_corpus, device)
            print("  valid " + "  ".join(f"{k}={v:.3f}" for k, v in accuracies.items()), flush=True)
            save_checkpoint(model, out, metadata={"step": step, "valid_accuracy": accuracies})

    save_checkpoint(model, out, metadata={"step": step, "note": "final"})
    print(f"trained {step} steps; wrote {out}")
    print(f"header slots are positions 0..{HEADER_LEN - 1}; scored positions start at {HEADER_LEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
