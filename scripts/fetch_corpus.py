#!/usr/bin/env python3
"""Build the English plaintext corpus that the benchmark trains and scores on.

Run this on the **build host**, which has internet; the benchmark container does
not. The output is three files of cleaned lowercase ``a-z`` plus a manifest:

    corpus/train.txt   ~96 MiB   the agent's training data
    corpus/valid.txt   ~1 MiB    the agent's held-out set
    corpus/test.txt    ~1 MiB    the eval's plaintext source — kept from the agent

The default source is WikiText-103 (raw), whose own train/validation/test splits
are article-disjoint, so ``test.txt`` is genuinely unseen text rather than a
neighbouring slice of the training stream.

Usage::

    uv run --extra corpus scripts/fetch_corpus.py --out corpus
    uv run --extra corpus scripts/fetch_corpus.py --source dir:/data/gutenberg --out corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

MIB = 1024 * 1024
LINE_WIDTH = 80
DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-103-raw-v1"
FETCH_VERSION = 1


def clean(text: str) -> str:
    """Reduce text to lowercase ``a-z``, dropping everything else."""
    return "".join(char for char in text.lower() if "a" <= char <= "z")


def wrap(letters: str, width: int = LINE_WIDTH) -> Iterator[str]:
    """Yield fixed-width lines so the corpus files stay greppable."""
    for start in range(0, len(letters), width):
        yield letters[start : start + width]


def _hf_split(dataset: str, config: str, split: str, limit_bytes: int | None) -> str:
    from datasets import load_dataset  # imported lazily: only the build host has it

    stream = load_dataset(dataset, config, split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for row in stream:
        text = row.get("text", "")
        if not text or text.lstrip().startswith("="):
            continue  # WikiText section headers, e.g. " = = History = = "
        cleaned = clean(text)
        if not cleaned:
            continue
        chunks.append(cleaned)
        total += len(cleaned)
        if limit_bytes is not None and total >= limit_bytes:
            break
    return "".join(chunks)[:limit_bytes] if limit_bytes else "".join(chunks)


def _dir_split(root: Path, split: str, limit_bytes: int | None) -> str:
    """Read ``<root>/<split>/**/*.txt`` — the offline escape hatch."""
    directory = root / split
    if not directory.is_dir():
        raise SystemExit(f"{directory} does not exist; expected <root>/{{train,valid,test}}/*.txt")
    chunks: list[str] = []
    total = 0
    for path in sorted(directory.rglob("*.txt")):
        cleaned = clean(path.read_text(encoding="utf-8", errors="ignore"))
        chunks.append(cleaned)
        total += len(cleaned)
        if limit_bytes is not None and total >= limit_bytes:
            break
    return "".join(chunks)[:limit_bytes] if limit_bytes else "".join(chunks)


def write_split(destination: Path, letters: str) -> dict[str, object]:
    """Write one split and return its manifest entry."""
    if not letters:
        raise SystemExit(f"refusing to write an empty split to {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for line in wrap(letters):
            handle.write(line + "\n")
    return {
        "path": destination.name,
        "letters": len(letters),
        "mib": round(len(letters) / MIB, 3),
        "sha256": hashlib.sha256(letters.encode()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the corpus and write ``manifest.json`` beside it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="corpus", help="output directory")
    parser.add_argument(
        "--source",
        default=f"hf:{DEFAULT_DATASET}:{DEFAULT_CONFIG}",
        help="'hf:<dataset>:<config>' or 'dir:<path>' with train/valid/test subdirectories",
    )
    parser.add_argument(
        "--train-mib", type=float, default=96.0, help="cap on training-split size in MiB"
    )
    parser.add_argument(
        "--holdout-mib",
        type=float,
        default=0.0,
        help="cap on valid/test size in MiB; 0 keeps the source split whole",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing corpus")
    args = parser.parse_args(argv)

    out = Path(args.out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"{manifest_path} already exists; pass --force to rebuild", file=sys.stderr)
        return 1

    train_cap = int(args.train_mib * MIB)
    holdout_cap = int(args.holdout_mib * MIB) or None

    if args.source.startswith("dir:"):
        root = Path(args.source[4:])
        splits = {
            "train": _dir_split(root, "train", train_cap),
            "valid": _dir_split(root, "valid", holdout_cap),
            "test": _dir_split(root, "test", holdout_cap),
        }
        source_info: dict[str, object] = {"kind": "dir", "root": str(root)}
    elif args.source.startswith("hf:"):
        _, dataset, config = args.source.split(":", 2)
        splits = {
            "train": _hf_split(dataset, config, "train", train_cap),
            "valid": _hf_split(dataset, config, "validation", holdout_cap),
            "test": _hf_split(dataset, config, "test", holdout_cap),
        }
        source_info = {"kind": "huggingface", "dataset": dataset, "config": config}
    else:
        parser.error(f"unrecognised --source {args.source!r}")
        return 2

    entries = {name: write_split(out / f"{name}.txt", letters) for name, letters in splits.items()}
    manifest = {
        "fetch_version": FETCH_VERSION,
        "built_at": datetime.now(UTC).isoformat(),
        "source": source_info,
        "cleaning": "lowercased, all characters outside a-z removed",
        "splits": entries,
        "total_mib": round(sum(entry["letters"] for entry in entries.values()) / MIB, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, entry in entries.items():
        print(f"{name:<6} {entry['mib']:>8.2f} MiB  sha256 {entry['sha256'][:16]}")
    print(f"total  {manifest['total_mib']:>8.2f} MiB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
