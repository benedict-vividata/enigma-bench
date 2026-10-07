"""``enigma-bench`` — the operator-facing command line.

The agent inside the container works through the MCP tools; this CLI is for
whoever sets the benchmark up and analyses it afterwards. Everything the agent
can do through MCP is available here too, which makes the harness testable
without an agent in the loop.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import runlog
from .checkpoint import describe_checkpoint, save_checkpoint
from .corpus import Corpus, CorpusError, load_manifest, load_split
from .dataset import build_example
from .evaluate import DEFAULT_BATCH_SIZE, DEFAULT_SAMPLES_PER_TIER
from .harness import resolve_checkpoint, run_evaluation, time_budget
from .machine import EnigmaMachine, EnigmaSettings
from .paths import corpus_dir, runlog_path, submissions_dir
from .spec import (
    SIZE,
    SIZE_ENV,
    SIZES,
    SPEC_VERSION,
    architecture,
    architecture_fingerprint,
    parameter_count,
)
from .tiers import TIERS, TIERS_BY_ID, resolve_tiers
from .tokenizer import describe_input


def _describe(_: argparse.Namespace) -> int:
    print(f"Enigma Bench — spec v{SPEC_VERSION}  fingerprint {architecture_fingerprint()}")
    print(f"model size: {SIZE.name} (set with {SIZE_ENV})")
    print(f"parameters: {parameter_count():,}\n")
    print("architecture:")
    for key, value in architecture().items():
        print(f"  {key:<22} {value}")
    print("\ntiers:")
    for index, tier in enumerate(TIERS):
        plug = f"{tier.plugboard_pairs} pairs" if tier.plugboard_pairs else "no plugboard"
        print(f"  [{index}] {tier.id:<20} {plug:<13} weight {tier.weight}")
        print(f"      {tier.description}")
    print("\npaths:")
    print(f"  corpus      {corpus_dir()}")
    print(f"  submissions {submissions_dir()}")
    print(f"  run log     {runlog_path()}")
    manifest = load_manifest()
    if manifest:
        print(f"  corpus built from {manifest.get('source', 'unknown')}")
    return 0


def _sizes(_: argparse.Namespace) -> int:
    """Print the capacity ladder, including the sizes this process is not running at."""
    print(f"Model sizes — this process is running '{SIZE.name}' (set with {SIZE_ENV})\n")
    header = f"{'size':<8} {'d_model':>7} {'layers':>6} {'heads':>5} {'d_ff':>5} "
    header += f"{'aspect':>6} {'params':>12}  fingerprint"
    print(header)
    print("-" * len(header))
    for size in SIZES.values():
        marker = "*" if size.name == SIZE.name else " "
        print(
            f"{size.name + marker:<8} {size.d_model:>7} {size.num_layers:>6} "
            f"{size.num_heads:>5} {size.d_ff:>5} {size.aspect_ratio:>6.1f} "
            f"{parameter_count(size):>12,}  {architecture_fingerprint(size)}"
        )
    print("\nEvery size keeps 64 channels per head and an MLP 4x the model width.")
    print("A checkpoint is only loadable at the size it was trained under.")
    return 0


def _selftest(_: argparse.Namespace) -> int:
    """Check the Enigma against published vectors and its structural invariants."""
    failures: list[str] = []

    base = EnigmaSettings(
        rotors=("I", "II", "III"),
        reflector="B",
        ring_settings=(0, 0, 0),
        ground_setting=(0, 0, 0),
    )
    expected = "BDZGOWCXLTKSBTMCDLPBMUQOF"
    actual = EnigmaMachine(base).encrypt("A" * 25)
    if actual != expected:
        failures.append(f"known-answer vector: got {actual}, want {expected}")

    rng = random.Random(1234)
    for _ in range(200):
        tier = rng.choice(TIERS)
        settings = tier.sample_settings(rng)
        plaintext = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(220))
        ciphertext = EnigmaMachine(settings).encrypt(plaintext)
        if EnigmaMachine(settings).encrypt(ciphertext) != plaintext:
            failures.append(f"not reciprocal under {settings}")
            break
        if any(a == b for a, b in zip(plaintext, ciphertext, strict=True)):
            failures.append(f"letter enciphered to itself under {settings}")
            break

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("ok — known-answer vector, reciprocity and no-fixed-point all pass")
    return 0


def _sample(args: argparse.Namespace) -> int:
    tier = TIERS_BY_ID[args.tier]
    corpus = _load_corpus(args)
    example = build_example(tier, corpus, random.Random(args.seed))
    print(f"tier      : {tier.id} — {tier.name}")
    print(f"machine   : {example.settings}")
    print(describe_input(list(example.input_ids)))
    print(f"plaintext : {example.plaintext_str().lower()}")
    return 0


def _load_corpus(args: argparse.Namespace) -> Corpus:
    root = Path(args.corpus) if getattr(args, "corpus", None) else None
    return load_split(getattr(args, "split", "valid"), root)


def _validate(args: argparse.Namespace) -> int:
    report = describe_checkpoint(resolve_checkpoint(args.checkpoint))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("valid") else 1


def _evaluate(args: argparse.Namespace) -> int:
    outcome = run_evaluation(
        args.checkpoint,
        label=args.label,
        notes=args.notes,
        tier_ids=args.tiers,
        samples_per_tier=args.samples_per_tier,
        device=args.device,
        batch_size=args.batch_size,
        corpus_root=Path(args.corpus) if args.corpus else None,
        extra={"submitted_via": "cli"},
    )
    if not outcome["ok"]:
        print(f"{outcome['status']}: {outcome['detail']}", file=sys.stderr)
        return 1
    result = outcome["result"]
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.format_table())
        record = outcome["record"]
        print(f"\nlogged as {record['run_id']} (sequence {record['sequence']})")
    return 0


def _runs(args: argparse.Namespace) -> int:
    path = Path(args.log) if args.log else runlog_path()
    records = runlog.read_records(path)
    if args.verify:
        ok, problems = runlog.verify_chain(path)
        for problem in problems:
            print(f"CHAIN {problem}", file=sys.stderr)
        print(f"chain: {'intact' if ok else 'BROKEN'} over {len(records)} records")
        if not ok:
            return 1
    if args.json:
        print(json.dumps(records[-args.limit :], indent=2, default=str))
        return 0
    if not records:
        print(f"no runs logged in {path}")
        return 0
    print(f"{'seq':>4} {'run id':<14} {'label':<22} {'status':<20} {'score':>7} {'grade':>6}")
    for record in runlog.summarise(records[-args.limit :]):
        score = record.get("score")
        rendered = "      -" if score is None else f"{score:7.2f}"
        print(
            f"{record['sequence']:>4} {record.get('run_id')!s:<14} "
            f"{record.get('label') or '-':<22} {record.get('status')!s:<20} "
            f"{rendered} {record.get('grade', '-')!s:>6}"
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    path = Path(args.log) if args.log else runlog_path()
    records = runlog.read_records(path)
    ok, problems = runlog.verify_chain(path)
    scored = [r for r in records if r.get("status") == "ok" and not r.get("partial")]
    best = runlog.best_record(scored)

    lines = [
        "# Enigma Bench run report",
        "",
        f"- run log: `{path}`",
        f"- records: {len(records)} ({len(scored)} full scored evaluations)",
        f"- chain integrity: {'intact' if ok else 'BROKEN — ' + '; '.join(problems)}",
        f"- spec: v{SPEC_VERSION}, model size `{SIZE.name}` "
        f"({parameter_count():,} parameters), fingerprint `{architecture_fingerprint()}`",
        "",
    ]
    if best:
        lines += [
            "## Best submission",
            "",
            f"- run id: `{best.get('run_id')}` (sequence {best.get('sequence')})",
            f"- label: {best.get('label') or '-'}",
            f"- score: **{best.get('score')}** / 100, grade `{best.get('grade_label')}`",
            f"- reached after {best.get('elapsed_since_first_run_seconds')} s of run time",
            "",
            "| tier | letter accuracy | exact match | loss |",
            "| --- | --- | --- | --- |",
        ]
        for tier in best.get("per_tier", []):
            lines.append(
                f"| `{tier['tier_id']}` | {tier['letter_accuracy']:.2%} | "
                f"{tier['exact_match']:.2%} | {tier['mean_loss']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Submission history",
        "",
        "| seq | label | status | score | grade | elapsed (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in runlog.summarise(records):
        lines.append(
            f"| {record['sequence']} | {record.get('label') or '-'} | {record.get('status')} | "
            f"{record.get('score', '-')} | {record.get('grade', '-')} | "
            f"{record.get('elapsed_since_first_run_seconds', '-')} |"
        )
    report = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


def _random_checkpoint(args: argparse.Namespace) -> int:
    """Write freshly initialised weights — the chance-level baseline and a format smoke test."""
    import torch

    from .model import EnigmaEncoder

    torch.manual_seed(args.seed)
    path = Path(args.path)
    save_checkpoint(EnigmaEncoder(), path, metadata={"note": "randomly initialised baseline"})
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def _budget(_: argparse.Namespace) -> int:
    print(json.dumps(time_budget(), indent=2, default=str))
    return 0


def _mcp(_: argparse.Namespace) -> int:
    from .mcp_server import main as serve

    serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(prog="enigma-bench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="print the frozen spec and the tier ladder")
    describe.set_defaults(handler=_describe)
    sizes = subparsers.add_parser("sizes", help="print the model-size ladder")
    sizes.set_defaults(handler=_sizes)
    selftest = subparsers.add_parser("selftest", help="check the Enigma against known vectors")
    selftest.set_defaults(handler=_selftest)
    budget = subparsers.add_parser("budget", help="show time and evaluation quota remaining")
    budget.set_defaults(handler=_budget)
    subparsers.add_parser("mcp", help="serve the MCP tools on stdio").set_defaults(handler=_mcp)

    sample = subparsers.add_parser("sample", help="print one encoded example from a tier")
    sample.add_argument("--tier", default=TIERS[0].id, choices=[tier.id for tier in TIERS])
    sample.add_argument(
        "--split",
        default="valid",
        choices=("train", "valid", "test"),
        help="corpus split to draw the plaintext from ('test' is the scorer's, and is "
        "unreadable inside the benchmark container)",
    )
    sample.add_argument("--corpus", help="corpus directory (defaults to ENIGMA_BENCH_CORPUS)")
    sample.add_argument("--seed", type=int, default=0)
    sample.set_defaults(handler=_sample)

    validate = subparsers.add_parser("validate", help="check a checkpoint against the spec")
    validate.add_argument("checkpoint")
    validate.set_defaults(handler=_validate)

    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint and log the result")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--label")
    evaluate.add_argument("--notes")
    evaluate.add_argument("--tiers", nargs="+", choices=[tier.id for tier in TIERS])
    evaluate.add_argument("--samples-per-tier", type=int, default=DEFAULT_SAMPLES_PER_TIER)
    evaluate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate.add_argument("--device", help="cuda or cpu; defaults to cuda when available")
    evaluate.add_argument("--corpus", help="corpus directory (defaults to ENIGMA_BENCH_CORPUS)")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    runs = subparsers.add_parser("runs", help="list logged evaluations")
    runs.add_argument("--limit", type=int, default=25)
    runs.add_argument("--log", help="run log path (defaults to ENIGMA_BENCH_RUNLOG)")
    runs.add_argument("--verify", action="store_true", help="also check the hash chain")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(handler=_runs)

    report = subparsers.add_parser("report", help="render a markdown report from the run log")
    report.add_argument("--log", help="run log path (defaults to ENIGMA_BENCH_RUNLOG)")
    report.add_argument("--out", help="write to this file instead of stdout")
    report.set_defaults(handler=_report)

    baseline = subparsers.add_parser(
        "random-checkpoint", help="write untrained weights (chance-level baseline)"
    )
    baseline.add_argument("path")
    baseline.add_argument("--seed", type=int, default=0)
    baseline.set_defaults(handler=_random_checkpoint)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``enigma-bench`` console script."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (CorpusError, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "resolve_tiers"]
