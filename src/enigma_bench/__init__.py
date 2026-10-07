"""Enigma Bench — a time-boxed benchmark for coding agents.

The task: inside an offline container with a GPU, train a fixed-architecture
transformer encoder that reads Enigma ciphertext (plus however much of the
machine key the tier reveals) and predicts the plaintext. Progress is measured
only through the ``evaluate_checkpoint`` MCP tool, which scores a submitted
weights file against a held-out corpus across seven difficulty tiers and
appends the result to an append-only, hash-chained run log.

Public surface:

* :mod:`enigma_bench.machine` — the Enigma itself
* :mod:`enigma_bench.spec` — the frozen architecture and sequence layout
* :mod:`enigma_bench.tokenizer`, :mod:`enigma_bench.dataset` — encoding examples
* :mod:`enigma_bench.tiers` — the difficulty ladder
* :mod:`enigma_bench.evaluate` — the scorer
* :mod:`enigma_bench.mcp_server` — the tools the agent sees
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
