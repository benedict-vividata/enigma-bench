"""Loading the English plaintext corpus.

The corpus is stored as cleaned lowercase ``a..z`` with newlines every 80
characters for greppability; the loader strips everything that is not a letter,
so a plaintext window is a contiguous run of real English with the spaces and
punctuation removed — exactly what a wartime clerk would have typed.

Three splits:

* ``train`` (~96 MiB) and ``valid`` (~2 MiB) live in the agent's workspace.
* ``test`` (~2 MiB) is the eval's plaintext source and is deliberately kept out
  of the agent's reach in the hardened container.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .paths import corpus_dir

SPLITS: Final[tuple[str, ...]] = ("train", "valid", "test")


class CorpusError(RuntimeError):
    """Raised when a corpus file is missing, empty, or too short to sample from."""


def clean(text: str) -> bytes:
    """Reduce arbitrary text to letter indices ``0..25`` as a ``bytes`` buffer."""
    return bytes(ord(char) - 97 for char in text.lower() if "a" <= char <= "z")


@dataclass(frozen=True, slots=True)
class Corpus:
    """An in-memory stream of letter indices with a content hash for provenance."""

    name: str
    letters: bytes
    sha256: str

    def __len__(self) -> int:
        return len(self.letters)

    def window(self, start: int, length: int) -> list[int]:
        """``length`` letters starting at ``start``, wrapping at the end of the stream."""
        if length <= 0:
            raise ValueError(f"window length must be positive, got {length}")
        if len(self.letters) < length:
            raise CorpusError(
                f"corpus {self.name!r} has {len(self.letters)} letters, "
                f"fewer than the requested window of {length}"
            )
        start %= len(self.letters)
        end = start + length
        if end <= len(self.letters):
            return list(self.letters[start:end])
        return list(self.letters[start:]) + list(self.letters[: end - len(self.letters)])

    def random_window(self, rng: random.Random, length: int) -> list[int]:
        """A window starting at a uniformly random offset."""
        return self.window(rng.randrange(len(self.letters)), length)

    @classmethod
    def from_text(cls, text: str, name: str = "inline") -> Corpus:
        """Build a corpus from a raw string (used by tests and quick experiments)."""
        letters = clean(text)
        if not letters:
            raise CorpusError(f"corpus {name!r} contains no a-z characters")
        return cls(name=name, letters=letters, sha256=hashlib.sha256(letters).hexdigest())

    @classmethod
    def from_path(cls, path: Path) -> Corpus:
        """Load and clean a corpus file."""
        if not path.is_file():
            raise CorpusError(
                f"corpus file {path} not found — run scripts/fetch_corpus.py to build it"
            )
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except PermissionError as exc:
            raise CorpusError(
                f"corpus file {path} is not readable by this user. The held-out 'test' split "
                "belongs to the scorer; train and validate against 'train' and 'valid'."
            ) from exc
        except OSError as exc:
            raise CorpusError(f"corpus file {path} could not be read: {exc}") from exc
        return cls.from_text(text, name=path.stem)


def split_path(split: str, root: Path | None = None) -> Path:
    """Path of one corpus split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return (root or corpus_dir()) / f"{split}.txt"


def load_split(split: str, root: Path | None = None) -> Corpus:
    """Load one corpus split by name."""
    return Corpus.from_path(split_path(split, root))


def load_manifest(root: Path | None = None) -> dict[str, object]:
    """Read ``manifest.json``, or return an empty dict when the corpus is unbuilt."""
    path = (root or corpus_dir()) / "manifest.json"
    if not path.is_file():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}
