"""Bidirectional n-gram topology built from a text corpus.

The topology is the "training data" of Diff-MLLM-5.1: for every observed
word it records which words appeared to its left and right, at distances
(n-gram orders) 1..max_n. Context totals are maintained incrementally so
scoring never has to re-sum a Counter.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word and punctuation tokens.

    Words may contain letters, digits, and apostrophes; the punctuation
    marks ``.``, ``!``, and ``?`` become standalone tokens.
    """
    return TOKEN_RE.findall(text.lower())


class BidirectionalTopology:
    """N-gram statistics over left and right contexts of a corpus.

    Attributes:
        max_n: Maximum n-gram order.
        left_counts: ``left_counts[n][left_context][word] -> count``.
        right_counts: ``right_counts[n][right_context][word] -> count``.
        left_totals: ``left_totals[n][context] -> total count`` (cached).
        right_totals: ``right_totals[n][context] -> total count`` (cached).
        unigrams: Corpus-wide word frequencies.
        vocab: Set of all observed tokens.
    """

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.right_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.right_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()

    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "BidirectionalTopology":
        """Build a topology from a raw corpus string."""
        topo = cls(max_n=max_n)
        topo.ingest(text)
        return topo

    def ingest(self, text: str) -> None:
        """Add *text* to the topology. May be called repeatedly.

        Raises:
            ValueError: If the text contains no usable tokens.
        """
        if not text or not text.strip():
            raise ValueError("corpus text is empty")

        flattened = WHITESPACE_RE.sub(" ", text).strip()
        ingested_tokens = 0
        for sentence in SENTENCE_SPLIT_RE.split(flattened):
            words = tokenize(sentence)
            if not words:
                continue
            ingested_tokens += len(words)
            self.vocab.update(words)
            self.unigrams.update(words)
            for n in range(1, self.max_n + 1):
                for i, target in enumerate(words):
                    if i >= n:
                        ctx = tuple(words[i - n : i])
                        self.left_counts[n][ctx][target] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
                    if i + n < len(words):
                        ctx = tuple(words[i + 1 : i + 1 + n])
                        self.right_counts[n][ctx][target] += 1
                        self.right_totals[n][ctx] = self.right_totals[n].get(ctx, 0) + 1

        if ingested_tokens == 0:
            raise ValueError("corpus text contains no usable tokens")
