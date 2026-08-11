"""Discrete diffusion decoding engine over a :class:`BidirectionalTopology`.

Generation starts from a fully masked sequence. Each step scores every
masked position against its bidirectional n-gram contexts, samples a word
under an annealed temperature, and then re-masks the least-confident
positions — sculpting text out of noise.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from .topology import BidirectionalTopology

logger = logging.getLogger(__name__)

MASK = "<mask>"
_FLOOR = 1e-5  # pseudo-count keeping zero-count probabilities finite
_UNIGRAM_WEIGHT = 0.1  # weight of the corpus-wide unigram prior
_BACKOFF_SIZE = 50  # top unigram words always included as candidates

StepCallback = Callable[[list[str], int, int], None]


@dataclass
class DenoiseResult:
    """Outcome of a denoising run."""

    sequence: list[str]
    confidences: list[float]


class DiscreteDiffusionEngine:
    """Iteratively denoises masked sequences using n-gram statistics.

    Args:
        topo: Trained bidirectional topology.
        rng: Random source. Pass ``random.Random(seed)`` for reproducible
            output; defaults to a nondeterministic instance.
    """

    def __init__(self, topo: BidirectionalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]

    # ------------------------------------------------------------- scoring
    def _active_contexts(self, seq: Sequence[str], idx: int):
        """Yield (side, n, context) for every fully-unmasked context of *idx*."""
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n : idx])
                if MASK not in ctx:
                    yield "left", n, ctx
            if idx + n < len(seq):
                ctx = tuple(seq[idx + 1 : idx + 1 + n])
                if MASK not in ctx:
                    yield "right", n, ctx

    def candidate_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Softmax distribution over candidate words for position *idx*.

        Pure function: *seq* is never modified. Candidates are the words
        observed in the active contexts plus a unigram backoff set; words
        outside this set have vanishing probability (only the floor
        pseudo-count distinguishes them) and are omitted.
        """
        base = 0.0
        contrib: dict[str, float] = {}
        for side, n, ctx in self._active_contexts(seq, idx):
            if side == "left":
                counts = self.topo.left_counts[n].get(ctx)
                total = self.topo.left_totals[n].get(ctx, 0)
            else:
                counts = self.topo.right_counts[n].get(ctx)
                total = self.topo.right_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )

        candidates = set(contrib) | set(self._backoff)
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT
            + base
            + contrib.get(w, 0.0)
            for w in candidates
        }
        max_energy = max(energies.values())
        exp_scores = {w: math.exp(e - max_energy) for w, e in energies.items()}
        total_exp = sum(exp_scores.values())
        return {w: e / total_exp for w, e in exp_scores.items()}

    # ------------------------------------------------------------ decoding
    def denoise(
        self,
        target_len: int,
        steps: int,
        prompt: Sequence[str] = (),
        on_step: StepCallback | None = None,
    ) -> DenoiseResult:
        """Sculpt a sequence of *target_len* tokens out of noise.

        Prompt tokens are locked in place at the start of the sequence.
        Each of the *steps* iterations fills remaining masks under an
        annealed temperature, then re-masks the least-confident fills.

        Args:
            target_len: Total sequence length (prompt + generated tokens).
            steps: Number of diffusion steps — the "effort" knob.
            prompt: Tokens to lock at the start of the sequence.
            on_step: Optional callback ``(seq, step, total_steps)`` invoked
                after every step for progress rendering.

        Returns:
            DenoiseResult with the final sequence and per-position
            confidences (locked prompt positions stay 0.0).
        """
        if target_len < 1:
            raise ValueError(f"target_len must be >= 1, got {target_len}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        seq = [MASK] * target_len
        locked: set[int] = set()
        for i, word in enumerate(prompt[:target_len]):
            seq[i] = word.lower()
            locked.add(i)

        confidences = [0.0] * target_len
        for t in range(1, steps + 1):
            temp = 1.2 * (1.0 - (t / steps)) + 0.2
            current: dict[int, float] = {}

            for i in range(target_len):
                if i in locked or seq[i] != MASK:
                    continue
                probs = self.candidate_distribution(seq, i)
                words = list(probs)
                weights = [p ** (1.0 / temp) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                seq[i] = chosen
                current[i] = probs[chosen]
                confidences[i] = probs[chosen]

            re_mask_ratio = max(0.0, 1.0 - (t / steps))
            num_to_remask = int(len(current) * re_mask_ratio)
            if num_to_remask > 0 and current:
                for i in sorted(current, key=current.get)[:num_to_remask]:
                    seq[i] = MASK
                    confidences[i] = 0.0

            if on_step is not None:
                on_step(list(seq), t, steps)

        logger.debug("denoise finished: %d tokens in %d steps", target_len, steps)
        return DenoiseResult(sequence=seq, confidences=confidences)
