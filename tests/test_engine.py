"""Tests for mllm51.engine."""

import random

import pytest

from mllm51.engine import MASK, DiscreteDiffusionEngine
from mllm51.topology import BidirectionalTopology

CORPUS = (
    "the cat sat on the mat. the cat ate the fish. the dog sat on the mat. "
    "a cat is an animal. the dog is an animal. animals eat food and sleep."
)


@pytest.fixture
def topo() -> BidirectionalTopology:
    return BidirectionalTopology.from_text(CORPUS)


@pytest.fixture
def engine(topo) -> DiscreteDiffusionEngine:
    return DiscreteDiffusionEngine(topo, rng=random.Random(42))


class TestCandidateDistribution:
    def test_is_pure_and_normalized(self, engine):
        seq = ["the", MASK, "sat"]
        before = list(seq)
        probs = engine.candidate_distribution(seq, 1)
        assert seq == before  # no mutation
        assert sum(probs.values()) == pytest.approx(1.0)
        assert all(p > 0 for p in probs.values())

    def test_observed_words_dominate(self, engine):
        seq = ["the", MASK, "sat"]
        probs = engine.candidate_distribution(seq, 1)
        # "cat"/"dog" appear between "the" and "sat" in the corpus
        assert probs.get("cat", 0) > 0
        best = max(probs, key=probs.get)
        assert best in probs  # sanity
        assert probs["cat"] + probs["dog"] > 0.5

    def test_mask_in_context_disables_that_context(self, engine):
        # Both neighbors masked -> only unigram prior + backoff drives scores
        probs = engine.candidate_distribution([MASK, MASK, MASK], 1)
        assert sum(probs.values()) == pytest.approx(1.0)


class TestDenoise:
    def test_fills_every_mask(self, engine):
        result = engine.denoise(target_len=8, steps=5, prompt=["the"])
        assert MASK not in result.sequence
        assert len(result.sequence) == 8
        assert len(result.confidences) == 8

    def test_prompt_tokens_are_locked(self, engine):
        result = engine.denoise(target_len=6, steps=5, prompt=["the", "cat"])
        assert result.sequence[:2] == ["the", "cat"]

    def test_seed_reproducibility(self, topo):
        a = DiscreteDiffusionEngine(topo, rng=random.Random(7)).denoise(8, 6, ["the"])
        b = DiscreteDiffusionEngine(topo, rng=random.Random(7)).denoise(8, 6, ["the"])
        assert a.sequence == b.sequence
        assert a.confidences == b.confidences

    def test_on_step_callback_fires_each_step(self, engine):
        calls = []
        engine.denoise(5, 4, [], on_step=lambda seq, t, total: calls.append((t, total)))
        assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_invalid_arguments_raise(self, engine):
        with pytest.raises(ValueError, match="target_len"):
            engine.denoise(0, 5)
        with pytest.raises(ValueError, match="steps"):
            engine.denoise(5, 0)
