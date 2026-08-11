"""Tests for mllm51.topology."""

import pytest

from mllm51.topology import BidirectionalTopology, tokenize


class TestTokenize:
    def test_lowercases_and_splits_words(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_becomes_tokens(self):
        assert tokenize("Really? Yes! OK.") == ["really", "?", "yes", "!", "ok", "."]

    def test_apostrophes_stay_in_words(self):
        assert tokenize("don't stop") == ["don't", "stop"]

    def test_empty_text(self):
        assert tokenize("") == []


class TestIngest:
    def test_unigrams_and_vocab(self):
        topo = BidirectionalTopology.from_text("a b. a b.")
        assert topo.unigrams == {"a": 2, "b": 2, ".": 2}
        assert topo.vocab == {"a", "b", "."}

    def test_left_counts(self):
        topo = BidirectionalTopology.from_text("a b. a b.")
        assert topo.left_counts[1][("a",)]["b"] == 2
        assert topo.left_totals[1][("a",)] == 2

    def test_right_counts(self):
        topo = BidirectionalTopology.from_text("a b. a b.")
        # "a" is immediately left of "b" twice
        assert topo.right_counts[1][("b",)]["a"] == 2
        assert topo.right_totals[1][("b",)] == 2

    def test_bigram_contexts(self):
        topo = BidirectionalTopology.from_text("x y z.")
        assert topo.left_counts[2][("x", "y")]["z"] == 1
        assert topo.right_counts[2][("y", "z")]["x"] == 1

    def test_sentences_do_not_bleed_into_each_other(self):
        topo = BidirectionalTopology.from_text("a b. c d.")
        # "b" and "c" are in different sentences: no cross-sentence context
        assert ("b",) not in topo.left_counts[1] or topo.left_counts[1][("b",)].get("c", 0) == 0
        assert topo.right_counts[1][("c",)].get("b", 0) == 0

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BidirectionalTopology().ingest("   ")

    def test_tokenless_corpus_raises(self):
        with pytest.raises(ValueError, match="no usable tokens"):
            BidirectionalTopology().ingest("— – …")

    def test_invalid_max_n_raises(self):
        with pytest.raises(ValueError, match="max_n"):
            BidirectionalTopology(max_n=0)

    def test_totals_match_counts(self):
        topo = BidirectionalTopology.from_text("the cat sat. the cat ran. the dog sat.")
        for n in range(1, topo.max_n + 1):
            for ctx, total in topo.left_totals[n].items():
                assert total == sum(topo.left_counts[n][ctx].values())
            for ctx, total in topo.right_totals[n].items():
                assert total == sum(topo.right_counts[n][ctx].values())
