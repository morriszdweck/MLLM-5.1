"""Tests for mllm51.cli."""

import builtins

import pytest

from mllm51.cli import DEFAULT_CORPUS, load_topology, main

CORPUS = (
    "the cat sat on the mat. the cat ate the fish. the dog sat on the mat. "
    "a cat is an animal. the dog is an animal. animals eat food and sleep."
)


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return path


class TestGenerate:
    def test_one_shot_generate(self, corpus_file, capsys):
        rc = main(["--corpus", str(corpus_file), "--seed", "42",
                   "generate", "the cat", "--steps", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Full Sculpted Text:" in out
        assert "Confidence Heatmap" in out
        assert "the cat" in out

    def test_seed_makes_output_reproducible(self, corpus_file, capsys):
        main(["--corpus", str(corpus_file), "--seed", "1", "generate", "the", "--steps", "4"])
        first = capsys.readouterr().out
        main(["--corpus", str(corpus_file), "--seed", "1", "generate", "the", "--steps", "4"])
        second = capsys.readouterr().out
        assert first == second

    def test_show_steps(self, corpus_file, capsys):
        rc = main(["--corpus", str(corpus_file), "--seed", "3",
                   "generate", "the", "--steps", "5", "--show-steps"])
        assert rc == 0
        assert "[Step 05/5]" in capsys.readouterr().out

    def test_missing_corpus_fails_cleanly(self, tmp_path, capsys):
        rc = main(["--corpus", str(tmp_path / "nope.txt"), "generate", "hi"])
        assert rc == 2
        assert "cannot load corpus" in capsys.readouterr().err

    def test_bad_extra_tokens_range_fails(self, corpus_file, capsys):
        rc = main(["--corpus", str(corpus_file),
                   "generate", "hi", "--extra-tokens", "18", "10"])
        assert rc == 2
        assert "--extra-tokens" in capsys.readouterr().err


class TestChat:
    def test_chat_roundtrip_then_eof(self, corpus_file, capsys, monkeypatch):
        inputs = iter(["the cat"])

        def fake_input(prompt=""):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr(builtins, "input", fake_input)
        rc = main(["--corpus", str(corpus_file), "--seed", "9", "chat", "--steps", "4"])
        assert rc == 0
        assert "Full Sculpted Text:" in capsys.readouterr().out

    def test_quit_command_exits(self, corpus_file, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda prompt="": "[QUIT]")
        assert main(["--corpus", str(corpus_file), "chat", "--steps", "4"]) == 0


class TestBundledCorpus:
    def test_default_corpus_loads(self):
        topo = load_topology(DEFAULT_CORPUS)
        assert len(topo.vocab) > 1000
