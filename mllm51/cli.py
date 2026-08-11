"""Command-line interface for Diff-MLLM-5.1.

Subcommands:
    chat      Interactive REPL (default when no subcommand is given).
    generate  One-shot generation from a prompt.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .engine import MASK, DenoiseResult, DiscreteDiffusionEngine
from .terminal import Term
from .topology import BidirectionalTopology, tokenize

logger = logging.getLogger("mllm51")

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"
_QUIT_COMMANDS = {"[quit]", "quit", "exit"}


# ------------------------------------------------------------------ setup
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mllm51",
        description="Diff-MLLM-5.1: a discrete diffusion language model that "
        "sculpts text out of noise using bidirectional n-gram statistics.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"path to a training corpus (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    def add_decoding_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--steps",
            type=int,
            default=30,
            help="diffusion steps — the effort knob (default: 30)",
        )
        sub.add_argument(
            "--extra-tokens",
            type=int,
            nargs=2,
            default=(10, 18),
            metavar=("MIN", "MAX"),
            help="range of tokens to generate beyond the prompt (default: 10 18)",
        )

    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate", help="one-shot generation from a prompt")
    gen.add_argument("prompt", help="prompt text; its tokens are locked in place")
    gen.add_argument(
        "--show-steps",
        action="store_true",
        help="render intermediate diffusion states",
    )
    add_decoding_args(gen)

    chat = subparsers.add_parser("chat", help="interactive REPL (default)")
    add_decoding_args(chat)

    return parser


def load_topology(corpus_path: Path) -> BidirectionalTopology:
    """Read *corpus_path* and build the topology.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the corpus has no usable tokens.
    """
    logger.info("loading corpus from %s", corpus_path)
    text = corpus_path.read_text(encoding="utf-8")
    topo = BidirectionalTopology.from_text(text, max_n=3)
    logger.info("topology built: %d vocab, %d unigram tokens", len(topo.vocab), sum(topo.unigrams.values()))
    return topo


# -------------------------------------------------------------- rendering
def make_step_printer(term: Term):
    """Return an on_step callback rendering a compact progress line."""

    def _print(seq: list[str], step: int, total_steps: int) -> None:
        interval = max(1, total_steps // 5)
        if step % interval != 0 and step != total_steps:
            return
        bar_len = 20
        filled = int(bar_len * (step / total_steps))
        bar = "█" * filled + "░" * (bar_len - filled)
        display = "".join(
            term.paint("red", "_") if w == MASK else term.paint("green", w[0]) for w in seq
        )
        print(term.paint("dim", f"[Step {step:02d}/{total_steps}] {bar}") + " " + display)

    return _print


def assemble_text(prompt_words: Sequence[str], result: DenoiseResult) -> tuple[str, str]:
    """Split the final sequence into (prompt_text, generated_text)."""
    prompt_str = " ".join(prompt_words)
    gen_words = [
        w for i, w in enumerate(result.sequence) if i >= len(prompt_words) and w != MASK
    ]
    gen_str = " ".join(gen_words)
    for mark in (",", ".", "!", "?"):
        gen_str = gen_str.replace(f" {mark}", mark)
    if prompt_str and not prompt_str.endswith((" ", ".", "!", "?")):
        prompt_str += " "
    if gen_str:
        gen_str = gen_str[0].upper() + gen_str[1:]
    return prompt_str, gen_str


def render_result(term: Term, prompt_words: Sequence[str], result: DenoiseResult) -> None:
    """Print the sculpted text and its confidence heatmap."""
    prompt_str, gen_str = assemble_text(prompt_words, result)

    print()
    print(term.paint("cyan", term.paint("bold", "Full Sculpted Text:")))
    print(
        term.paint("cyan", prompt_str)
        + term.paint("magenta", term.paint("bold", gen_str))
    )

    heat_chars = []
    for i, word in enumerate(result.sequence):
        if word == MASK:
            continue
        if i < len(prompt_words):
            heat_chars.append(term.paint("cyan", "█"))
        else:
            conf = result.confidences[i]
            color = "green" if conf > 0.6 else "yellow" if conf > 0.3 else "red"
            heat_chars.append(term.paint(color, "█"))
    print()
    print(term.paint("dim", "Confidence Heatmap (Prompt | Generated):"))
    print("".join(heat_chars))
    print(term.paint("dim", "[Green = High Conf, Yellow = Med Conf, Red = Low Conf]"))


# -------------------------------------------------------------- commands
def _decode(
    engine: DiscreteDiffusionEngine,
    prompt_text: str,
    steps: int,
    extra_tokens: tuple[int, int],
    on_step,
) -> tuple[list[str], DenoiseResult]:
    prompt_words = tokenize(prompt_text)
    extra_min, extra_max = extra_tokens
    target_len = len(prompt_words) + engine.rng.randint(extra_min, extra_max)
    result = engine.denoise(target_len=target_len, steps=steps, prompt=prompt_words, on_step=on_step)
    return prompt_words, result


def cmd_generate(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    on_step = make_step_printer(term) if args.show_steps else None
    prompt_words, result = _decode(engine, args.prompt, args.steps, args.extra_tokens, on_step)
    render_result(term, prompt_words, result)
    return 0


def cmd_chat(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    print(term.paint("magenta", term.paint("bold", "\n💬 Diff-MLLM-5.1 READY. Type a prompt to sculpt from noise. [QUIT] to exit.")))
    print(term.paint("dim", "(The model will iteratively denoise [MASK] tokens into words based on your prompt)\n"))

    step_printer = make_step_printer(term)
    while True:
        try:
            user = input(term.paint("bold", "You > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in _QUIT_COMMANDS:
            break

        print(term.paint("magenta", term.paint("bold", "MLLM-5.1 (Diffusion) > ")), flush=True)
        prompt_words, result = _decode(engine, user, args.steps, args.extra_tokens, step_printer)
        render_result(term, prompt_words, result)
        print()
    return 0


# ------------------------------------------------------------------ main
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    term = Term.detect(args.no_color)

    corpus_path: Path = args.corpus or DEFAULT_CORPUS
    try:
        topo = load_topology(corpus_path)
    except (OSError, ValueError) as exc:
        print(f"mllm51: error: cannot load corpus {corpus_path}: {exc}", file=sys.stderr)
        return 2

    if args.extra_tokens[0] > args.extra_tokens[1]:
        print("mllm51: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
        return 2

    engine = DiscreteDiffusionEngine(topo, rng=random.Random(args.seed))
    if args.command == "generate":
        return cmd_generate(args, engine, term)
    return cmd_chat(args, engine, term)


if __name__ == "__main__":
    raise SystemExit(main())
