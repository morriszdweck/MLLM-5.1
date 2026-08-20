# MLLM-5.1

**Discrete Diffusion Micro Language Model — sculpt text from noise.**

MLLM-5.1 trades a bit of quality for speed. Its diffusion architecture looks
**both forward and backward** when predicting tokens and **re-masks low-confidence
fills** to fix its own mistakes. More steps = more refinement — the **effort knob**.

> **Single-file, zero-dependency.** `python MLLM-5.1.py` is the whole model.

```
 ███╗   ███╗██╗     ██╗     ███╗   ███╗      ███████╗        ██╗
 ████╗ ████║██║     ██║     ████╗ ████║      ██╔════╝        ██║
 ██╔████╔██║██║     ██║     ██╔████╔██║█████╗███████╗        ██║
 ██║╚██╔╝██║██║     ██║     ██║╚██╔╝██║╚════╝╚════██║        ██║
 ██║ ╚═╝ ██║███████╗███████╗██║ ╚═╝ ██║      ███████║██╗     ██║
 ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝      ╚══════╝╚═╝     ╚═╝
        Discrete Diffusion · Bidirectional · Sculpt-from-Noise
```

This repository contains **MLLM-5.1 (Preview-36P rebuilt)** — renewed from scratch as a self-contained Python file.

## Quick start

Requires **Python 3.10+**, no pip, no venv.

```bash
git clone https://github.com/morriszdweck/MLLM-5.1.git
cd MLLM-5.1

# chat (default)
python MLLM-5.1.py
python MLLM-5.1.py chat --steps 30 --seed 42

# one-shot
python MLLM-5.1.py generate "what is an atom" --steps 30 --seed 42
python MLLM-5.1.py generate "hello world" --show-steps --extra-tokens 8 14
```

Also works as an installed CLI if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
mllm51 generate "what is an atom" --steps 30 --seed 42
mllm51 chat
# or without install
python -m mllm51 generate "hello" --steps 20
```

On macOS Homebrew Python, `pip install` without a venv is blocked (PEP 668) — use a venv above or `pipx install git+https://github.com/morriszdweck/MLLM-5.1.git`.

## Flags

| Flag | Description |
|---|---|
| `--steps N` | Diffusion steps — the **effort knob** (default `30`). Higher = more refinement. |
| `--extra-tokens MIN MAX` | Tokens to generate beyond the prompt (default `10 18`). |
| `--seed N` | Deterministic sampling. Works before or after the subcommand. |
| `--corpus PATH` | Train on your own text file (default: embedded built-in corpus). |
| `--show-steps` | Visualize intermediate diffusion states. |
| `--no-color` | Disable ANSI colors (auto-disabled when piping or `NO_COLOR=1`). |
| `-v` | Debug logging. |
| `--version` | Print version. |

Chat extras (inside the REPL):

```
[quit] / quit / exit / :q   — leave
:help                       — help
:clear                      — clear screen
:steps N                    — change effort live
:seed N  / :seed none       — change seed live
:show on|off                — toggle step visualization
```

## How it works

1. **Topology building** (`BidirectionalTopology`). The corpus is tokenized (`\b[a-zA-Z0-9']+\b|[.!?]`) and counted into left/right n-gram tables for distances **1–3**. Counts and totals are cached; sentences never bleed across boundaries. (`topology` section in `MLLM-5.1.py`).

2. **Discrete diffusion** (`DiscreteDiffusionEngine`). Generation starts fully masked (`[MASK] * target_len`, prompt tokens locked at the front). Each step:
   - score every `[MASK]` against its bidirectional contexts (log-prob weighted by `n`, plus unigram prior),
   - sample under **annealed temperature** `1.2*(1 - t/steps)+0.2` (hot → cold),
   - re-mask the least-confident `1 - t/steps` fraction — sculpting text from noise.

3. **Rendering**. Progress bar + per-token confidence heatmap (green/yellow/red) and detokenized text.

Deterministic with `--seed` — candidate ordering is sorted so results reproduce **across processes** (`PYTHONHASHSEED`-independent).

## Examples

```bash
# reproducible atom answer
python MLLM-5.1.py --seed 42 generate "what is an atom" --steps 15

# watch it denoise
python MLLM-5.1.py --seed 7 generate "the cat" --steps 12 --show-steps

# custom corpus
python MLLM-5.1.py --corpus ./my.txt --seed 1 generate "hello"

# effort trade-off
python MLLM-5.1.py generate "explain diffusion" --steps 10
python MLLM-5.1.py generate "explain diffusion" --steps 40  # slower, sharper
```

## Project layout

```
MLLM-5.1.py        ← self-contained model (just run it)
mllm51/            ← installable package (re-exports same logic)
  topology.py        n-gram topology
  engine.py          diffusion engine (sorted candidates → deterministic)
  cli.py             CLI (generate / chat)
  terminal.py        ANSI colors with NO_COLOR/tty detection
data/corpus.txt    ← source for the embedded corpus (also inside MLLM-5.1.py)
tests/             ← pytest suite (29 tests)
pyproject.toml     ← pip metadata
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -v
```

## Roadmap

- [x] Effort control (`--steps`)
- [x] Single-file self-contained runner
- [x] Cross-process deterministic sampling
- [ ] Playground
- [ ] Studio

Planned models: Tasmania 1072P · Meridian 200P · Fjord 50P · Mesa 10P  
Current: **MLLM-5.1 Preview-36P (rebuilt v5.1.1)** — the preview model.

---
*Core principles preserved: bidirectional context, iterative denoising, confidence-guided remasking, sculpt-from-noise.*
