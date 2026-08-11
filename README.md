# MLLM-5.1

**BETA** — the diffusion version of MLLM with a new architecture.

MLLM-5.1 is designed to be more efficient than MLLM-5, trading a bit of
quality for speed. Its diffusion architecture lets the model look both
forward and backward when predicting tokens, and re-mask low-confidence
tokens to fix its own errors.

This repository contains **MLLM-5.1 Preview-36P**, the current preview model.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/morriszdweck/MLLM-5.1.git
cd MLLM-5.1

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On macOS with Homebrew Python (and other PEP 668 systems), plain
`pip install` is blocked with an `externally-managed-environment` error —
the virtual environment above is the fix. Alternatively, install the CLI
isolated with [pipx](https://pipx.pypa.io/):

```bash
pipx install git+https://github.com/morriszdweck/MLLM-5.1.git
```

(Everything also works without installing: run `python -m mllm51` from the
repository root.)

## Usage

Interactive chat:

```bash
mllm51 chat
# or simply
mllm51
```

One-shot generation:

```bash
mllm51 generate "what is an atom" --steps 30 --seed 42
```

Useful flags:

| Flag | Description |
| --- | --- |
| `--steps N` | Number of diffusion steps — the **effort** knob. More steps = more refinement. |
| `--extra-tokens MIN MAX` | How many tokens to generate beyond the prompt (default `10 18`). |
| `--seed N` | Reproducible output. |
| `--corpus PATH` | Train on your own text file instead of the bundled corpus (`data/corpus.txt`). |
| `--show-steps` | Render intermediate diffusion states (one-shot mode). |
| `--no-color` | Disable ANSI colors (auto-disabled when piping). |
| `-v` | Debug logging. |

## How it works

1. **Training (topology building).** The corpus is tokenized and counted
   into a *bidirectional n-gram topology*: for every word, which words
   appear to its left and right at distances 1–3
   (`mllm51/topology.py`).
2. **Generation (discrete diffusion).** The output starts as a row of
   `[MASK]` tokens. Each step, every masked position is scored against its
   bidirectional contexts, a word is sampled under an annealed temperature,
   and the least-confident fills are re-masked for another pass — text is
   sculpted out of noise (`mllm51/engine.py`).
3. **Rendering.** The CLI shows the denoising progress, the final sculpted
   text, and a per-token confidence heatmap (`mllm51/cli.py`).

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest
```

## Roadmap

- [x] Effort control (`--steps`)
- [ ] Playground with MLLM-5.1 (coming soon)
- [ ] Studio (sometime later)

Planned models:

- MLLM-5.1 Tasmania 1072P
- MLLM-5.1 Meridian 200P
- MLLM-5.1 Fjord 50P
- MLLM-5.1 Mesa 10P

Current models:

- MLLM-5.1 Preview-36P (preview of the model)
