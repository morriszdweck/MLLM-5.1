# MLLM-5.1

**Document Autocomplete Micro Language Model — ghost-text for your editor.**

MLLM-5.1 is a tiny, causal n-gram model that **continues your document** left-to-right. Type a prefix, get a dim ghost suggestion, `Tab` to accept — like Copilot for prose. No server, no install, just `python MLLM-5.1.py` or the browser playground.

> **Single-file, zero-dependency.** `python MLLM-5.1.py` is the whole model. `index.html` is the whole playground.

```
 ███╗   ███╗██╗     ██╗     ███╗   ███╗      ███████╗   ██╗  ██╗
 ████╗ ████║██║     ██║     ████╗ ████║      ██╔════╝   ██║  ██║
 ██╔████╔██║██║     ██║     ██╔████╔██║█████╗███████╗   ███████║
 ██║╚██╔╝██║██║     ██║     ██║╚██╔╝██║╚════╝╚════██║   ╚════██║
 ██║ ╚═╝ ██║███████╗███████╗██║ ╚═╝ ██║      ███████║██╗██║  ██║
 ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝      ╚══════╝╚═╝╚═╝  ╚═╝
        Autocomplete · Causal · Ghost-Text · Document Editor
```

**Live playground:** https://mllm-5-1.netlify.app — also `https://mllm-5-synapse.netlify.app` — open `index.html` locally via `file://`.

## Quick start — Python

Requires **Python 3.10+**, no pip, no venv.

```bash
git clone https://github.com/morriszdweck/MLLM-5.1.git
cd MLLM-5.1

# one-shot autocomplete
python MLLM-5.1.py autocomplete "what is an atom" --steps 12 --seed 42
python MLLM-5.1.py autocomplete "the quick brown" --temperature 0.3 --threshold 0.15

# aliases (generate is autocomplete, chat is REPL)
python MLLM-5.1.py generate "hello world" --steps 16
python MLLM-5.1.py chat --steps 16 --seed 7

# interactive autocomplete REPL (ghost + heatmap)
python MLLM-5.1.py
python MLLM-5.1.py chat
# inside REPL: type prefix → shows ghost continuation
# :help, :clear, :steps N, :temp N, :seed N
```

Also installable if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
mllm51 autocomplete "the quick brown" --steps 12
mllm51 generate "hello" --temperature 0.2
python -m mllm51 chat
```

On Homebrew Python, use a venv (PEP 668) or `pipx install git+https://github.com/morriszdweck/MLLM-5.1.git`.

## Quick start — HTML Playground

```bash
open index.html              # local, offline, no server
# or visit the deployed URL
open https://mllm-5-1.netlify.app
```

Editor features:

- **Ghost text** overlay (textarea + dim `#9aa0b6` ghost behind, sync scroll)
- **Tab** accept full ghost, **Ctrl+→** accept word, **Esc** dismiss, **Ctrl+Space** trigger
- **Controls:** Temperature (0.2–1.2), Steps/Length (5–40), Seed, Max n-gram (2–5), Threshold, Anneal, Auto-trigger 180ms debounce
- **Stats:** vocab/tokens/sentences, confidence heatmap (green>0.35 yellow>0.15 red), top-5 alternatives, word/char count
- **Corpus:** paste or upload `.txt`, Rebuild (causal 1..3), Reset, Export

Self-contained — all CSS/JS inline, no fetch, no npm, no build.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--steps N` | `16` | Max tokens to generate (effort knob) |
| `--extra-tokens MIN MAX` | `steps..steps` | Range beyond prefix (alias) |
| `--temperature T` | `0.35` | `0.0` greedy → `1.2` creative |
| `--threshold T` | `0.0` | Min confidence to emit |
| `--max-ngram N` | `3` | Causal context size (1–5) |
| `--seed N` | random | Deterministic sampling (sorted candidates → cross-process reproducible) |
| `--corpus PATH` | embedded | Train on your own text |
| `--show-steps` | off | Show diffusion steps (legacy `generate`) |
| `--no-color` | auto | Disable ANSI (or `NO_COLOR=1`) |
| `-v` | off | Debug logging |

## How it works

1. **Topology.** Tokenize `\b[a-zA-Z0-9']+\b|[.!?]` lowercased, sentence-split `(?<=[.!?])\s+`. Build **causal** left counts `left_counts[n][ctx][word]` for `n=1..3` (right side kept for legacy diffusion). `mllm51/topology.py` / `MLLM-5.1.py`.

2. **Autocomplete.** Given prefix tokens `w_{<t}`, score each candidate continuation by `log(count/total+FLOOR)*n + log(unigram+1)*0.1`, softmax → probs, sample `p^(1/T)` with `random.Random(seed)`. Candidates = `sorted(observed ∪ top50)` so `PYTHONHASHSEED`-independent. Threshold gates low-confidence ghosts.

3. **Diffusion heritage.** `denoise()` still available for `generate --show-steps`: start `[MASK]*len`, annealed `1.2*(1-t/steps)+0.2`, re-mask lowest `1-t/steps`. Autocomplete is left-to-right specialization of the same topology.

## Project layout

```
MLLM-5.1.py        ← self-contained autocomplete runner (run it)
index.html         ← self-contained playground (ghost-text editor) — also playground.html
mllm51/            ← installable package (same engine, causal + diffusion)
  topology.py        n-gram topology
  engine.py          DiscreteDiffusionEngine + causal complete()
  cli.py             CLI (autocomplete/generate/chat)
  terminal.py        ANSI Term
data/corpus.txt    ← source for BUILT_IN_CORPUS
tests/             ← 29 pytest
netlify.toml       ← static deploy config
```

## Deploy

Static — no build. Published on Netlify:

- **Prod:** https://mllm-5-1.netlify.app
- **Mirror:** https://mllm-5-synapse.netlify.app

Redeploy:

```bash
npx netlify deploy --prod --dir=. --site mllm-5-1  # or --site mllm-5-synapse
# or link first: npx netlify link --id <siteId>
```

`netlify.toml`:
```toml
[build]
  publish = "."
  command = "echo 'no build - static deploy'"
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -v
python MLLM-5.1.py --seed 42 autocomplete "hello" --steps 8 --temperature 0.2
```

## Models

- **MLLM-5.1** — 5.1 final (autocomplete, causal). No longer preview.
- Future: Tasmania 1072P · Meridian 200P · Fjord 50P · Mesa 10P (placeholders)

---
*Core principles preserved: causal n-gram topology, autocomplete ghost-text, confidence-gated, deterministic.*
