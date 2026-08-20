"""Diff-MLLM-5.1 — a toy discrete-diffusion language model over n-gram statistics."""

from .engine import MASK, DenoiseResult, DiscreteDiffusionEngine
from .topology import BidirectionalTopology, tokenize

__version__ = "5.1.1"

__all__ = [
    "MASK",
    "BidirectionalTopology",
    "DenoiseResult",
    "DiscreteDiffusionEngine",
    "tokenize",
    "__version__",
]
