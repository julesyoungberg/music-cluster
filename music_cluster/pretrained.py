"""Pretrained audio embeddings: a network's description of how a track sounds.

The sorter's 96 hand-made features cap genre sorting on FMA at about 45% top-1;
the AST embedding below reaches 0.646 on the same simulated libraries
(EVALUATION.md). It is optional: set `sorting.feature_set: ast` and install
PyTorch and Hugging Face `transformers` (`pip install "cratewise[ast]"`). The
model's weights (about 350 MB) download on first use.

Each track's embedding is the mean over three windows from a centred excerpt,
so a long DJ track and a 30-second clip are both described by their middle
rather than an intro.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)

# Clips in, one row per clip out.
Embedder = Callable[[List[np.ndarray]], np.ndarray]

WINDOWS = 3
BATCH = 8


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    sample_rate: int
    window: float
    licence: str
    about: str


MODELS: Dict[str, ModelSpec] = {
    "ast": ModelSpec(
        name="ast",
        repo="MIT/ast-finetuned-audioset-10-10-0.4593",
        sample_rate=16000,
        window=10.24,
        licence="BSD-3-Clause",
        about="Audio Spectrogram Transformer fine-tuned on AudioSet, 768 dimensions",
    ),
}


def spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding {name!r}; choose from {', '.join(sorted(MODELS))}"
        ) from None


class MissingDependency(RuntimeError):
    """PyTorch or transformers is not installed."""


def require() -> None:
    """Fail now, before hours of analysis, if the models cannot be loaded."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependency(
            'Pretrained embeddings need PyTorch and transformers: pip install "cratewise[ast]"'
        ) from exc


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check(name: str) -> None:
    """Load the model and embed a second of silence, so a broken setup fails
    before the hour of feature analysis that runs first."""
    model = spec(name)
    require()
    rows = np.asarray(_loaded(name)([np.zeros(model.sample_rate, dtype=np.float32)]))
    if rows.ndim != 2 or rows.shape[0] != 1:
        raise MissingDependency(f"{model.repo} returned shape {rows.shape} for one clip")


@functools.cache
def _loaded(name: str) -> Embedder:
    return load_model(spec(name))


def load_model(model: ModelSpec) -> Embedder:
    """A callable that embeds a batch of clips at the model's sample rate."""
    require()
    import torch
    import transformers

    device = _device()
    if model.name == "ast":
        extractor = transformers.ASTFeatureExtractor.from_pretrained(model.repo)
        net = transformers.ASTModel.from_pretrained(model.repo).to(device).eval()

        def embed_ast(clips: List[np.ndarray]) -> np.ndarray:
            inputs = extractor(clips, sampling_rate=model.sample_rate, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = net(**inputs).pooler_output
            return out.float().cpu().numpy()

        return embed_ast

    raise ValueError(f"No loader for {model.name!r}")


def windows(path: str, model: ModelSpec) -> Optional[List[np.ndarray]]:
    """Up to three consecutive windows from the middle of the file, or None."""
    import librosa

    span = model.window * WINDOWS
    try:
        try:
            duration = float(librosa.get_duration(path=path))
        except Exception:
            duration = 0.0
        offset = max(0.0, (duration - span) / 2) if duration > span else 0.0
        y, _ = librosa.load(path, sr=model.sample_rate, mono=True, offset=offset, duration=span)
    except Exception as exc:
        logger.warning("Could not decode %s: %s", path, exc)
        return None
    if y is None or len(y) < model.sample_rate:
        return None
    size = int(model.window * model.sample_rate)
    clips = [y[i : i + size] for i in range(0, len(y), size)]
    # A short tail is noise next to the full windows; keep it only if it is all there is.
    full = [clip for clip in clips if len(clip) >= size // 2]
    return [np.ascontiguousarray(c, dtype=np.float32) for c in (full or clips[:1])]


def _pool(rows: np.ndarray) -> np.ndarray:
    """Mean of unit-length window embeddings, so one loud window can't dominate."""
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms > 0, norms, 1.0)).mean(axis=0)


def embed_files(
    paths: Sequence[str], model_name: str, embedder: Optional[Embedder] = None
) -> List[Optional[np.ndarray]]:
    """One embedding per file, in order; None where a file could not be decoded.

    Args:
        embedder: Use this instead of loading the named model (for tests).
    """
    model = spec(model_name)
    run = embedder or _loaded(model.name)
    out: List[Optional[np.ndarray]] = []
    for start in range(0, len(paths), BATCH):
        decoded = [windows(path, model) for path in paths[start : start + BATCH]]
        clips = [clip for found in decoded if found for clip in found]
        rows = np.asarray(run(clips), dtype=float) if clips else np.zeros((0, 0))
        if rows.shape[0] != len(clips):
            # A wrong count would pool one track's windows into another's vector.
            raise RuntimeError(f"{model.repo} returned {rows.shape[0]} rows for {len(clips)} clips")
        cursor = 0
        for found in decoded:
            if not found:
                out.append(None)
                continue
            out.append(_pool(rows[cursor : cursor + len(found)]))
            cursor += len(found)
    return out
