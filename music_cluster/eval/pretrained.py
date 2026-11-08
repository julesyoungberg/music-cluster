"""Pretrained audio embeddings, measured as an extra feature set.

The sorter's 96 hand-made features look like the ceiling on every sorting
suite: on FMA the best classifier on them gets about 45%, and the confusions
on a real DJ library (deep vs future dub, club vs garage) are about texture
and style, not tempo or key. A network trained on a lot of music might
separate those. This module computes such an embedding for the eval datasets
so `cratewise eval run --embedding NAME` can score it with the same
simulated libraries as the hand features: the embedding on its own, and
appended to the hand features.

Nothing here is used by the app. The models need PyTorch and Hugging Face
`transformers` (`pip install torch transformers`), download their weights
on first use, and are only imported when an embedding is asked for.

Each track's embedding is the mean over three windows from a centred excerpt,
so a long DJ track and a 30-second dataset clip are both described by their
middle rather than an intro. Results are cached per file like the analysis
cache, keyed by the model, so a second run is instant.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..library import resolve_worker_count
from .analysis import _stamp
from .datasets import Item, data_dir


logger = logging.getLogger(__name__)

Progress = Callable[[int, int], None]
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
    "clap": ModelSpec(
        name="clap",
        repo="laion/larger_clap_music",
        sample_rate=48000,
        window=10.0,
        licence="Apache-2.0",
        about="LAION CLAP trained on music and text, 512 dimensions",
    ),
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
            "Pretrained embeddings need PyTorch and transformers: pip install torch transformers"
        ) from exc


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model: ModelSpec) -> Embedder:
    """A callable that embeds a batch of clips at the model's sample rate."""
    require()
    import torch
    import transformers

    device = _device()
    if model.name == "clap":
        processor = transformers.ClapProcessor.from_pretrained(model.repo)
        net = transformers.ClapModel.from_pretrained(model.repo).to(device).eval()

        def embed_clap(clips: List[np.ndarray]) -> np.ndarray:
            try:
                inputs = processor(
                    audio=clips, sampling_rate=model.sample_rate, return_tensors="pt"
                )
            except TypeError:
                # Older transformers name the argument `audios`.
                inputs = processor(
                    audios=clips, sampling_rate=model.sample_rate, return_tensors="pt"
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = net.get_audio_features(**inputs)
            return out.float().cpu().numpy()

        return embed_clap

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


def cache_path(cache_name: str, model: str, root: Optional[str] = None) -> Path:
    return data_dir(root) / ".cache" / f"{cache_name}.{model}.jsonl"


def embed(
    items: Sequence[Item],
    model_name: str,
    cache_name: str,
    root: Optional[str] = None,
    workers: int = -1,
    progress: Optional[Progress] = None,
    embedder: Optional[Embedder] = None,
) -> List[Optional[np.ndarray]]:
    """One embedding per item, in order; None where a file could not be decoded.

    Args:
        embedder: Use this instead of loading the named model (for tests).
    """
    model = spec(model_name)
    path = cache_path(cache_name, model.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = f"{model.repo}|{model.window}|{WINDOWS}"

    cached: Dict[str, np.ndarray] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("signature") == signature:
                    cached[entry["key"]] = np.asarray(entry["vector"], dtype=float)

    keys: List[str] = []
    todo: List[Item] = []
    for item in items:
        try:
            key = f"{item.path}|{_stamp(item.path)}"
        except OSError:
            key = f"{item.path}|missing"
        keys.append(key)
        if key not in cached and not key.endswith("|missing"):
            todo.append(item)

    if todo:
        run = embedder or load_model(model)
        total = len(todo)
        done = 0
        with (
            open(path, "a", encoding="utf-8") as out,
            ThreadPoolExecutor(max_workers=resolve_worker_count(workers)) as pool,
        ):
            # Decoding runs ahead on threads while the model works through batches.
            for start in range(0, total, BATCH):
                batch = todo[start : start + BATCH]
                decoded = list(pool.map(lambda item: windows(item.path, model), batch))
                clips = [clip for found in decoded if found for clip in found]
                rows = np.asarray(run(clips), dtype=float) if clips else np.zeros((0, 0))
                cursor = 0
                for item, found in zip(batch, decoded, strict=True):
                    done += 1
                    if progress:
                        progress(done, total)
                    if not found:
                        continue
                    vector = _pool(rows[cursor : cursor + len(found)])
                    cursor += len(found)
                    key = f"{item.path}|{_stamp(item.path)}"
                    cached[key] = vector
                    out.write(
                        json.dumps(
                            {
                                "key": key,
                                "signature": signature,
                                "vector": [round(float(v), 6) for v in vector],
                            }
                        )
                        + "\n"
                    )

    return [cached.get(key) for key in keys]
