"""The synthetic music library the end-to-end tests run against.

Three families of pulsed tones, tagged like real DJ files. They are as
distinguishable to the feature extractor as three genre folders are to a DJ,
which is what makes an end-to-end assertion about sorting mean anything.

Importable without pytest so the browser tests can seed the same library from
a plain script.
"""

from pathlib import Path
from typing import Optional

import numpy as np


SAMPLE_RATE = 22050
SECONDS = 4.0

# (folder, carrier frequency, pulse rate, genre tag, tag BPM)
FAMILIES = [
    ("Deep House", 110.0, 2.0, "Deep House", 122),
    ("Techno", 440.0, 4.0, "Techno", 132),
    ("Drum & Bass", 1800.0, 8.0, "Drum & Bass", 174),
]

TRACKS_PER_FOLDER = 8
NEW_TRACKS_PER_FAMILY = 2


def write_track(path: Path, freq: float, pulse: float, jitter: float = 0.0) -> str:
    """A pulsed tone with enough structure for every feature family to exist."""
    import soundfile

    time = np.linspace(0, SECONDS, int(SAMPLE_RATE * SECONDS), endpoint=False)
    envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * pulse * time)))
    carrier = np.sin(2 * np.pi * freq * (1 + jitter) * time)
    # A second partial gives the spectral shape something to vary with.
    overtone = 0.3 * np.sin(2 * np.pi * freq * 2.5 * (1 + jitter) * time)
    signal = envelope * (carrier + overtone) * 0.4

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), signal.astype(np.float32), SAMPLE_RATE)
    return str(path)


def tag(path: str, artist: str, title: str, genre: str, bpm: int) -> None:
    from mutagen.flac import FLAC

    audio = FLAC(path)
    audio["artist"] = [artist]
    audio["title"] = [title]
    audio["genre"] = [genre]
    audio["bpm"] = [str(bpm)]
    audio.save()


def build_library(root: Path, seed: int = 20240501) -> Path:
    """Write ``<root>/DJ/<genre>/*.flac`` and ``<root>/NewMusic/*.flac``."""
    rng = np.random.default_rng(seed)

    for folder, freq, pulse, genre, bpm in FAMILIES:
        for index in range(TRACKS_PER_FOLDER):
            path = root / "DJ" / folder / f"{folder} Artist - Track {index + 1}.flac"
            write_track(path, freq, pulse, jitter=float(rng.normal(0, 0.01)))
            tag(str(path), f"{folder} Artist", f"Track {index + 1}", genre, bpm)

    for folder, freq, pulse, genre, bpm in FAMILIES:
        for index in range(NEW_TRACKS_PER_FAMILY):
            path = root / "NewMusic" / f"New {folder} {index + 1}.flac"
            write_track(path, freq, pulse, jitter=float(rng.normal(0, 0.01)))
            tag(str(path), "New Artist", f"New {folder} {index + 1}", genre, bpm)

    return root


def import_and_fit(
    db_path: str, config_path: str, music_root: Path, collection_name: str = "Crates"
) -> Optional[dict]:
    """Import the genre folders as groups and fit the sorter, as `init` + `fit` would."""
    from music_cluster import groups as groups_mod
    from music_cluster.config import Config
    from music_cluster.database import Database

    config = Config(config_path)
    config.set(["database", "path"], db_path)
    config.set(["feature_extraction", "excerpt_seconds"], 0)
    config.save()

    db = Database(db_path)
    collection = groups_mod.ensure_collection(db, collection_name, "End-to-end fixture")
    groups_mod.import_folders_as_groups(
        db, collection["id"], str(music_root / "DJ"), config, workers=1
    )
    return groups_mod.fit_collection(db, db.get_collection(collection_id=collection["id"]), config)
