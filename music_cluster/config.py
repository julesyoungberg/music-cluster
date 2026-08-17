"""Configuration management for music-cluster."""

import copy
import os
from typing import Any, Dict, Optional

import yaml

from . import profiles


DEFAULT_CONFIG: Dict[str, Any] = {
    "database": {
        "path": "~/.music-cluster/library.db",
    },
    "feature_extraction": {
        "sample_rate": 44100,
        "frame_size": 2048,
        "hop_size": 1024,
        "mfcc_coefficients": 20,
        "analysis_version": "2.0.0",
        # Analysing the middle of a track is both faster and more representative
        # than intros/outros. 0 disables excerpting.
        "excerpt_seconds": 90,
    },
    "sorting": {
        # How the reference space is built before distances are measured.
        # auto | none | pca | lda | nca
        "projection": "auto",
        "pca_variance": 0.95,
        "max_components": 40,
        # Share of the space devoted to what separates the existing groups.
        "discriminant_weight": 0.7,
        # Distance to a group blends its k nearest references with its centroid.
        "neighbors": 5,
        "knn_weight": 0.7,
        # Per-group distances are divided by that group's own spread so a broad
        # 900-track folder does not out-compete a tight 8-track seed group.
        "scale_normalization": True,
        "temperature": 0.5,
        # Decision thresholds.
        "auto_accept_confidence": 0.6,
        "min_margin": 0.08,
        "novelty_factor": 3.0,
        # Relative importance of each feature family.
        "feature_weights": {
            "timbre": 1.0,
            "rhythm": 1.0,
            "harmony": 1.0,
            "dynamics": 1.0,
        },
        # Add accepted tracks to their group as new references on commit.
        "learn_on_commit": True,
    },
    "discovery": {
        "algorithm": "hdbscan",
        "granularity": "normal",
        "min_group_size": 8,
        "max_candidates": 30,
        "exemplars_per_candidate": 5,
        "detection_method": "silhouette",
    },
    "organize": {
        # playlist | copy | move | symlink
        "mode": "playlist",
        "playlist_format": "m3u8",
        "playlist_dir": "~/.music-cluster/playlists",
        "relative_paths": False,
        "on_conflict": "skip",  # skip | rename | overwrite
        "dry_run_first": True,
    },
    "labeling": {
        # Optional LLM assist for naming discovered groups.
        "llm_enabled": False,
        "llm_model": "claude-sonnet-5",
        "llm_api_key_env": "ANTHROPIC_API_KEY",
    },
    "performance": {
        "batch_size": 100,
        "num_workers": -1,  # -1 = use all CPUs
    },
}


class Config:
    """Layered configuration: defaults overridden by a user YAML file."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.environ.get(
                "MUSIC_CLUSTER_CONFIG", os.path.expanduser("~/.music-cluster/config.yaml")
            )

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        defaults = copy.deepcopy(DEFAULT_CONFIG)
        if not os.path.exists(self.config_path):
            return defaults
        try:
            with open(self.config_path, "r") as handle:
                user_config = yaml.safe_load(handle) or {}
            return self._merge_configs(defaults, user_config)
        except Exception as exc:
            print(f"Warning: failed to load config from {self.config_path}: {exc}")
            print("Using default configuration.")
            return defaults

    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        result = dict(default)
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def save(self) -> None:
        directory = os.path.dirname(self.config_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.config_path, "w") as handle:
            yaml.dump(self.config, handle, default_flow_style=False, sort_keys=False)

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, keys: list, value: Any) -> None:
        """Set a nested value by key path, creating intermediate dicts."""
        target = self.config
        for key in keys[:-1]:
            if not isinstance(target.get(key), dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def section(self, name: str, profile: Optional[str] = None) -> Dict[str, Any]:
        """A copy of one top-level section, resolved for an audio profile.

        Three layers, each overriding the one before:

        1. the section as configured (defaults plus the user's YAML),
        2. the profile's own defaults, for the handful of settings where
           samples genuinely need different values than tracks,
        3. the user's per-profile overrides under ``<section>.profiles.<name>``,
           which is how a profile default gets argued with.
        """
        resolved = copy.deepcopy(self.get(name, default={}) or {})
        overrides = resolved.pop("profiles", None) or {}

        profile_name = profiles.normalize(profile)
        resolved = self._merge_configs(resolved, profiles.get(profile_name).overrides(name))

        user_overrides = overrides.get(profile_name) if isinstance(overrides, dict) else None
        if isinstance(user_overrides, dict):
            resolved = self._merge_configs(resolved, user_overrides)

        return resolved

    def get_db_path(self) -> str:
        path = os.environ.get("MUSIC_CLUSTER_DB") or self.get(
            "database", "path", default="~/.music-cluster/library.db"
        )
        return os.path.expanduser(path)

    def get_cache_dir(self) -> str:
        """Where derived data (waveforms) lives — beside the database by default."""
        path = os.environ.get("MUSIC_CLUSTER_CACHE") or self.get(
            "database", "cache_dir", default=None
        )
        if not path:
            path = os.path.join(os.path.dirname(self.get_db_path()), "cache")
        return os.path.expanduser(path)

    def extractor_config(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Keyword arguments for :class:`~music_cluster.extractor.FeatureExtractor`."""
        section = self.section("feature_extraction", profile)
        return {
            "sample_rate": section.get("sample_rate", 44100),
            "frame_size": section.get("frame_size", 2048),
            "hop_size": section.get("hop_size", 1024),
            "n_mfcc": section.get("mfcc_coefficients", 20),
            "excerpt_seconds": section.get("excerpt_seconds", 90),
            "max_seconds": section.get("max_seconds"),
            "profile": profiles.normalize(profile),
        }

    @staticmethod
    def create_default_config(config_path: Optional[str] = None) -> "Config":
        config = Config(config_path)
        config.save()
        return config
