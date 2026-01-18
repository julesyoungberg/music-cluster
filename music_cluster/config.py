"""Configuration management for music-cluster."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


DEFAULT_CONFIG = {
    "database": {
        "path": "~/.music-cluster/library.db"
    },
    "feature_extraction": {
        "sample_rate": 44100,
        "frame_size": 2048,
        "hop_size": 1024,
        "mfcc_coefficients": 20,
        "analysis_version": "1.0.0"
    },
    "clustering": {
        "default_algorithm": "hdbscan",
        "auto_detect_k": True,
        "default_granularity": "normal",
        "min_clusters": 5,
        "max_clusters": 100,
        "min_cluster_size": 3,
        "detection_method": "silhouette"
    },
    "export": {
        "playlist_format": "m3u",
        "relative_paths": False,
        "include_representative": True
    },
    "performance": {
        "batch_size": 100,
        "num_workers": -1,  # -1 = use all CPUs
        "cache_features": True
    }
}


class Config:
    """Configuration manager for music-cluster."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Optional path to config file. If None, uses default location.
        """
        if config_path is None:
            config_path = os.path.expanduser("~/.music-cluster/config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    user_config = yaml.safe_load(f) or {}
                # Merge with defaults (user config overrides defaults)
                return self._merge_configs(DEFAULT_CONFIG, user_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration.")
                return DEFAULT_CONFIG.copy()
        else:
            return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with default config.
        
        Args:
            default: Default configuration
            user: User configuration
            
        Returns:
            Merged configuration
        """
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self) -> None:
        """Save current configuration to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value by key path.
        
        Args:
            *keys: Keys to traverse (e.g., 'database', 'path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_db_path(self) -> str:
        """Get the database path with expansion.
        
        Returns:
            Expanded database path
        """
        db_path = self.get("database", "path", default="~/.music-cluster/library.db")
        return os.path.expanduser(db_path)
    
    @staticmethod
    def create_default_config(config_path: Optional[str] = None) -> "Config":
        """Create and save default configuration file.
        
        Args:
            config_path: Optional path for config file
            
        Returns:
            Config instance
        """
        config = Config(config_path)
        config.save()
        return config
