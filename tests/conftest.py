"""Shared test fixtures for music-cluster tests."""
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_features():
    """Generate sample audio features for testing."""
    # 96-dimensional feature vector matching librosa extraction
    features = np.random.rand(96)
    # Set specific BPM (index -10) for testing
    features[-10] = 120.0
    return features


@pytest.fixture
def sample_features_batch():
    """Generate a batch of sample features for clustering tests."""
    n_samples = 50
    features = np.random.rand(n_samples, 96)
    # Set BPMs to reasonable range
    features[:, -10] = np.random.uniform(80, 180, n_samples)
    return features


@pytest.fixture
def mock_db_path(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test_music.db"


@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data for testing."""
    n_clusters = 5
    n_samples = 50
    
    labels = np.random.randint(0, n_clusters, n_samples)
    features = np.random.rand(n_samples, 96)
    features[:, -10] = np.random.uniform(80, 180, n_samples)  # BPMs
    
    tracks = []
    for i in range(n_samples):
        tracks.append({
            'id': i + 1,
            'file_path': f'/test/track_{i}.mp3',
            'filename': f'track_{i}.mp3',
            'cluster_id': int(labels[i])
        })
    
    return {
        'labels': labels,
        'features': features,
        'tracks': tracks,
        'n_clusters': n_clusters
    }
