# Music Cluster

A Python-based CLI tool for analyzing, clustering, and classifying music tracks using audio feature extraction and machine learning.

## Features

- **Comprehensive Audio Analysis**: Extracts timbral, rhythmic, harmonic, and loudness features using Essentia
- **Intelligent Clustering**: K-means clustering with automatic cluster count detection
- **Fast Classification**: Classify new tracks in < 1 second
- **Playlist Generation**: Export clusters as M3U playlists with representative tracks
- **Flexible Control**: Use exact cluster counts or intuitive granularity levels (fewer/less/normal/more/finer)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-cluster.git
cd music-cluster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```bash
# Initialize the database
python -m music_cluster.cli init
# Or if music-cluster command is installed:
# music-cluster init

# Analyze your music library
music-cluster analyze ~/Music/Library --recursive

# Create clusters with auto-detection
music-cluster cluster --name "my_clusters"

# Export playlists
music-cluster export --output ~/Music/Playlists

# Classify new tracks
music-cluster classify ~/Downloads/NewAlbum --recursive
```

## Usage

### Initialize

Create the database and configuration file:

```bash
music-cluster init
```

### Analyze Music Library

Extract audio features from your music collection:

```bash
# Analyze a directory recursively
music-cluster analyze ~/Music --recursive

# Analyze specific file types
music-cluster analyze ~/Music -r --extensions mp3,flac,wav

# Re-analyze existing tracks
music-cluster analyze ~/Music -r --update

# Use multiple workers for faster processing
music-cluster analyze ~/Music -r --workers 8
```

### Create Clusters

Cluster your music by similarity:

```bash
# Auto-detect optimal number of clusters
music-cluster cluster --name "auto_2024"

# Use granularity control for intuitive adjustment
music-cluster cluster --granularity fewer    # Broader clusters
music-cluster cluster --granularity more     # More specific clusters
music-cluster cluster --granularity finer    # Very fine-grained clusters

# Specify exact number of clusters
music-cluster cluster --clusters 25

# Show clustering quality metrics
music-cluster cluster --show-metrics
```

### Export Playlists

Generate M3U playlists from clusters:

```bash
# Export to default location (./playlists)
music-cluster export

# Export to specific directory
music-cluster export --output ~/Music/Playlists/Clusters

# Use relative paths in playlists
music-cluster export --relative-paths

# Export as JSON
music-cluster export --format json
```

### Classify New Tracks

Quickly classify new music:

```bash
# Classify a single track
music-cluster classify ~/Downloads/new_song.mp3

# Classify an entire directory
music-cluster classify ~/Downloads/NewAlbum --recursive

# Add classified tracks to existing playlists
music-cluster classify ~/Downloads/NewAlbum -r --export
```

### View Information

Inspect your database and clusters:

```bash
# Show database statistics
music-cluster info

# List all clusterings
music-cluster list

# Show clusters in a specific clustering
music-cluster show my_clusters

# Show tracks in a specific cluster
music-cluster describe 5
```

## Configuration

Configuration file location: `~/.music-cluster/config.yaml`

```yaml
database:
  path: ~/.music-cluster/library.db

feature_extraction:
  sample_rate: 44100
  frame_size: 2048
  hop_size: 1024
  mfcc_coefficients: 20
  analysis_version: "1.0.0"

clustering:
  default_algorithm: kmeans
  auto_detect_k: true
  default_granularity: normal
  min_clusters: 5
  max_clusters: 100
  min_cluster_size: 3
  detection_method: silhouette

export:
  playlist_format: m3u
  relative_paths: false
  include_representative: true

performance:
  batch_size: 100
  num_workers: -1  # -1 = use all CPUs
  cache_features: true
```

## How It Works

1. **Feature Extraction**: Analyzes audio files using Essentia to extract ~100-150 dimensional feature vectors covering timbral, rhythmic, harmonic, and loudness characteristics
2. **Clustering**: Uses K-means clustering with silhouette analysis to automatically determine the optimal number of clusters
3. **Classification**: New tracks are classified by finding the nearest cluster centroid in feature space
4. **Export**: Generates playlists with representative tracks that best represent each cluster

## Performance

- **Analysis**: ~1-2 seconds per track
- **10,000 track library**: 3-5 hours initial analysis (with multiprocessing)
- **Clustering**: Minutes for large libraries
- **Classification**: < 1 second per track
- **Storage**: ~1-2 KB per track

## Requirements

- Python 3.9+
- Audio file formats: MP3, FLAC, WAV, M4A, OGG

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
