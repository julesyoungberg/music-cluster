# Music Cluster

A Python-based CLI tool for analyzing, clustering, and classifying music tracks using audio feature extraction and machine learning.

## Features

- **Comprehensive Audio Analysis**: Extracts timbral, rhythmic (BPM), harmonic, and loudness features
- **Multiple Clustering Algorithms**: K-means, Hierarchical (Ward linkage), and HDBSCAN (density-based)
- **Genre-Aware Naming**: Auto-generates cluster names with genre, BPM, and characteristics
- **Flexible Naming Schemes**: Configure what to include in names (genre, BPM, descriptors)
- **Smart Search**: Find tracks and see which clusters they belong to
- **Fast Classification**: Classify new tracks in < 1 second
- **Playlist Generation**: Export clusters as M3U playlists with representative tracks
- **Detailed Statistics**: View cluster size distributions, quality metrics, and more

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
# 1. Initialize the database
music-cluster init

# 2. Analyze your music library (extracts BPM, energy, spectral features, etc.)
music-cluster analyze ~/Music/techno --recursive

# 3. Create fine-grained clusters
music-cluster cluster --name techno_detailed --clusters 15 --show-metrics

# 4. Auto-label clusters with genres and BPM ranges
music-cluster label-clusters techno_detailed

# 5. View your organized clusters
music-cluster show techno_detailed

# 6. Search for specific artists
music-cluster search "Aphex Twin" --clustering techno_detailed

# 7. View detailed statistics
music-cluster stats techno_detailed

# 8. Export as playlists
music-cluster export --output ~/Music/playlists

# 9. Classify new tracks
music-cluster classify ~/Downloads/NewAlbum --recursive
```

**Note:** If the `music-cluster` command isn't found, make sure you've installed the package with `pip install -e .` (see [INSTALL.md](INSTALL.md)). You can also use `python -m music_cluster.cli` instead of `music-cluster`.

## Desktop UI

Music Cluster includes a modern desktop GUI built with Tauri and Svelte.

### Quick Start (Single Command)

Start both the API server and UI with one command:

**Option 1: Shell script (macOS/Linux)**
```bash
./start-dev.sh
```

**Option 2: Python script (Cross-platform)**
```bash
python start-dev.py
```

**Option 3: npm script (from ui directory)**
```bash
cd ui
npm install  # First time only
npm run start:all
```

### Manual Start

1. **Start the API server:**
   ```bash
   uvicorn music_cluster.api:app --reload --port 8000
   ```

2. **In a separate terminal, start the UI:**
   ```bash
   cd ui
   npm install
   npm run dev
   ```

3. **Or run as a Tauri desktop app:**
   ```bash
   cd ui
   npm install
   npm run tauri:dev
   ```

### Building the Desktop App

```bash
cd ui
npm install
npm run tauri:build
```

This creates native executables in `src-tauri/target/release/bundle/`.

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
# Auto-detect optimal number of clusters (k-means)
music-cluster cluster --name "auto_2024" --show-metrics

# Use different algorithms
music-cluster cluster --name "hierarchical_test" --algorithm hierarchical --clusters 10
music-cluster cluster --name "density_based" --algorithm hdbscan --min-size 20

# Use granularity control for intuitive adjustment
music-cluster cluster --granularity fewer    # Broader clusters
music-cluster cluster --granularity more     # More specific clusters
music-cluster cluster --granularity finer    # Very fine-grained clusters

# Specify exact number of clusters
music-cluster cluster --clusters 25 --show-metrics
```

### Label Clusters

Auto-generate descriptive names with genre classification:

```bash
# Generate names with genre, BPM ranges, and characteristics
music-cluster label-clusters my_clustering

# Preview names before applying (dry run)
music-cluster label-clusters my_clustering --dry-run

# Customize naming scheme
music-cluster label-clusters my_clustering --no-bpm          # Skip BPM
music-cluster label-clusters my_clustering --bpm-average     # Use average instead of range
music-cluster label-clusters my_clustering --no-descriptors  # Just genre + BPM

# Manually rename specific clusters
music-cluster rename-cluster my_clustering 5 "Deep Minimal Techno"
```

**Example Generated Names:**
- `Tech House 95-145 BPM Complex`
- `Techno 130 BPM Bass-Heavy`
- `Deep House 120-125 BPM Dark`
- `Drum & Bass 160-170 BPM Bright`

### Search and Explore

Find tracks and analyze your collection:

```bash
# Search for tracks by artist or name
music-cluster search "K-LONE" --clustering my_clustering

# View detailed statistics
music-cluster stats my_clustering

# Compare two clusterings side-by-side
music-cluster compare clustering1 clustering2

# View tracks in a specific cluster
music-cluster describe 42
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

1. **Feature Extraction**: Analyzes audio files using librosa to extract ~96 dimensional feature vectors including:
   - Timbral features (MFCCs, spectral centroid, rolloff, contrast, zero-crossing rate)
   - Rhythmic features (BPM/tempo, onset strength statistics)
   - Harmonic features (chroma/pitch class profile)
   - Dynamic features (RMS energy, spectral bandwidth)

2. **Clustering**: Supports multiple algorithms:
   - **K-means**: Fast, works well with spherical clusters
   - **Hierarchical (Ward)**: Better for nested/hierarchical structures
   - **HDBSCAN**: Density-based, finds clusters of varying densities without specifying count

3. **Genre Classification**: Uses BPM, bass presence, energy, and spectral characteristics to classify:
   - Electronic/Dance: Techno, House, Tech House, Deep House, Drum & Bass, Dubstep, etc.
   - Auto-generates descriptive names with configurable schemes

4. **Classification**: New tracks are classified by finding the nearest cluster centroid in feature space

5. **Export**: Generates M3U playlists with representative tracks that best represent each cluster

## Performance

- **Analysis**: ~1-2 seconds per track
- **10,000 track library**: 3-5 hours initial analysis (with multiprocessing)
- **Clustering**: Minutes for large libraries
- **Classification**: < 1 second per track
- **Storage**: ~1-2 KB per track

## Supported Audio Formats

The tool supports **all audio formats that FFmpeg can decode**, including:

**Compressed formats:**
- MP3, AAC, M4A, OGG (Vorbis), Opus, WMA

**Lossless formats:**
- FLAC, WAV, AIFF/AIF, APE (Monkey's Audio), ALAC (Apple Lossless), WavPack (WV)

**Mixed format libraries are fully supported** - you can analyze collections with different file types together.

## Requirements

- Python 3.9+
- FFmpeg (for audio decoding)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
