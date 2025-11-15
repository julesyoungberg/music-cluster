# Music Clustering CLI Tool - Technical Specification

## Overview

A Python-based CLI tool for analyzing, clustering, and classifying music tracks using audio feature extraction and machine learning. The tool processes directories with thousands of tracks, generates playlists based on similarity, and enables fast classification of new tracks.

---

## Architecture

### Core Components

1. **Feature Extraction Service** - Essentia-based audio analysis
2. **Storage Layer** - SQLite database with feature vectors
3. **Clustering Engine** - K-means with automatic cluster count detection
4. **Classification Service** - Fast nearest-neighbor lookup
5. **CLI Interface** - Command-line interface using Click
6. **Export Module** - M3U playlist generation with representative tracks

---

## Technology Stack

### Core Libraries

- **Audio Analysis**: `essentia` or `essentia-tensorflow` for feature extraction
- **Database**: SQLite with potential `sqlite-vss` for vector similarity
- **Machine Learning**: `scikit-learn` for clustering and evaluation
- **CLI Framework**: `Click` for command-line interface
- **Audio I/O**: `librosa` or `soundfile` for audio loading
- **Progress**: `tqdm` for progress bars
- **Serialization**: `numpy` for feature vectors, `pickle` for model persistence

### Python Version

- Python 3.9+ (for modern type hints and performance)

---

## Feature Extraction

### Comprehensive Feature Set

**Timbral Features:**
- MFCCs (Mel-Frequency Cepstral Coefficients): 20 coefficients
- Spectral centroid (brightness)
- Spectral rolloff (frequency distribution)
- Spectral flux (texture)
- Spectral contrast (peaks vs valleys)
- Zero-crossing rate (noisiness)

**Rhythmic Features:**
- BPM (tempo)
- Beat strength/regularity
- Onset rate (note density)
- Rhythm patterns (Essentia rhythm descriptors)

**Harmonic Features:**
- Key and mode
- Chroma features (12-dimensional pitch class profile)
- Harmonic-to-noise ratio
- Dissonance

**Loudness & Dynamics:**
- RMS energy
- Dynamic range
- Loudness (LUFS or similar)

**High-Level Features:**
- Danceability
- Energy
- Acoustic vs electronic

**Aggregation Strategy:**
- Compute features over short frames (e.g., 50ms)
- Aggregate using mean and standard deviation
- Results in ~100-150 dimensional feature vector per track

### Normalization

- StandardScaler for zero mean, unit variance
- Store scaler parameters for consistent classification

---

## Database Schema

```sql
-- Track metadata
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    duration REAL,
    file_size INTEGER,
    checksum TEXT,  -- For detecting duplicates/changes
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_version TEXT  -- Track feature extraction version
);

-- Feature vectors (serialized)
CREATE TABLE features (
    track_id INTEGER PRIMARY KEY,
    feature_vector BLOB NOT NULL,  -- Pickled numpy array
    feature_dim INTEGER,
    normalized BOOLEAN DEFAULT 0,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Clustering metadata
CREATE TABLE clusterings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    algorithm TEXT DEFAULT 'kmeans',
    num_clusters INTEGER,
    parameters TEXT,  -- JSON of hyperparameters
    silhouette_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual clusters
CREATE TABLE clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clustering_id INTEGER NOT NULL,
    cluster_index INTEGER NOT NULL,
    size INTEGER,
    representative_track_id INTEGER,  -- Track closest to centroid
    centroid BLOB,  -- Pickled numpy array
    FOREIGN KEY (clustering_id) REFERENCES clusterings(id) ON DELETE CASCADE,
    FOREIGN KEY (representative_track_id) REFERENCES tracks(id),
    UNIQUE(clustering_id, cluster_index)
);

-- Cluster membership
CREATE TABLE cluster_members (
    cluster_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    distance_to_centroid REAL,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    PRIMARY KEY (cluster_id, track_id)
);

-- Index for fast lookups
CREATE INDEX idx_tracks_filepath ON tracks(filepath);
CREATE INDEX idx_cluster_members_track ON cluster_members(track_id);
```

---

## Clustering Implementation

### K-Means with Auto-Detection

**Cluster Count Detection:**
1. **Elbow Method**: Plot inertia vs k, find elbow point
2. **Silhouette Analysis**: Find k with highest average silhouette score
3. **Gap Statistic**: Compare with random data
4. **Calinski-Harabasz Index**: Ratio of between/within cluster variance

**Default Strategy:**
- Test k from `sqrt(n/2)` to `min(100, n/10)` where n = number of tracks
- Use silhouette score as primary metric
- Allow user override with `--clusters N` for exact count
- Support `--granularity` option for intuitive control (fewer|less|normal|more|finer)

**Granularity Multipliers:**
- `fewer`: 0.5x the optimal k (larger, broader clusters)
- `less`: 0.75x the optimal k
- `normal`: 1.0x the optimal k (default, pure auto-detection)
- `more`: 1.5x the optimal k
- `finer`: 2.0x the optimal k (smaller, more specific clusters)

The algorithm first determines the optimal k using silhouette analysis, then applies the granularity multiplier.

**Algorithm Configuration:**
```python
KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

### Cluster Quality Metrics

- Silhouette score (overall and per-cluster)
- Inertia (within-cluster sum of squares)
- Davies-Bouldin index (cluster separation)
- Cluster size distribution

---

## CLI Interface

### Commands

#### 1. Initialize
```bash
music-cluster init [--db-path PATH]
```
- Creates database at `~/.music-cluster/library.db` (or custom path)
- Initializes schema
- Creates config file

#### 2. Analyze
```bash
music-cluster analyze PATH [OPTIONS]

Options:
  --recursive, -r          Scan subdirectories
  --update, -u             Re-analyze existing tracks
  --extensions EXT         File extensions (default: mp3,flac,wav,m4a,ogg)
  --batch-size N           Process N tracks at a time (default: 100)
  --workers N              Parallel workers (default: CPU count)
  --skip-errors            Continue on corrupted files
```

**Behavior:**
- Scan directory for audio files
- Check database for existing analysis (by checksum)
- Extract features for new/changed tracks
- Display progress bar with ETA
- Save results incrementally (commit every batch)
- Print summary: tracks analyzed, skipped, errors

#### 3. Cluster
```bash
music-cluster cluster [OPTIONS]

Options:
  --name NAME              Name for this clustering
  --clusters N             Exact number of clusters (overrides auto-detection)
  --granularity LEVEL      Cluster granularity: fewer|less|normal|more|finer (default: normal)
  --min-size N             Minimum cluster size (default: 3)
  --max-clusters N         Maximum k to test during auto-detection (default: 100)
  --method METHOD          Detection method: silhouette|elbow|gap (default: silhouette)
  --show-metrics           Display clustering quality metrics
```

**Behavior:**
- Load all feature vectors from database
- Determine optimal k using specified method (unless `--clusters` is provided)
- Apply granularity multiplier to optimal k (if using `--granularity`)
- Run K-means clustering
- Calculate cluster centroids and representatives
- Save clustering to database
- Print summary with cluster sizes and quality score

**Note:** `--clusters` and `--granularity` are mutually exclusive. Use `--clusters` for exact control, or `--granularity` for relative adjustment.

#### 4. Export
```bash
music-cluster export [OPTIONS]

Options:
  --output DIR             Output directory (default: ./playlists)
  --format FORMAT          Playlist format: m3u|m3u8|json (default: m3u)
  --clustering NAME        Which clustering to export (default: latest)
  --relative-paths         Use relative paths in playlists
  --include-representative Include representative track first in playlist
```

**Behavior:**
- Create output directory if needed
- Generate one playlist file per cluster
- Playlist naming: `cluster_01_jazz.m3u` (with auto-generated labels if possible)
- Add comment header with cluster info and representative track
- Optional: export JSON with full cluster metadata

#### 5. Classify
```bash
music-cluster classify PATH [OPTIONS]

Options:
  --recursive, -r          Process directory recursively
  --clustering NAME        Use specific clustering (default: latest)
  --threshold DIST         Max distance for classification (default: none)
  --export                 Add to corresponding playlist files
  --show-features          Display extracted features
```

**Behavior:**
- Extract features from new track(s)
- Find nearest cluster centroid (Euclidean distance)
- Print results: cluster ID, distance, representative track
- Optionally append to existing playlist
- Fast operation: < 1 second per track

#### 6. Info & Stats
```bash
music-cluster info                    # Database stats
music-cluster list                    # List all clusterings
music-cluster show CLUSTERING_NAME    # Show clusters in a clustering
music-cluster describe CLUSTER_ID     # Show tracks in a cluster
```

### Configuration File

Location: `~/.music-cluster/config.yaml`

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
  default_granularity: normal  # fewer|less|normal|more|finer
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

---

## Project Structure

```
music-cluster/
├── music_cluster/
│   ├── __init__.py
│   ├── cli.py              # Click commands and entry point
│   ├── extractor.py        # Feature extraction with Essentia
│   ├── features.py         # Feature vector utilities
│   ├── clustering.py       # K-means and cluster detection
│   ├── classifier.py       # New track classification
│   ├── database.py         # SQLite operations and models
│   ├── exporter.py         # M3U playlist generation
│   ├── config.py           # Configuration management
│   └── utils.py            # Helpers (checksums, file handling)
├── tests/
│   ├── test_extractor.py
│   ├── test_clustering.py
│   ├── test_database.py
│   └── fixtures/           # Sample audio files
├── docs/
│   └── SPEC.md             # This file
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-3)

**Day 1: Project Setup**
- Initialize repository structure
- Set up virtual environment
- Install dependencies
- Create CLI skeleton with Click
- Implement config management

**Day 2: Database Layer**
- Implement SQLite schema
- Create database initialization
- Build CRUD operations for tracks and features
- Add transaction management

**Day 3: Feature Extraction**
- Implement Essentia-based feature extraction
- Create comprehensive feature set
- Add normalization pipeline
- Implement batch processing with progress bars
- Error handling for corrupted files

### Phase 2: Clustering (Days 4-5)

**Day 4: Clustering Engine**
- Implement K-means clustering
- Build cluster count auto-detection (silhouette, elbow)
- Calculate centroids and find representative tracks
- Store clustering results in database

**Day 5: Classification**
- Build fast nearest-neighbor classification
- Implement distance thresholding
- Add batch classification for directories

### Phase 3: CLI & Export (Days 6-7)

**Day 6: CLI Commands**
- Implement `analyze` command with multiprocessing
- Implement `cluster` command with auto-detection
- Implement `classify` command
- Add info/stats commands

**Day 7: Export & Polish**
- Build M3U playlist exporter
- Add representative track to playlists
- Implement relative/absolute path handling
- CLI output formatting and error messages

### Phase 4: Optimization (Days 8-9)

**Day 8: Performance**
- Add multiprocessing for feature extraction
- Implement feature caching (checksum-based)
- Optimize database queries with indexes
- Profile and optimize bottlenecks

**Day 9: Testing & Documentation**
- Write unit tests for core components
- Integration tests for CLI commands
- Update README with examples
- Add docstrings and type hints

---

## Performance Considerations

### Feature Extraction
- **Bottleneck**: Audio file I/O and Essentia processing
- **Solution**: Multiprocessing with worker pool
- **Expected**: ~1-2 seconds per track on modern CPU
- **For 10k tracks**: ~3-5 hours initial analysis

### Clustering
- **Bottleneck**: K-means iterations and k-selection
- **Solution**: Efficient numpy operations, limit max k
- **Expected**: Minutes for 10k tracks, linear scaling

### Classification
- **Bottleneck**: Feature extraction of new tracks
- **Solution**: Fast nearest-centroid lookup
- **Expected**: < 1 second per track (extraction + lookup)

### Database
- **Bottleneck**: Loading all feature vectors
- **Solution**: Lazy loading, indexed queries, optional vector DB
- **Storage**: ~1-2 KB per track (metadata + features)
- **For 10k tracks**: ~10-20 MB database

---

## Dependencies

### requirements.txt
```
essentia-tensorflow>=2.1b6.dev1110  # or essentia
librosa>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
click>=8.1.0
tqdm>=4.65.0
soundfile>=0.12.0
pyyaml>=6.0
```

### Optional Dependencies
```
sqlite-vss>=0.1.0      # Vector similarity search
matplotlib>=3.7.0      # Visualization
umap-learn>=0.5.0      # Dimensionality reduction
```

---

## Example Workflows

### Initial Setup & Analysis
```bash
# Initialize project
cd ~/workspace/music-cluster
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize database
music-cluster init

# Analyze entire library
music-cluster analyze ~/Music/Library --recursive

# This may take several hours for large libraries
# Progress bar will show ETA
```

### Create Clusters
```bash
# Auto-detect optimal number of clusters (default: normal granularity)
music-cluster cluster --name "auto_2024" --show-metrics

# Create fewer, broader clusters
music-cluster cluster --name "broad_clusters" --granularity fewer

# Create more, finer-grained clusters
music-cluster cluster --name "detailed_clusters" --granularity finer

# Or specify exact cluster count
music-cluster cluster --name "manual_20" --clusters 20

# Export playlists
music-cluster export --output ~/Music/Playlists/Clusters
```

### Classify New Music
```bash
# Classify a new album (fast)
music-cluster classify ~/Downloads/NewAlbum --recursive

# Output:
# NewAlbum/track1.mp3 -> Cluster 7 (distance: 0.23)
#   Representative: Library/Artist/Song.mp3
# NewAlbum/track2.mp3 -> Cluster 3 (distance: 0.31)
#   Representative: Library/OtherArtist/Track.mp3

# Add to existing playlists
music-cluster classify ~/Downloads/NewAlbum --recursive --export
```

### Inspect Results
```bash
# Show all clusterings
music-cluster list

# Show clusters in a clustering
music-cluster show auto_2024

# Describe specific cluster
music-cluster describe 7
```

---

## Future Enhancements

### Phase 5 (Optional)

**Interactive Features:**
- `music-cluster listen CLUSTER_ID` - Play representative tracks
- `music-cluster merge CLUSTER_A CLUSTER_B` - Combine clusters
- `music-cluster split CLUSTER_ID` - Re-cluster within a cluster

**Visualization:**
- 2D scatter plot using UMAP/t-SNE
- Cluster dendrogram for hierarchical view
- Feature importance analysis

**Advanced Classification:**
- Confidence scores (distance percentiles)
- Multi-label classification (track belongs to multiple clusters)
- Outlier detection (tracks that don't fit any cluster well)

**Alternative Algorithms:**
- DBSCAN (density-based, finds natural clusters)
- HDBSCAN (hierarchical DBSCAN)
- Gaussian Mixture Models (soft clustering)

**Smart Features:**
- Genre/mood prediction from cluster analysis
- Automatic cluster labeling using music metadata
- Playlist generation by "walking" through cluster space
- Similarity-based recommendations

---

## Success Metrics

**Functionality:**
- ✅ Processes 10,000+ tracks reliably
- ✅ Classification < 1 second per track
- ✅ Meaningful cluster separation (silhouette > 0.3)
- ✅ Generates usable M3U playlists

**Performance:**
- ✅ Multiprocessing utilizes all CPU cores
- ✅ Database < 2 KB per track
- ✅ Handles corrupted files gracefully
- ✅ Resume capability after interruption

**Usability:**
- ✅ Clear progress indicators
- ✅ Intuitive CLI commands
- ✅ Helpful error messages
- ✅ Comprehensive documentation

---

## Database Location

Default: `~/.music-cluster/library.db`

This centralizes the database outside the project directory, allowing the tool to be used across different music libraries while maintaining a single feature database.

Users can override with `--db-path` flag or config file setting.

---

## Notes

- Feature extraction is the most time-intensive operation; recommend running overnight for very large libraries
- Clustering quality depends on library diversity; homogeneous libraries may produce less distinct clusters
- Representative tracks are purely mathematical (centroid distance); may not always be subjectively "representative"
- Consider running multiple clusterings with different k values to explore different granularities
