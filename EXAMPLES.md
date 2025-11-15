# Example Workflows

## Quick Start Example

This example walks through the complete workflow with a small music library.

### 1. Initialize the Database

```bash
python -m music_cluster.cli init
```

Output:
```
Created configuration file: /Users/you/.music-cluster/config.yaml
Initialized database: /Users/you/.music-cluster/library.db

✓ Setup complete! You can now analyze your music library.
```

### 2. Analyze Your Music

Analyze a folder of music files:

```bash
python -m music_cluster.cli analyze ~/Music/TestLibrary --recursive
```

Output:
```
Scanning for audio files in /Users/you/Music/TestLibrary...
Found 50 audio files
Analyzing 50 files...
Extracting features: 100%|████████████████| 50/50 [00:45<00:00,  1.11it/s]

✓ Analysis complete!
  Analyzed: 50 tracks
  Total in database: 50 tracks
```

### 3. Create Clusters

Auto-detect optimal number of clusters:

```bash
python -m music_cluster.cli cluster --name "my_first_clustering"
```

Output:
```
Loading 50 feature vectors...
Finding optimal number of clusters (method: silhouette)...
Finding optimal k: 100%|████████████████| 15/15 [00:30<00:00,  2.00s/it]
Optimal k: 8
Running K-means with 8 clusters...
Finding representative tracks...
Saving clustering to database...

✓ Clustering complete!
  Clustering ID: 1
  Name: my_first_clustering
  Number of clusters: 8
```

### 4. Export Playlists

```bash
python -m music_cluster.cli export --output ~/Music/Playlists
```

Output:
```
Exporting clustering: my_first_clustering
Exporting 8 clusters to /Users/you/Music/Playlists...

✓ Export complete!
  Created 9 files in /Users/you/Music/Playlists
```

The output includes:
- `cluster_00.m3u` through `cluster_07.m3u`
- `clustering_summary.txt` with overview

### 5. Classify New Tracks

When you get new music:

```bash
python -m music_cluster.cli classify ~/Downloads/NewAlbum --recursive
```

Output:
```
Using clustering: my_first_clustering (ID: 1)
Classifying 12 tracks...
✓ track01.mp3
  → Cluster 3 (distance: 0.45)
  Representative: SimilarSong.mp3
✓ track02.mp3
  → Cluster 1 (distance: 0.38)
  Representative: AnotherSong.mp3
...
```

## Advanced Workflows

### Using Granularity Control

Create different granularity levels:

```bash
# Fewer, broader clusters (good for high-level organization)
python -m music_cluster.cli cluster --granularity fewer --name "broad_clusters"

# More detailed clusters (good for fine-grained sorting)
python -m music_cluster.cli cluster --granularity finer --name "detailed_clusters"
```

### Exact Cluster Count

If you know exactly how many clusters you want:

```bash
python -m music_cluster.cli cluster --clusters 15 --name "15_clusters"
```

### Parallel Processing for Large Libraries

Speed up analysis with multiple workers:

```bash
python -m music_cluster.cli analyze ~/Music/LargeLibrary \
  --recursive \
  --workers 8 \
  --batch-size 50
```

### View Clustering Quality

```bash
python -m music_cluster.cli cluster \
  --name "quality_test" \
  --show-metrics
```

Output includes:
```
Quality Metrics:
  Silhouette Score: 0.457
  Davies-Bouldin Score: 0.823
  Calinski-Harabasz Score: 245.3

Cluster Size Distribution:
  Min: 3 tracks
  Max: 25 tracks
  Mean: 12.5 tracks
```

### Export as JSON

For programmatic access:

```bash
python -m music_cluster.cli export \
  --format json \
  --output ./cluster_data
```

### View Database Info

```bash
python -m music_cluster.cli info
```

Output:
```
Music Cluster Database Info
==================================================
Database: /Users/you/.music-cluster/library.db
Total tracks: 1250
Analyzed tracks: 1250
Clusterings: 3

Recent Clusterings:
  - detailed_clusters (ID: 3)
    Clusters: 25, Created: 2024-01-15 14:32:10
  - broad_clusters (ID: 2)
    Clusters: 8, Created: 2024-01-15 14:15:20
```

### List All Clusterings

```bash
python -m music_cluster.cli list
```

### Show Specific Clustering

```bash
python -m music_cluster.cli show detailed_clusters
```

Output:
```
Clustering: detailed_clusters (ID: 3)
Total clusters: 25

Cluster 0:
  Size: 45 tracks
  Representative: Electronic_Track_01.mp3

Cluster 1:
  Size: 32 tracks
  Representative: Rock_Song_05.mp3
...
```

### Describe a Cluster

```bash
python -m music_cluster.cli describe 15
```

Output:
```
Cluster 5 (ID: 15)
Size: 28 tracks

Representative Track: Jazz_Standard_03.mp3

Tracks:
  1. Jazz_Standard_03.mp3
     Distance: 0.000
  2. Blue_Note_Session.mp3
     Distance: 0.234
  3. Smooth_Jazz_Evening.mp3
     Distance: 0.267
...
```

## Use Cases

### Use Case 1: DJ Set Preparation

Create playlists based on energy and style:

```bash
# Analyze your DJ library
python -m music_cluster.cli analyze ~/DJ/Library -r

# Create detailed clusters for similar tracks
python -m music_cluster.cli cluster --granularity finer --name "dj_sets"

# Export as playlists
python -m music_cluster.cli export --output ~/DJ/Playlists
```

### Use Case 2: Music Discovery

Find similar tracks in your library:

```bash
# Classify a track you like to find similar ones
python -m music_cluster.cli classify ~/Music/favorite_track.mp3

# Then check the cluster with `describe` to see all similar tracks
python -m music_cluster.cli describe <cluster_id>
```

### Use Case 3: Library Organization

Organize a large, unsorted music collection:

```bash
# Analyze everything
python -m music_cluster.cli analyze ~/Music/Unsorted -r --workers 8

# Create broad categories first
python -m music_cluster.cli cluster --granularity fewer --name "categories"

# Export and review
python -m music_cluster.cli export --output ~/Music/Categories

# Refine with more granular clustering
python -m music_cluster.cli cluster --granularity more --name "subcategories"
```

### Use Case 4: Playlist Generation

Create playlists that flow well together:

```bash
# Cluster your library
python -m music_cluster.cli cluster --name "flow_playlists"

# Export with representative tracks first
python -m music_cluster.cli export \
  --output ~/Playlists \
  --include-representative
```

Each playlist will start with the track most representative of that cluster's sound, followed by similar tracks.

## Tips and Best Practices

1. **Start with analysis**: Always run `analyze` on your entire library before clustering
2. **Experiment with granularity**: Try different levels to find what works for your library
3. **Use descriptive names**: Name your clusterings meaningfully (e.g., "electronic_subgenres")
4. **Check quality scores**: Higher silhouette scores (closer to 1.0) indicate better-defined clusters
5. **Re-analyze when needed**: Use `--update` if you modify or get better quality versions of tracks
6. **Classify incrementally**: Don't re-cluster when adding new music; just classify new tracks
7. **Backup your database**: The database at `~/.music-cluster/library.db` contains all your analysis

## Performance Tips

- **For 1000+ tracks**: Use `--workers 8` or more for parallel processing
- **For 10,000+ tracks**: Process in batches, use `--batch-size 50`
- **Low memory**: Use `--workers 1` and smaller batch sizes
- **Fast SSD**: Analysis speed is I/O bound; SSDs help significantly
