"""CLI interface for music-cluster."""

import click
import json
import os
import pickle
import sys
import platform
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import numpy as np

from .config import Config
from .database import Database
from .extractor import FeatureExtractor
from .clustering import ClusterEngine
from .classifier import TrackClassifier
from .exporter import PlaylistExporter
from .cluster_namer import ClusterNamer
from .utils import (
    find_audio_files, get_file_info, compute_file_checksum,
    parse_extensions, DEFAULT_AUDIO_EXTENSIONS
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Music Cluster - Analyze, cluster, and classify music tracks.
    
    A comprehensive tool for organizing music libraries using audio analysis,
    machine learning clustering, and automatic genre classification.
    
    Quick Start:
    
        1. music-cluster init
        
        2. music-cluster analyze ~/Music/techno --recursive
        
        3. music-cluster cluster --name techno_detailed --clusters 15
        
        4. music-cluster label-clusters techno_detailed
        
        5. music-cluster show techno_detailed
        
        6. music-cluster export --output ~/Music/playlists
    
    For detailed help on any command:
    
        music-cluster COMMAND --help
    """
    pass


@cli.command()
@click.option('--db-path', type=str, default=None, help='Database path (default: ~/.music-cluster/library.db)')
def init(db_path):
    """Initialize database and configuration."""
    # Create config
    config = Config.create_default_config()
    click.echo(f"Created configuration file: {config.config_path}")
    
    # Get database path
    if db_path:
        db_path = os.path.expanduser(db_path)
    else:
        db_path = config.get_db_path()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize database
    db = Database(db_path)
    click.echo(f"Initialized database: {db_path}")
    click.echo("\n✓ Setup complete! You can now analyze your music library.")


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('-r', '--recursive', is_flag=True, help='Scan subdirectories')
@click.option('-u', '--update', is_flag=True, help='Re-analyze existing tracks')
@click.option('--extensions', type=str, default='mp3,flac,wav,m4a,ogg,aiff,aif,opus,aac,wma,ape,alac,wv', 
              help='Comma-separated file extensions (default: all common formats)')
@click.option('--batch-size', type=int, default=100, help='Batch size for saving')
@click.option('--workers', type=int, default=-1, help='Number of parallel workers (-1 = all CPUs)')
@click.option('--skip-errors', is_flag=True, help='Continue on errors')
def analyze(path, recursive, update, extensions, batch_size, workers, skip_errors):
    """Analyze audio files and extract features.
    
    Extracts comprehensive audio features from music files including:
    - Timbral characteristics (MFCCs, spectral features)
    - Rhythmic features (BPM, onset strength)
    - Harmonic features (chroma)
    - Loudness and dynamics
    
    Examples:
        music-cluster analyze ~/Music --recursive
        music-cluster analyze ~/Music/techno -r --update
        music-cluster analyze ~/Music -r --workers 8
    """
    from .cli_helpers import (determine_worker_count, filter_files_to_analyze, 
                              process_files_parallel, process_files_sequential)
    
    # Load config
    config = Config()
    db = Database(config.get_db_path())
    
    # Parse extensions
    ext_set = parse_extensions(extensions)
    
    # Find audio files
    click.echo(f"Scanning for audio files in {path}...")
    audio_files = find_audio_files(path, recursive=recursive, extensions=ext_set)
    
    if not audio_files:
        click.echo("No audio files found.")
        return
    
    click.echo(f"Found {len(audio_files)} audio files")
    
    # Filter files that need analysis
    files_to_analyze = filter_files_to_analyze(audio_files, db, update)
    
    if not files_to_analyze:
        click.echo("All files already analyzed. Use --update to re-analyze.")
        return
    
    click.echo(f"Analyzing {len(files_to_analyze)} files...")
    
    # Determine number of workers
    workers = determine_worker_count(workers)
    
    # Initialize extractor config
    extractor_config = {
        'sample_rate': config.get('feature_extraction', 'sample_rate', default=44100),
        'frame_size': config.get('feature_extraction', 'frame_size', default=2048),
        'hop_size': config.get('feature_extraction', 'hop_size', default=1024),
        'n_mfcc': config.get('feature_extraction', 'mfcc_coefficients', default=20)
    }
    analysis_version = config.get('feature_extraction', 'analysis_version', default='1.0.0')
    
    # Process files
    if workers > 1:
        analyzed_count, error_count = process_files_parallel(
            files_to_analyze, extractor_config, workers, batch_size,
            db, analysis_version, skip_errors
        )
    else:
        analyzed_count, error_count = process_files_sequential(
            files_to_analyze, extractor_config, batch_size,
            db, analysis_version, skip_errors
        )
    
    click.echo(f"\n✓ Analysis complete!")
    click.echo(f"  Analyzed: {analyzed_count} tracks")
    if error_count > 0:
        click.echo(f"  Errors: {error_count} tracks")
    click.echo(f"  Total in database: {db.count_features()} tracks")




@cli.command()
@click.option('--name', type=str, default=None, help='Name for this clustering')
@click.option('--clusters', type=int, default=None, help='Exact number of clusters (not used for HDBSCAN)')
@click.option('--granularity', type=click.Choice(['fewer', 'less', 'normal', 'more', 'finer']),
              default='normal', help='Cluster granularity level')
@click.option('--algorithm', type=click.Choice(['kmeans', 'hierarchical', 'hdbscan']),
              default='kmeans', help='Clustering algorithm')
@click.option('--min-size', type=int, default=3, help='Minimum cluster size')
@click.option('--max-clusters', type=int, default=100, help='Maximum k to test')
@click.option('--method', type=click.Choice(['silhouette', 'elbow', 'calinski']),
              default='silhouette', help='Detection method')
@click.option('--epsilon', type=float, default=0.0, help='HDBSCAN distance threshold (0=auto, higher=fewer clusters)')
@click.option('--min-samples', type=int, default=None, help='HDBSCAN min samples (defaults to min-size)')
@click.option('--show-metrics', is_flag=True, help='Display clustering metrics')
def cluster(name, clusters, granularity, algorithm, min_size, max_clusters, method, epsilon, min_samples, show_metrics):
    """Perform clustering on analyzed tracks."""
    # Load config and database
    config = Config()
    db = Database(config.get_db_path())
    
    # Check if we have features
    feature_count = db.count_features()
    if feature_count == 0:
        click.echo("Error: No analyzed tracks found. Run 'music-cluster analyze' first.")
        return
    
    click.echo(f"Loading {feature_count} feature vectors...")
    
    # Load features
    feature_matrix, track_ids = db.get_all_features()
    
    if len(feature_matrix) < 2:
        click.echo("Error: Need at least 2 tracks to cluster.")
        return
    
    # Check for conflicting options
    if clusters and granularity != 'normal':
        click.echo("Warning: --clusters and --granularity are mutually exclusive. Using --clusters.")
        granularity = 'normal'
    
    # Initialize clustering engine
    engine = ClusterEngine(
        min_clusters=min_size,
        max_clusters=max_clusters,
        detection_method=method,
        algorithm=algorithm
    )
    
    # HDBSCAN doesn't need optimal k finding
    if algorithm == 'hdbscan':
        # Perform HDBSCAN clustering
        labels, centroids, cluster_metrics = engine.cluster(
            feature_matrix,
            show_progress=True,
            min_cluster_size=min_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon
        )
        n_clusters = len(centroids)
    else:
        # Determine number of clusters
        if clusters:
            n_clusters = clusters
            click.echo(f"Using specified cluster count: {n_clusters}")
        else:
            click.echo(f"Finding optimal number of clusters (method: {method})...")
            optimal_k, metrics = engine.find_optimal_k(feature_matrix, show_progress=True)
            
            # Apply granularity
            n_clusters = engine.apply_granularity(optimal_k, granularity)
            
            click.echo(f"Optimal k: {optimal_k}")
            if granularity != 'normal':
                click.echo(f"Adjusted for '{granularity}' granularity: {n_clusters}")
        
        # Perform clustering
        labels, centroids, cluster_metrics = engine.cluster(feature_matrix, n_clusters, show_progress=True)
    
    # Find representative tracks
    click.echo("Finding representative tracks...")
    representatives = engine.find_representative_tracks(feature_matrix, labels, centroids, track_ids)
    
    # Compute distances
    distances = engine.compute_distances_to_centroids(feature_matrix, labels, centroids)
    
    # Save to database
    click.echo("Saving clustering to database...")
    
    # Prepare parameters
    params = {
        'granularity': granularity,
        'method': method,
        'min_size': min_size
    }
    if algorithm == 'hdbscan':
        params['epsilon'] = epsilon
        params['min_samples'] = min_samples
    
    # Create clustering record
    clustering_id = db.add_clustering(
        name=name,
        algorithm=algorithm,
        num_clusters=n_clusters,
        parameters=json.dumps(params),
        silhouette_score=cluster_metrics.get('silhouette_score')
    )
    
    # For HDBSCAN, get unique labels (excluding noise)
    if algorithm == 'hdbscan':
        unique_labels = np.unique(labels[labels >= 0])
    else:
        unique_labels = range(n_clusters)
    
    # Save clusters and members
    for cluster_idx in unique_labels:
        cluster_mask = labels == cluster_idx
        cluster_size = int(np.sum(cluster_mask))
        
        if cluster_size == 0:
            continue
        
        # Get representative
        rep_track_id = representatives.get(cluster_idx)
        
        # Create cluster
        cluster_id = db.add_cluster(
            clustering_id=clustering_id,
            cluster_index=int(cluster_idx),
            size=cluster_size,
            representative_track_id=rep_track_id,
            centroid=centroids[cluster_idx if algorithm != 'hdbscan' else list(unique_labels).index(cluster_idx)]
        )
        
        # Add members
        for i, (track_id, label, distance) in enumerate(zip(track_ids, labels, distances)):
            if label == cluster_idx:
                db.add_cluster_member(cluster_id, track_id, distance)
    
    # Display results
    click.echo(f"\n✓ Clustering complete!")
    click.echo(f"  Clustering ID: {clustering_id}")
    if name:
        click.echo(f"  Name: {name}")
    click.echo(f"  Number of clusters: {n_clusters}")
    if algorithm == 'hdbscan' and cluster_metrics.get('n_noise', 0) > 0:
        click.echo(f"  Noise points: {cluster_metrics['n_noise']} ({cluster_metrics['noise_percentage']:.1f}%)")
    
    if show_metrics and cluster_metrics:
        click.echo("\nQuality Metrics:")
        if cluster_metrics.get('silhouette_score'):
            click.echo(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.3f}")
        if cluster_metrics.get('davies_bouldin_score'):
            click.echo(f"  Davies-Bouldin Score: {cluster_metrics['davies_bouldin_score']:.3f}")
        if cluster_metrics.get('calinski_harabasz_score'):
            click.echo(f"  Calinski-Harabasz Score: {cluster_metrics['calinski_harabasz_score']:.1f}")
        
        click.echo("\nCluster Size Distribution:")
        click.echo(f"  Min: {cluster_metrics['min_cluster_size']} tracks")
        click.echo(f"  Max: {cluster_metrics['max_cluster_size']} tracks")
        click.echo(f"  Mean: {cluster_metrics['mean_cluster_size']:.1f} tracks")


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('-r', '--recursive', is_flag=True, help='Process directory recursively')
@click.option('--clustering', type=str, default=None, help='Clustering name (default: latest)')
@click.option('--threshold', type=float, default=None, help='Max distance threshold')
@click.option('--export', is_flag=True, help='Add to playlists')
@click.option('--show-features', is_flag=True, help='Display features')
def classify(path, recursive, clustering, threshold, export, show_features):
    """Classify new tracks to existing clusters."""
    # Load config and database
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering_info = db.get_clustering(name=clustering) if clustering else db.get_clustering()
    if not clustering_info:
        click.echo("Error: No clustering found. Run 'music-cluster cluster' first.")
        return
    
    click.echo(f"Using clustering: {clustering_info.get('name', 'Unnamed')} (ID: {clustering_info['id']})")
    
    # Load clusters and centroids
    clusters = db.get_clusters_by_clustering(clustering_info['id'])
    if not clusters:
        click.echo("Error: No clusters found in this clustering.")
        return
    
    # Build centroid matrix
    centroids = []
    cluster_ids = []
    for cluster in clusters:
        if cluster['centroid']:
            centroid = pickle.loads(cluster['centroid'])
            centroids.append(centroid)
            cluster_ids.append(cluster['id'])
    
    centroids = np.array(centroids)
    
    # Initialize classifier
    classifier = TrackClassifier(centroids, cluster_ids)
    
    # Find audio files to classify
    audio_files = find_audio_files(path, recursive=recursive)
    if not audio_files:
        click.echo("No audio files found.")
        return
    
    # Initialize extractor
    extractor_config = {
        'sample_rate': config.get('feature_extraction', 'sample_rate', default=44100),
        'frame_size': config.get('feature_extraction', 'frame_size', default=2048),
        'hop_size': config.get('feature_extraction', 'hop_size', default=1024),
        'n_mfcc': config.get('feature_extraction', 'mfcc_coefficients', default=20)
    }
    extractor = FeatureExtractor(**extractor_config)
    
    # Classify each file
    click.echo(f"Classifying {len(audio_files)} tracks...")
    
    for filepath in audio_files:
        # Extract features
        features = extractor.extract(filepath)
        if features is None:
            click.echo(f"✗ {os.path.basename(filepath)}: Failed to extract features")
            continue
        
        # Classify
        cluster_id, distance = classifier.classify(features, threshold)
        
        if cluster_id == -1:
            click.echo(f"✗ {os.path.basename(filepath)}: No match (distance: {distance:.2f})")
        else:
            # Get cluster info
            cluster_info = db.get_cluster(cluster_id)
            rep_track = db.get_track_by_id(cluster_info['representative_track_id'])
            
            click.echo(f"✓ {os.path.basename(filepath)}")
            click.echo(f"  → Cluster {cluster_info['cluster_index']} (distance: {distance:.2f})")
            if rep_track:
                click.echo(f"  Representative: {rep_track['filename']}")


@cli.command()
@click.option('--output', type=str, default='./playlists', help='Output directory')
@click.option('--format', type=click.Choice(['m3u', 'm3u8', 'json']), 
              default='m3u', help='Playlist format')
@click.option('--clustering', type=str, default=None, help='Clustering name (default: latest)')
@click.option('--relative-paths', is_flag=True, help='Use relative paths')
@click.option('--include-representative', is_flag=True, default=True, 
              help='Include representative track first')
def export(output, format, clustering, relative_paths, include_representative):
    """Export clusters as playlists."""
    # Load config and database
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering_info = db.get_clustering(name=clustering) if clustering else db.get_clustering()
    if not clustering_info:
        click.echo("Error: No clustering found. Run 'music-cluster cluster' first.")
        return
    
    click.echo(f"Exporting clustering: {clustering_info.get('name', 'Unnamed')}")
    
    # Get clusters
    clusters = db.get_clusters_by_clustering(clustering_info['id'])
    if not clusters:
        click.echo("Error: No clusters found.")
        return
    
    # Prepare cluster data
    clusters_data = []
    for cluster in clusters:
        members = db.get_cluster_members(cluster['id'])
        rep_track = db.get_track_by_id(cluster['representative_track_id']) if cluster['representative_track_id'] else None
        
        clusters_data.append({
            'cluster': cluster,
            'tracks': members,
            'representative': rep_track
        })
    
    # Export
    exporter = PlaylistExporter(
        output_dir=output,
        playlist_format=format,
        relative_paths=relative_paths,
        include_representative=include_representative
    )
    
    click.echo(f"Exporting {len(clusters)} clusters to {output}...")
    created_files = exporter.export_all_clusters(clustering_info, clusters_data)
    
    click.echo(f"\n✓ Export complete!")
    click.echo(f"  Created {len(created_files)} files in {output}")


@cli.command()
def info():
    """Show database statistics."""
    config = Config()
    db = Database(config.get_db_path())
    
    track_count = db.count_tracks()
    feature_count = db.count_features()
    clusterings = db.get_all_clusterings()
    
    click.echo("Music Cluster Database Info")
    click.echo("=" * 50)
    click.echo(f"Database: {config.get_db_path()}")
    click.echo(f"Total tracks: {track_count}")
    click.echo(f"Analyzed tracks: {feature_count}")
    click.echo(f"Clusterings: {len(clusterings)}")
    
    if clusterings:
        click.echo("\nRecent Clusterings:")
        for clustering in clusterings[:5]:
            click.echo(f"  - {clustering.get('name', 'Unnamed')} (ID: {clustering['id']})")
            click.echo(f"    Clusters: {clustering['num_clusters']}, Created: {clustering.get('created_at', 'Unknown')}")


@cli.command(name='list')
def list_cmd():
    """List all clusterings."""
    config = Config()
    db = Database(config.get_db_path())
    
    clusterings = db.get_all_clusterings()
    
    if not clusterings:
        click.echo("No clusterings found.")
        return
    
    click.echo(f"Found {len(clusterings)} clustering(s):\n")
    
    for clustering in clusterings:
        click.echo(f"ID: {clustering['id']}")
        click.echo(f"Name: {clustering.get('name', 'Unnamed')}")
        click.echo(f"Algorithm: {clustering.get('algorithm', 'kmeans')}")
        click.echo(f"Clusters: {clustering['num_clusters']}")
        if clustering.get('silhouette_score'):
            click.echo(f"Quality Score: {clustering['silhouette_score']:.3f}")
        click.echo(f"Created: {clustering.get('created_at', 'Unknown')}")
        click.echo()


@cli.command()
@click.argument('clustering_name', type=str)
def show(clustering_name):
    """Show clusters in a clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    clustering = db.get_clustering(name=clustering_name)
    if not clustering:
        click.echo(f"Error: Clustering '{clustering_name}' not found.")
        return
    
    clusters = db.get_clusters_by_clustering(clustering['id'])
    
    click.echo(f"Clustering: {clustering.get('name', 'Unnamed')} (ID: {clustering['id']})")
    click.echo(f"Total clusters: {len(clusters)}\n")
    
    for cluster in clusters:
        rep_track = db.get_track_by_id(cluster['representative_track_id']) if cluster['representative_track_id'] else None
        
        cluster_label = f"Cluster {cluster['cluster_index']}"
        if cluster.get('name'):
            cluster_label += f": {cluster['name']}"
        
        click.echo(cluster_label)
        click.echo(f"  Size: {cluster['size']} tracks")
        if rep_track:
            click.echo(f"  Representative: {rep_track['filename']}")
        click.echo()


@cli.command()
@click.argument('cluster_id', type=int)
def describe(cluster_id):
    """Show tracks in a cluster."""
    config = Config()
    db = Database(config.get_db_path())
    
    cluster = db.get_cluster(cluster_id)
    if not cluster:
        click.echo(f"Error: Cluster {cluster_id} not found.")
        return
    
    members = db.get_cluster_members(cluster_id)
    
    click.echo(f"Cluster {cluster['cluster_index']} (ID: {cluster_id})")
    click.echo(f"Size: {len(members)} tracks\n")
    
    if cluster['representative_track_id']:
        rep_track = db.get_track_by_id(cluster['representative_track_id'])
        if rep_track:
            click.echo(f"Representative Track: {rep_track['filename']}\n")
    
    click.echo("Tracks:")
    for i, track in enumerate(members, 1):
        click.echo(f"{i:3d}. {track['filename']}")
        if 'distance_to_centroid' in track:
            click.echo(f"     Distance: {track['distance_to_centroid']:.3f}")


@cli.command(name='rename-cluster')
@click.argument('clustering_name', type=str)
@click.argument('cluster_index', type=int)
@click.argument('new_name', type=str)
def rename_cluster_cmd(clustering_name, cluster_index, new_name):
    """Rename a specific cluster."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering = db.get_clustering(name=clustering_name)
    if not clustering:
        click.echo(f"Error: Clustering '{clustering_name}' not found.")
        return
    
    # Get cluster
    cluster = db.get_cluster_by_index(clustering['id'], cluster_index)
    if not cluster:
        click.echo(f"Error: Cluster {cluster_index} not found in clustering '{clustering_name}'.")
        return
    
    # Update name
    db.update_cluster_name(cluster['id'], new_name)
    
    click.echo(f"✓ Renamed Cluster {cluster_index} to '{new_name}'")


@cli.command(name='label-clusters')
@click.argument('clustering_name', type=str)
@click.option('--dry-run', is_flag=True, help='Show generated names without saving')
@click.option('--no-genre', is_flag=True, help='Exclude genre classification from names')
@click.option('--no-bpm', is_flag=True, help='Exclude BPM information from names')
@click.option('--no-descriptors', is_flag=True, help='Exclude characteristics (Bass-Heavy, Dark, etc.)')
@click.option('--bpm-average', is_flag=True, help='Use average BPM instead of range')
def label_clusters_cmd(clustering_name, dry_run, no_genre, no_bpm, no_descriptors, bpm_average):
    """Auto-generate descriptive names for all clusters.
    
    Analyzes audio characteristics to generate meaningful names including:
    - Genre classification (Techno, House, Drum & Bass, etc.)
    - BPM information (range or average)
    - Distinctive characteristics (Bass-Heavy, Bright, Dark, etc.)
    
    Naming scheme can be customized with flags:
        --no-genre: Skip genre classification
        --no-bpm: Skip BPM information  
        --no-descriptors: Skip characteristics
        --bpm-average: Use average BPM instead of range
    
    Examples:
        music-cluster label-clusters my_clustering --dry-run
        music-cluster label-clusters techno_fine_kmeans
        music-cluster label-clusters my_clustering --no-bpm
        music-cluster label-clusters my_clustering --bpm-average
    """
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering = db.get_clustering(name=clustering_name)
    if not clustering:
        click.echo(f"Error: Clustering '{clustering_name}' not found.")
        return
    
    # Get clusters
    clusters = db.get_clusters_by_clustering(clustering['id'])
    if not clusters:
        click.echo("No clusters found.")
        return
    
    # Unpickle centroids with error handling
    for cluster in clusters:
        if cluster['centroid']:
            try:
                cluster['centroid'] = pickle.loads(cluster['centroid'])
            except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as e:
                click.echo(f"Warning: Failed to unpickle centroid for cluster {cluster['cluster_index']}: {e}", err=True)
                click.echo(f"  Skipping this cluster for name generation.", err=True)
                cluster['centroid'] = None
    
    # Load features and labels
    click.echo("Loading features...")
    feature_matrix, track_ids = db.get_all_features()
    
    # Reconstruct labels from cluster membership
    labels = np.full(len(track_ids), -1, dtype=int)
    for cluster in clusters:
        members = db.get_cluster_members(cluster['id'])
        for member in members:
            try:
                # Note: member comes from cluster_members table, use track_id not id
                track_idx = track_ids.index(member['id']) if 'id' in member else track_ids.index(member.get('track_id', -1))
                labels[track_idx] = cluster['cluster_index']
            except (ValueError, KeyError):
                # Track not in feature list (possibly deleted or not analyzed), skip
                continue
    
    # Generate names
    click.echo("Generating cluster names...")
    namer = ClusterNamer(
        include_genre=not no_genre,
        include_bpm=not no_bpm,
        use_bpm_range=not bpm_average,
        include_descriptors=not no_descriptors
    )
    names = namer.generate_names_for_clustering(clusters, feature_matrix, labels)
    
    # Display and optionally save
    click.echo(f"\nGenerated names for {len(names)} clusters:\n")
    for cluster_idx, name in sorted(names.items()):
        cluster = next(c for c in clusters if c['cluster_index'] == cluster_idx)
        click.echo(f"Cluster {cluster_idx}: {name} ({cluster['size']} tracks)")
        
        if not dry_run:
            db.update_cluster_name(cluster['id'], name)
    
    if dry_run:
        click.echo("\n(Dry run - no names were saved. Run without --dry-run to save.)")
    else:
        click.echo(f"\n✓ Updated {len(names)} cluster names")


@cli.command()
@click.argument('query', type=str)
@click.option('--clustering', type=str, default=None, help='Clustering name (default: all tracks)')
@click.option('--limit', type=int, default=20, help='Maximum results to show')
def search(query, clustering, limit):
    """Search for tracks by name or artist.
    
    Searches track filenames and paths (case-insensitive). If a clustering
    is specified, shows which cluster each track belongs to.
    
    Examples:
        music-cluster search "Aphex Twin"
        music-cluster search "K-LONE" --clustering techno_fine_kmeans
        music-cluster search "remix" --limit 50
    """
    config = Config()
    db = Database(config.get_db_path())
    
    try:
        # Get all tracks
        all_tracks = db.get_all_tracks()
    except Exception as e:
        click.echo(f"Error accessing database: {e}", err=True)
        return
    
    if not all_tracks:
        click.echo("No tracks in database.")
        return
    
    # Filter by query (case-insensitive)
    query_lower = query.lower()
    matching_tracks = [
        t for t in all_tracks
        if query_lower in t['filename'].lower() or query_lower in t['filepath'].lower()
    ]
    
    if not matching_tracks:
        click.echo(f"No tracks found matching '{query}'")
        return
    
    # If clustering specified, show which clusters they belong to
    if clustering:
        clustering_info = db.get_clustering(name=clustering)
        if not clustering_info:
            click.echo(f"Error: Clustering '{clustering}' not found.")
            return
        
        click.echo(f"Found {len(matching_tracks)} track(s) matching '{query}' in clustering '{clustering}':\n")
        
        for track in matching_tracks[:limit]:
            cluster_id = db.get_track_cluster(track['id'], clustering_info['id'])
            if cluster_id:
                cluster = db.get_cluster(cluster_id)
                cluster_label = f"Cluster {cluster['cluster_index']}"
                if cluster.get('name'):
                    cluster_label += f": {cluster['name']}"
                click.echo(f"✓ {track['filename']}")
                click.echo(f"  {cluster_label}")
                click.echo(f"  {track['filepath']}")
                click.echo()
            else:
                click.echo(f"✗ {track['filename']} (not in clustering)")
                click.echo(f"  {track['filepath']}")
                click.echo()
    else:
        click.echo(f"Found {len(matching_tracks)} track(s) matching '{query}':\n")
        for track in matching_tracks[:limit]:
            click.echo(f"• {track['filename']}")
            click.echo(f"  {track['filepath']}")
            click.echo()
    
    if len(matching_tracks) > limit:
        click.echo(f"... and {len(matching_tracks) - limit} more. Use --limit to see more.")


@cli.command()
@click.argument('clustering_name', type=str)
def stats(clustering_name):
    """Show detailed statistics for a clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering = db.get_clustering(name=clustering_name)
    if not clustering:
        click.echo(f"Error: Clustering '{clustering_name}' not found.")
        return
    
    clusters = db.get_clusters_by_clustering(clustering['id'])
    
    if not clusters:
        click.echo("No clusters found.")
        return
    
    # Compute statistics
    sizes = [c['size'] for c in clusters if c['size'] > 0]
    
    if not sizes:
        click.echo("Error: No valid clusters with tracks found.")
        return
    
    total_tracks = sum(sizes)
    
    click.echo(f"\nStatistics for '{clustering_name}'")
    click.echo("=" * 60)
    click.echo(f"Algorithm:        {clustering.get('algorithm', 'kmeans')}")
    click.echo(f"Total Clusters:   {len(clusters)}")
    click.echo(f"Total Tracks:     {total_tracks}")
    
    if clustering.get('silhouette_score'):
        click.echo(f"Quality Score:    {clustering['silhouette_score']:.3f}")
    
    click.echo(f"\nCluster Size Distribution:")
    click.echo(f"  Smallest:       {min(sizes)} tracks")
    click.echo(f"  Largest:        {max(sizes)} tracks")
    click.echo(f"  Mean:           {np.mean(sizes):.1f} tracks")
    click.echo(f"  Median:         {np.median(sizes):.0f} tracks")
    click.echo(f"  Std Dev:        {np.std(sizes):.1f}")
    
    # Size buckets
    small = len([s for s in sizes if s < 50])
    medium = len([s for s in sizes if 50 <= s < 150])
    large = len([s for s in sizes if s >= 150])
    
    click.echo(f"\nSize Buckets:")
    click.echo(f"  Small (<50):    {small} clusters")
    click.echo(f"  Medium (50-149): {medium} clusters")
    click.echo(f"  Large (≥150):   {large} clusters")
    
    # Named clusters
    named = len([c for c in clusters if c.get('name')])
    if named > 0:
        click.echo(f"\nNamed Clusters:   {named} / {len(clusters)}")
    
    # Show top 5 largest clusters
    click.echo(f"\nTop 5 Largest Clusters:")
    sorted_clusters = sorted(clusters, key=lambda c: c['size'], reverse=True)[:5]
    for i, cluster in enumerate(sorted_clusters, 1):
        name = cluster.get('name', '')
        label = f"Cluster {cluster['cluster_index']}"
        if name:
            label += f": {name}"
        click.echo(f"  {i}. {label} ({cluster['size']} tracks)")


@cli.command()
@click.argument('clustering1', type=str)
@click.argument('clustering2', type=str)
def compare(clustering1, clustering2):
    """Compare two clusterings side-by-side."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get both clusterings
    c1 = db.get_clustering(name=clustering1)
    c2 = db.get_clustering(name=clustering2)
    
    if not c1:
        click.echo(f"Error: Clustering '{clustering1}' not found.")
        return
    if not c2:
        click.echo(f"Error: Clustering '{clustering2}' not found.")
        return
    
    clusters1 = db.get_clusters_by_clustering(c1['id'])
    clusters2 = db.get_clusters_by_clustering(c2['id'])
    
    # Display comparison
    click.echo("\nClustering Comparison")
    click.echo("=" * 60)
    click.echo(f"{clustering1:<30} vs {clustering2:<30}")
    click.echo("=" * 60)
    
    click.echo(f"{'Algorithm:':<20} {c1.get('algorithm', 'kmeans'):<30} {c2.get('algorithm', 'kmeans'):<30}")
    click.echo(f"{'Clusters:':<20} {c1['num_clusters']:<30} {c2['num_clusters']:<30}")
    
    if c1.get('silhouette_score') and c2.get('silhouette_score'):
        s1 = c1['silhouette_score']
        s2 = c2['silhouette_score']
        better = "←" if s1 > s2 else "→"
        click.echo(f"{'Silhouette Score:':<20} {s1:<30.3f} {s2:<30.3f} {better}")
    
    # Cluster size stats
    sizes1 = [c['size'] for c in clusters1 if c['size'] > 0]
    sizes2 = [c['size'] for c in clusters2 if c['size'] > 0]
    
    if not sizes1 or not sizes2:
        click.echo(f"\nError: Cannot compare clusterings with no valid clusters.")
        if not sizes1:
            click.echo(f"  '{clustering1}' has no clusters with tracks.")
        if not sizes2:
            click.echo(f"  '{clustering2}' has no clusters with tracks.")
        return
    
    click.echo(f"\n{'Cluster Size Distribution:'}")
    click.echo(f"{'  Min:':<20} {min(sizes1):<30} {min(sizes2):<30}")
    click.echo(f"{'  Max:':<20} {max(sizes1):<30} {max(sizes2):<30}")
    click.echo(f"{'  Mean:':<20} {np.mean(sizes1):<30.1f} {np.mean(sizes2):<30.1f}")
    click.echo(f"{'  Std Dev:':<20} {np.std(sizes1):<30.1f} {np.std(sizes2):<30.1f}")


if __name__ == '__main__':
    cli()
