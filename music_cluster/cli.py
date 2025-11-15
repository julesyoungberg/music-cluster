"""CLI interface for music-cluster."""

import click
import json
import os
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

from .config import Config
from .database import Database
from .extractor import FeatureExtractor
from .clustering import ClusterEngine
from .classifier import TrackClassifier
from .exporter import PlaylistExporter
from .utils import (
    find_audio_files, get_file_info, compute_file_checksum,
    parse_extensions, DEFAULT_AUDIO_EXTENSIONS
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Music Cluster - Analyze, cluster, and classify music tracks."""
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
    """Analyze audio files and extract features."""
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
    files_to_analyze = []
    for filepath in audio_files:
        existing = db.get_track_by_filepath(filepath)
        if existing and not update:
            # Skip if already analyzed and not updating
            continue
        files_to_analyze.append(filepath)
    
    if not files_to_analyze:
        click.echo("All files already analyzed. Use --update to re-analyze.")
        return
    
    click.echo(f"Analyzing {len(files_to_analyze)} files...")
    
    # Determine number of workers
    if workers == -1:
        workers = mp.cpu_count()
    elif workers <= 0:
        workers = 1
    
    # Initialize extractor config
    extractor_config = {
        'sample_rate': config.get('feature_extraction', 'sample_rate', default=44100),
        'frame_size': config.get('feature_extraction', 'frame_size', default=2048),
        'hop_size': config.get('feature_extraction', 'hop_size', default=1024),
        'n_mfcc': config.get('feature_extraction', 'mfcc_coefficients', default=20)
    }
    analysis_version = config.get('feature_extraction', 'analysis_version', default='1.0.0')
    
    # Process files
    analyzed_count = 0
    error_count = 0
    skipped_count = 0
    
    with tqdm(total=len(files_to_analyze), desc="Extracting features") as pbar:
        if workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_extract_features_worker, filepath, extractor_config): filepath
                    for filepath in files_to_analyze
                }
                
                batch_results = []
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                            analyzed_count += 1
                            
                            # Save in batches
                            if len(batch_results) >= batch_size:
                                _save_batch_to_db(db, batch_results, analysis_version)
                                batch_results = []
                        else:
                            error_count += 1
                            if not skip_errors:
                                click.echo(f"\nError analyzing {filepath}")
                    except Exception as e:
                        error_count += 1
                        if not skip_errors:
                            click.echo(f"\nError analyzing {filepath}: {e}")
                    
                    pbar.update(1)
                
                # Save remaining batch
                if batch_results:
                    _save_batch_to_db(db, batch_results, analysis_version)
        else:
            # Sequential processing
            extractor = FeatureExtractor(**extractor_config)
            batch_results = []
            
            for filepath in files_to_analyze:
                try:
                    # Extract features
                    features = extractor.extract(filepath)
                    if features is not None:
                        duration = extractor.get_audio_duration(filepath)
                        file_info = get_file_info(filepath)
                        checksum = compute_file_checksum(filepath)
                        
                        batch_results.append({
                            'filepath': filepath,
                            'file_info': file_info,
                            'duration': duration,
                            'checksum': checksum,
                            'features': features
                        })
                        analyzed_count += 1
                        
                        # Save in batches
                        if len(batch_results) >= batch_size:
                            _save_batch_to_db(db, batch_results, analysis_version)
                            batch_results = []
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    if not skip_errors:
                        click.echo(f"\nError analyzing {filepath}: {e}")
                
                pbar.update(1)
            
            # Save remaining batch
            if batch_results:
                _save_batch_to_db(db, batch_results, analysis_version)
    
    click.echo(f"\n✓ Analysis complete!")
    click.echo(f"  Analyzed: {analyzed_count} tracks")
    if error_count > 0:
        click.echo(f"  Errors: {error_count} tracks")
    click.echo(f"  Total in database: {db.count_features()} tracks")


from typing import Optional, Dict

def _extract_features_worker(filepath: str, config: dict) -> Optional[Dict]:
    """Worker function for parallel feature extraction."""
    try:
        extractor = FeatureExtractor(**config)
        features = extractor.extract(filepath)
        
        if features is not None:
            duration = extractor.get_audio_duration(filepath)
            file_info = get_file_info(filepath)
            checksum = compute_file_checksum(filepath)
            
            return {
                'filepath': filepath,
                'file_info': file_info,
                'duration': duration,
                'checksum': checksum,
                'features': features
            }
    except Exception as e:
        # Return a marker with error info for optional logging upstream
        return {'filepath': filepath, 'error': str(e)}
    return None


def _save_batch_to_db(db: Database, batch_results: List[dict], analysis_version: str):
    """Save a batch of results to database."""
    for result in batch_results:
        # Add track
        track_id = db.add_track(
            filepath=result['filepath'],
            filename=result['file_info']['filename'],
            duration=result['duration'],
            file_size=result['file_info']['file_size'],
            checksum=result['checksum'],
            analysis_version=analysis_version
        )
        
        # Add features
        db.add_features(track_id, result['features'])


@cli.command()
@click.option('--name', type=str, default=None, help='Name for this clustering')
@click.option('--clusters', type=int, default=None, help='Exact number of clusters')
@click.option('--granularity', type=click.Choice(['fewer', 'less', 'normal', 'more', 'finer']),
              default='normal', help='Cluster granularity level')
@click.option('--min-size', type=int, default=3, help='Minimum cluster size')
@click.option('--max-clusters', type=int, default=100, help='Maximum k to test')
@click.option('--method', type=click.Choice(['silhouette', 'elbow', 'calinski']),
              default='silhouette', help='Detection method')
@click.option('--show-metrics', is_flag=True, help='Display clustering metrics')
def cluster(name, clusters, granularity, min_size, max_clusters, method, show_metrics):
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
        detection_method=method
    )
    
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
    
    # Create clustering record
    clustering_id = db.add_clustering(
        name=name,
        algorithm='kmeans',
        num_clusters=n_clusters,
        parameters=json.dumps({
            'granularity': granularity,
            'method': method,
            'min_size': min_size
        }),
        silhouette_score=cluster_metrics.get('silhouette_score')
    )
    
    # Save clusters and members
    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_size = int(np.sum(cluster_mask))
        
        if cluster_size == 0:
            continue
        
        # Get representative
        rep_track_id = representatives.get(cluster_idx)
        
        # Create cluster
        cluster_id = db.add_cluster(
            clustering_id=clustering_id,
            cluster_index=cluster_idx,
            size=cluster_size,
            representative_track_id=rep_track_id,
            centroid=centroids[cluster_idx]
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
        
        click.echo(f"Cluster {cluster['cluster_index']}:")
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


if __name__ == '__main__':
    cli()
