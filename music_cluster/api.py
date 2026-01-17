"""
FastAPI REST API for Music Cluster UI.

This API wraps all CLI functionality for use by the Tauri + Svelte frontend.

To run:
    pip install fastapi uvicorn
    uvicorn music_cluster.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import json
import pickle

from .config import Config
from .database import Database
from .clustering import ClusterEngine
from .extractor import FeatureExtractor
from .classifier import TrackClassifier
from .exporter import PlaylistExporter
from .cluster_namer import ClusterNamer
from .utils import find_audio_files, parse_extensions

app = FastAPI(title="Music Cluster API", version="1.0.0")

# Enable CORS for frontend (Tauri, Electron, or web browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task status (in production, use Redis or database)
task_status: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    path: str
    recursive: bool = True
    update: bool = False
    extensions: str = "mp3,flac,wav,m4a,ogg,aiff,aif,opus,aac,wma,ape,alac,wv"
    workers: int = -1
    skip_errors: bool = True


class ClusterRequest(BaseModel):
    name: Optional[str] = None
    clusters: Optional[int] = None
    granularity: str = "normal"
    algorithm: str = "kmeans"
    min_size: int = 3
    max_clusters: int = 100
    method: str = "silhouette"
    epsilon: float = 0.0
    min_samples: Optional[int] = None
    show_metrics: bool = False


class ClassifyRequest(BaseModel):
    path: str
    recursive: bool = True
    clustering: Optional[str] = None
    threshold: Optional[float] = None


class LabelClustersRequest(BaseModel):
    clustering_name: str
    no_genre: bool = False
    no_bpm: bool = False
    no_descriptors: bool = False
    bpm_average: bool = False
    dry_run: bool = False


class ExportRequest(BaseModel):
    output: str = "./playlists"
    format: str = "m3u"
    clustering: Optional[str] = None
    relative_paths: bool = False
    include_representative: bool = True


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Music Cluster API",
        "version": "1.0.0",
        "endpoints": {
            "info": "/api/info",
            "tracks": "/api/tracks",
            "clusterings": "/api/clusterings",
            "analyze": "/api/analyze",
            "cluster": "/api/cluster",
        }
    }


@app.get("/api/info")
async def get_info():
    """Get database statistics."""
    config = Config()
    db = Database(config.get_db_path())
    
    return {
        "database_path": config.get_db_path(),
        "total_tracks": db.count_tracks(),
        "analyzed_tracks": db.count_features(),
        "clusterings": len(db.get_all_clusterings())
    }


@app.get("/api/tracks")
async def get_tracks(limit: int = 100, offset: int = 0):
    """Get list of tracks."""
    config = Config()
    db = Database(config.get_db_path())
    
    tracks = db.get_all_tracks()
    total = len(tracks)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "tracks": tracks[offset:offset + limit]
    }


@app.get("/api/tracks/{track_id}")
async def get_track(track_id: int):
    """Get a specific track."""
    config = Config()
    db = Database(config.get_db_path())
    
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    return track


@app.get("/api/clusterings")
async def get_clusterings():
    """Get all clusterings."""
    config = Config()
    db = Database(config.get_db_path())
    
    clusterings = db.get_all_clusterings()
    return {"clusterings": clusterings}


@app.get("/api/clusterings/{clustering_id}")
async def get_clustering(clustering_id: int):
    """Get a specific clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    clustering = db.get_clustering(clustering_id=clustering_id)
    if not clustering:
        raise HTTPException(status_code=404, detail="Clustering not found")
    
    clusters = db.get_clusters_by_clustering(clustering_id)
    
    # Remove centroid BLOBs from cluster data (not JSON-serializable and not needed for listing)
    clusters_serializable = []
    for cluster in clusters:
        cluster_dict = dict(cluster)
        # Exclude centroid binary data from response
        if 'centroid' in cluster_dict:
            del cluster_dict['centroid']
        clusters_serializable.append(cluster_dict)
    
    return {
        **clustering,
        "clusters": clusters_serializable
    }


@app.get("/api/clusterings/name/{name}")
async def get_clustering_by_name(name: str):
    """Get clustering by name."""
    config = Config()
    db = Database(config.get_db_path())
    
    clustering = db.get_clustering(name=name)
    if not clustering:
        raise HTTPException(status_code=404, detail="Clustering not found")
    
    clusters = db.get_clusters_by_clustering(clustering['id'])
    
    # Remove centroid BLOBs from cluster data (not JSON-serializable and not needed for listing)
    clusters_serializable = []
    for cluster in clusters:
        cluster_dict = dict(cluster)
        # Exclude centroid binary data from response
        if 'centroid' in cluster_dict:
            del cluster_dict['centroid']
        clusters_serializable.append(cluster_dict)
    
    return {
        **clustering,
        "clusters": clusters_serializable
    }


@app.get("/api/clusters/{cluster_id}")
async def get_cluster(cluster_id: int):
    """Get a specific cluster with its tracks."""
    config = Config()
    db = Database(config.get_db_path())
    
    cluster = db.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    # Convert cluster dict, handling centroid
    cluster_dict = dict(cluster)
    # Centroid is already unpickled by get_cluster, but numpy arrays aren't JSON-serializable
    # Convert to list if present, or exclude it
    if 'centroid' in cluster_dict and cluster_dict['centroid'] is not None:
        if isinstance(cluster_dict['centroid'], np.ndarray):
            cluster_dict['centroid'] = cluster_dict['centroid'].tolist()
        elif isinstance(cluster_dict['centroid'], bytes):
            # If still bytes, exclude it (shouldn't happen after get_cluster unpickles)
            del cluster_dict['centroid']
    
    members = db.get_cluster_members(cluster_id)
    
    return {
        **cluster_dict,
        "tracks": members
    }


@app.get("/api/stats/{clustering_id}")
async def get_stats(clustering_id: int):
    """Get statistics for a clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    clustering = db.get_clustering(clustering_id=clustering_id)
    if not clustering:
        raise HTTPException(status_code=404, detail="Clustering not found")
    
    clusters = db.get_clusters_by_clustering(clustering_id)
    sizes = [c['size'] for c in clusters if c['size'] > 0]
    
    if not sizes:
        raise HTTPException(status_code=400, detail="No valid clusters found")
    
    total_tracks = sum(sizes)
    
    # Size buckets
    small = len([s for s in sizes if s < 50])
    medium = len([s for s in sizes if 50 <= s < 150])
    large = len([s for s in sizes if s >= 150])
    
    # Named clusters
    named = len([c for c in clusters if c.get('name')])
    
    return {
        "clustering_id": clustering_id,
        "algorithm": clustering.get('algorithm', 'kmeans'),
        "num_clusters": len(clusters),
        "total_tracks": total_tracks,
        "silhouette_score": clustering.get('silhouette_score'),
        "size_distribution": {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes))
        },
        "size_buckets": {
            "small": small,
            "medium": medium,
            "large": large
        },
        "named_clusters": named
    }


@app.post("/api/analyze")
async def analyze_tracks(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start analyzing audio files (runs in background)."""
    task_id = f"analyze_{Path(request.path).name}_{id(request)}"
    
    # Check if path exists
    if not Path(request.path).exists():
        raise HTTPException(status_code=400, detail="Path does not exist")
    
    # Initialize task status
    task_status[task_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "completed": 0,
        "errors": 0,
        "current_file": None
    }
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis,
        request,
        task_id
    )
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Analysis started in background"
    }


async def run_analysis(request: AnalyzeRequest, task_id: str):
    """Background task for analysis."""
    try:
        from .cli_helpers import (
            determine_worker_count, filter_files_to_analyze,
            process_files_parallel, process_files_sequential
        )
        
        config = Config()
        db = Database(config.get_db_path())
        
        # Find audio files
        ext_set = parse_extensions(request.extensions)
        audio_files = find_audio_files(request.path, recursive=request.recursive, extensions=ext_set)
        
        if not audio_files:
            task_status[task_id] = {"status": "error", "message": "No audio files found"}
            return
        
        # Filter files
        files_to_analyze = filter_files_to_analyze(audio_files, db, request.update)
        
        if not files_to_analyze:
            task_status[task_id] = {"status": "complete", "message": "All files already analyzed", "total": 0, "completed": 0}
            return
        
        task_status[task_id]["total"] = len(files_to_analyze)
        
        # Extract config
        extractor_config = {
            'sample_rate': config.get('feature_extraction', 'sample_rate', default=44100),
            'frame_size': config.get('feature_extraction', 'frame_size', default=2048),
            'hop_size': config.get('feature_extraction', 'hop_size', default=1024),
            'n_mfcc': config.get('feature_extraction', 'mfcc_coefficients', default=20)
        }
        analysis_version = config.get('feature_extraction', 'analysis_version', default='1.0.0')
        
        # Process files
        workers = determine_worker_count(request.workers)
        batch_size = config.get('performance', 'batch_size', default=100)
        
        if workers > 1:
            analyzed_count, error_count = process_files_parallel(
                files_to_analyze, extractor_config, workers, batch_size,
                db, analysis_version, request.skip_errors
            )
        else:
            analyzed_count, error_count = process_files_sequential(
                files_to_analyze, extractor_config, batch_size,
                db, analysis_version, request.skip_errors
            )
        
        task_status[task_id] = {
            "status": "complete",
            "analyzed": analyzed_count,
            "errors": error_count,
            "total": len(files_to_analyze),
            "progress": 100
        }
        
    except Exception as e:
        task_status[task_id] = {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a background task."""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = task_status[task_id]
    if status.get("total", 0) > 0:
        status["progress"] = int((status.get("completed", 0) / status["total"]) * 100)
    
    return status


@app.post("/api/cluster")
async def create_cluster(request: ClusterRequest):
    """Create a new clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Check if we have features
    feature_count = db.count_features()
    if feature_count == 0:
        raise HTTPException(status_code=400, detail="No analyzed tracks found. Run analysis first.")
    
    # Load features
    feature_matrix, track_ids = db.get_all_features()
    
    if len(feature_matrix) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tracks to cluster.")
    
    # Initialize clustering engine
    engine = ClusterEngine(
        min_clusters=request.min_size,
        max_clusters=request.max_clusters,
        detection_method=request.method,
        algorithm=request.algorithm
    )
    
    # Determine number of clusters
    if request.algorithm == 'hdbscan':
        # HDBSCAN doesn't need optimal k finding
        labels, centroids, cluster_metrics = engine.cluster(
            feature_matrix,
            show_progress=False,
            min_cluster_size=request.min_size,
            min_samples=request.min_samples,
            cluster_selection_epsilon=request.epsilon
        )
        n_clusters = len(centroids)
    else:
        if request.clusters:
            n_clusters = request.clusters
        else:
            optimal_k, metrics = engine.find_optimal_k(feature_matrix, show_progress=False)
            n_clusters = engine.apply_granularity(optimal_k, request.granularity)
        
        # Perform clustering
        labels, centroids, cluster_metrics = engine.cluster(
            feature_matrix, n_clusters, show_progress=False
        )
    
    # Find representatives
    representatives = engine.find_representative_tracks(feature_matrix, labels, centroids, track_ids)
    distances = engine.compute_distances_to_centroids(feature_matrix, labels, centroids)
    
    # Save to database
    params = {
        'granularity': request.granularity,
        'method': request.method,
        'min_size': request.min_size
    }
    if request.algorithm == 'hdbscan':
        params['epsilon'] = request.epsilon
        params['min_samples'] = request.min_samples
    
    clustering_id = db.add_clustering(
        name=request.name,
        algorithm=request.algorithm,
        num_clusters=n_clusters,
        parameters=json.dumps(params),
        silhouette_score=cluster_metrics.get('silhouette_score')
    )
    
    # For HDBSCAN, get unique labels (excluding noise)
    if request.algorithm == 'hdbscan':
        unique_labels = np.unique(labels[labels >= 0])
    else:
        unique_labels = range(n_clusters)
    
    # Save clusters
    for cluster_idx in unique_labels:
        cluster_mask = labels == cluster_idx
        cluster_size = int(np.sum(cluster_mask))
        
        if cluster_size == 0:
            continue
        
        rep_track_id = representatives.get(cluster_idx)
        
        # Get centroid for this cluster
        if request.algorithm == 'hdbscan':
            centroid_idx = list(unique_labels).index(cluster_idx)
        else:
            centroid_idx = cluster_idx
        
        cluster_id = db.add_cluster(
            clustering_id=clustering_id,
            cluster_index=int(cluster_idx),
            size=cluster_size,
            representative_track_id=rep_track_id,
            centroid=centroids[centroid_idx]
        )
        
        # Add members
        for track_id, label, distance in zip(track_ids, labels, distances):
            if label == cluster_idx:
                db.add_cluster_member(cluster_id, track_id, distance)
    
    result = {
        "clustering_id": clustering_id,
        "name": request.name,
        "num_clusters": n_clusters,
    }
    
    if request.show_metrics:
        result["metrics"] = cluster_metrics
    
    return result


@app.post("/api/classify")
async def classify_tracks(request: ClassifyRequest):
    """Classify new tracks to existing clusters."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering_info = db.get_clustering(name=request.clustering) if request.clustering else db.get_clustering()
    if not clustering_info:
        raise HTTPException(status_code=404, detail="No clustering found. Run clustering first.")
    
    # Load clusters and centroids
    clusters = db.get_clusters_by_clustering(clustering_info['id'])
    if not clusters:
        raise HTTPException(status_code=404, detail="No clusters found in this clustering.")
    
    # Build centroid matrix
    centroids = []
    cluster_ids = []
    for cluster in clusters:
        if cluster['centroid']:
            centroid = pickle.loads(cluster['centroid'])
            centroids.append(centroid)
            cluster_ids.append(cluster['id'])
    
    if not centroids:
        raise HTTPException(status_code=400, detail="No valid centroids found.")
    
    centroids = np.array(centroids)
    
    # Initialize classifier
    classifier = TrackClassifier(centroids, cluster_ids)
    
    # Find audio files to classify
    audio_files = find_audio_files(request.path, recursive=request.recursive)
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files found.")
    
    # Initialize extractor
    extractor_config = {
        'sample_rate': config.get('feature_extraction', 'sample_rate', default=44100),
        'frame_size': config.get('feature_extraction', 'frame_size', default=2048),
        'hop_size': config.get('feature_extraction', 'hop_size', default=1024),
        'n_mfcc': config.get('feature_extraction', 'mfcc_coefficients', default=20)
    }
    extractor = FeatureExtractor(**extractor_config)
    
    # Classify each file
    results = []
    for filepath in audio_files:
        # Extract features
        features = extractor.extract(filepath)
        if features is None:
            results.append({
                "filepath": filepath,
                "filename": Path(filepath).name,
                "status": "error",
                "error": "Failed to extract features"
            })
            continue
        
        # Classify
        cluster_id, distance = classifier.classify(features, request.threshold)
        
        if cluster_id == -1:
            results.append({
                "filepath": filepath,
                "filename": Path(filepath).name,
                "status": "no_match",
                "distance": float(distance)
            })
        else:
            # Get cluster info
            cluster_info = db.get_cluster(cluster_id)
            rep_track = db.get_track_by_id(cluster_info['representative_track_id']) if cluster_info['representative_track_id'] else None
            
            results.append({
                "filepath": filepath,
                "filename": Path(filepath).name,
                "status": "classified",
                "cluster_id": cluster_id,
                "cluster_index": cluster_info['cluster_index'],
                "cluster_name": cluster_info.get('name'),
                "distance": float(distance),
                "representative": rep_track['filename'] if rep_track else None
            })
    
    return {
        "clustering_id": clustering_info['id'],
        "clustering_name": clustering_info.get('name'),
        "results": results
    }


@app.post("/api/label-clusters")
async def label_clusters(request: LabelClustersRequest):
    """Auto-generate descriptive names for all clusters."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering = db.get_clustering(name=request.clustering_name)
    if not clustering:
        raise HTTPException(status_code=404, detail=f"Clustering '{request.clustering_name}' not found.")
    
    # Get clusters
    clusters = db.get_clusters_by_clustering(clustering['id'])
    if not clusters:
        raise HTTPException(status_code=400, detail="No clusters found.")
    
    # Unpickle centroids
    for cluster in clusters:
        if cluster['centroid']:
            cluster['centroid'] = pickle.loads(cluster['centroid'])
    
    # Load features and labels
    feature_matrix, track_ids = db.get_all_features()
    
    # Reconstruct labels from cluster membership
    labels = np.full(len(track_ids), -1, dtype=int)
    for cluster in clusters:
        members = db.get_cluster_members(cluster['id'])
        for member in members:
            try:
                track_idx = track_ids.index(member['id']) if 'id' in member else track_ids.index(member.get('track_id', -1))
                labels[track_idx] = cluster['cluster_index']
            except (ValueError, KeyError):
                continue
    
    # Generate names
    namer = ClusterNamer(
        include_genre=not request.no_genre,
        include_bpm=not request.no_bpm,
        use_bpm_range=not request.bpm_average,
        include_descriptors=not request.no_descriptors
    )
    names = namer.generate_names_for_clustering(clusters, feature_matrix, labels)
    
    # Save names if not dry run
    if not request.dry_run:
        for cluster in clusters:
            cluster_idx = cluster['cluster_index']
            if cluster_idx in names:
                db.update_cluster_name(cluster['id'], names[cluster_idx])
    
    # Format results
    results = []
    for cluster_idx, name in sorted(names.items()):
        cluster = next(c for c in clusters if c['cluster_index'] == cluster_idx)
        results.append({
            "cluster_index": cluster_idx,
            "name": name,
            "size": cluster['size']
        })
    
    return {
        "clustering_name": request.clustering_name,
        "dry_run": request.dry_run,
        "names": results
    }


@app.post("/api/export")
async def export_playlists(request: ExportRequest):
    """Export clusters as playlists."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get clustering
    clustering_info = db.get_clustering(name=request.clustering) if request.clustering else db.get_clustering()
    if not clustering_info:
        raise HTTPException(status_code=404, detail="No clustering found. Run clustering first.")
    
    # Get clusters
    clusters = db.get_clusters_by_clustering(clustering_info['id'])
    if not clusters:
        raise HTTPException(status_code=400, detail="No clusters found.")
    
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
        output_dir=request.output,
        playlist_format=request.format,
        relative_paths=request.relative_paths,
        include_representative=request.include_representative
    )
    
    created_files = exporter.export_all_clusters(clustering_info, clusters_data)
    
    return {
        "clustering_id": clustering_info['id'],
        "clustering_name": clustering_info.get('name'),
        "output_dir": request.output,
        "files_created": len(created_files),
        "files": created_files
    }


@app.get("/api/search")
async def search_tracks(query: str, clustering: Optional[str] = None, limit: int = 20):
    """Search for tracks."""
    config = Config()
    db = Database(config.get_db_path())
    
    all_tracks = db.get_all_tracks()
    query_lower = query.lower()
    
    matching_tracks = [
        t for t in all_tracks
        if query_lower in t['filename'].lower() or query_lower in t['filepath'].lower()
    ]
    
    results = matching_tracks[:limit]
    
    # If clustering specified, add cluster info
    if clustering:
        clustering_info = db.get_clustering(name=clustering)
        if clustering_info:
            for track in results:
                cluster_id = db.get_track_cluster(track['id'], clustering_info['id'])
                if cluster_id:
                    cluster = db.get_cluster(cluster_id)
                    track['cluster'] = {
                        "id": cluster_id,
                        "index": cluster['cluster_index'],
                        "name": cluster.get('name')
                    }
    
    return {
        "query": query,
        "total": len(matching_tracks),
        "limit": limit,
        "tracks": results
    }


@app.post("/api/compare")
async def compare_clusterings(clustering1: str, clustering2: str):
    """Compare two clusterings side-by-side."""
    config = Config()
    db = Database(config.get_db_path())
    
    # Get both clusterings
    c1 = db.get_clustering(name=clustering1)
    c2 = db.get_clustering(name=clustering2)
    
    if not c1:
        raise HTTPException(status_code=404, detail=f"Clustering '{clustering1}' not found.")
    if not c2:
        raise HTTPException(status_code=404, detail=f"Clustering '{clustering2}' not found.")
    
    clusters1 = db.get_clusters_by_clustering(c1['id'])
    clusters2 = db.get_clusters_by_clustering(c2['id'])
    
    # Cluster size stats
    sizes1 = [c['size'] for c in clusters1]
    sizes2 = [c['size'] for c in clusters2]
    
    return {
        "clustering1": {
            "name": clustering1,
            "id": c1['id'],
            "algorithm": c1.get('algorithm', 'kmeans'),
            "num_clusters": c1['num_clusters'],
            "silhouette_score": c1.get('silhouette_score'),
            "size_distribution": {
                "min": int(min(sizes1)) if sizes1 else 0,
                "max": int(max(sizes1)) if sizes1 else 0,
                "mean": float(np.mean(sizes1)) if sizes1 else 0,
                "std": float(np.std(sizes1)) if sizes1 else 0
            }
        },
        "clustering2": {
            "name": clustering2,
            "id": c2['id'],
            "algorithm": c2.get('algorithm', 'kmeans'),
            "num_clusters": c2['num_clusters'],
            "silhouette_score": c2.get('silhouette_score'),
            "size_distribution": {
                "min": int(min(sizes2)) if sizes2 else 0,
                "max": int(max(sizes2)) if sizes2 else 0,
                "mean": float(np.mean(sizes2)) if sizes2 else 0,
                "std": float(np.std(sizes2)) if sizes2 else 0
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
