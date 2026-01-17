"""
Example FastAPI REST API for Music Cluster UI.

This demonstrates how to wrap your existing CLI functionality
in a REST API that a frontend can consume.

To run:
    pip install fastapi uvicorn
    uvicorn music_cluster.api_example:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path
import numpy as np

from .config import Config
from .database import Database
from .clustering import ClusterEngine
from .extractor import FeatureExtractor
from .utils import find_audio_files

app = FastAPI(title="Music Cluster API", version="1.0.0")

# Enable CORS for frontend (Tauri, Electron, or web browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    path: str
    recursive: bool = True
    update: bool = False
    extensions: str = "mp3,flac,wav,m4a,ogg"
    workers: int = -1


class ClusterRequest(BaseModel):
    name: Optional[str] = None
    clusters: Optional[int] = None
    granularity: str = "normal"
    algorithm: str = "kmeans"
    min_size: int = 3
    show_metrics: bool = False


class ClassifyRequest(BaseModel):
    path: str
    recursive: bool = True
    clustering: Optional[str] = None
    threshold: Optional[float] = None


# In-memory task status (in production, use Redis or database)
task_status: Dict[str, Dict[str, Any]] = {}


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
    
    clustering = db.get_clustering(id=clustering_id)
    if not clustering:
        raise HTTPException(status_code=404, detail="Clustering not found")
    
    clusters = db.get_clusters_by_clustering(clustering_id)
    
    return {
        **clustering,
        "clusters": clusters
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
    
    return {
        **clustering,
        "clusters": clusters
    }


@app.get("/api/clusters/{cluster_id}")
async def get_cluster(cluster_id: int):
    """Get a specific cluster with its tracks."""
    config = Config()
    db = Database(config.get_db_path())
    
    cluster = db.get_cluster(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    members = db.get_cluster_members(cluster_id)
    
    return {
        **cluster,
        "tracks": members
    }


@app.get("/api/stats/{clustering_id}")
async def get_stats(clustering_id: int):
    """Get statistics for a clustering."""
    config = Config()
    db = Database(config.get_db_path())
    
    clustering = db.get_clustering(id=clustering_id)
    if not clustering:
        raise HTTPException(status_code=404, detail="Clustering not found")
    
    clusters = db.get_clusters_by_clustering(clustering_id)
    sizes = [c['size'] for c in clusters if c['size'] > 0]
    
    if not sizes:
        raise HTTPException(status_code=400, detail="No valid clusters found")
    
    return {
        "clustering_id": clustering_id,
        "algorithm": clustering.get('algorithm', 'kmeans'),
        "num_clusters": len(clusters),
        "total_tracks": sum(sizes),
        "silhouette_score": clustering.get('silhouette_score'),
        "size_distribution": {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes))
        }
    }


@app.post("/api/analyze")
async def analyze_tracks(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start analyzing audio files (runs in background)."""
    task_id = f"analyze_{Path(request.path).name}"
    
    # Check if path exists
    if not Path(request.path).exists():
        raise HTTPException(status_code=400, detail="Path does not exist")
    
    # Initialize task status
    task_status[task_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "completed": 0,
        "errors": 0
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
        ext_set = set(request.extensions.split(','))
        audio_files = find_audio_files(request.path, recursive=request.recursive, extensions=ext_set)
        
        if not audio_files:
            task_status[task_id] = {"status": "error", "message": "No audio files found"}
            return
        
        # Filter files
        files_to_analyze = filter_files_to_analyze(audio_files, db, request.update)
        
        if not files_to_analyze:
            task_status[task_id] = {"status": "complete", "message": "All files already analyzed"}
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
                db, analysis_version, skip_errors=True
            )
        else:
            analyzed_count, error_count = process_files_sequential(
                files_to_analyze, extractor_config, batch_size,
                db, analysis_version, skip_errors=True
            )
        
        task_status[task_id] = {
            "status": "complete",
            "analyzed": analyzed_count,
            "errors": error_count,
            "total": len(files_to_analyze)
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
    
    return task_status[task_id]


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
        max_clusters=100,
        detection_method='silhouette',
        algorithm=request.algorithm
    )
    
    # Determine number of clusters
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
    import json
    params = {
        'granularity': request.granularity,
        'method': 'silhouette',
        'min_size': request.min_size
    }
    
    clustering_id = db.add_clustering(
        name=request.name,
        algorithm=request.algorithm,
        num_clusters=n_clusters,
        parameters=json.dumps(params),
        silhouette_score=cluster_metrics.get('silhouette_score')
    )
    
    # Save clusters
    for cluster_idx in range(n_clusters):
        cluster_mask = labels == cluster_idx
        cluster_size = int(np.sum(cluster_mask))
        
        if cluster_size == 0:
            continue
        
        rep_track_id = representatives.get(cluster_idx)
        
        cluster_id = db.add_cluster(
            clustering_id=clustering_id,
            cluster_index=int(cluster_idx),
            size=cluster_size,
            representative_track_id=rep_track_id,
            centroid=centroids[cluster_idx]
        )
        
        # Add members
        for track_id, label, distance in zip(track_ids, labels, distances):
            if label == cluster_idx:
                db.add_cluster_member(cluster_id, track_id, distance)
    
    return {
        "clustering_id": clustering_id,
        "name": request.name,
        "num_clusters": n_clusters,
        "metrics": cluster_metrics if request.show_metrics else None
    }


@app.post("/api/search")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
