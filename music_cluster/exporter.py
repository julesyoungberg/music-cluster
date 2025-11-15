"""Playlist exporter for music-cluster."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .utils import get_relative_path, ensure_directory, format_duration


class PlaylistExporter:
    """Export clusters as playlists."""
    
    def __init__(self, output_dir: str, playlist_format: str = "m3u",
                 relative_paths: bool = False, include_representative: bool = True):
        """Initialize playlist exporter.
        
        Args:
            output_dir: Directory to save playlists
            playlist_format: Format (m3u, m3u8, json)
            relative_paths: Use relative paths in playlists
            include_representative: Include representative track first
        """
        self.output_dir = output_dir
        self.playlist_format = playlist_format.lower()
        self.relative_paths = relative_paths
        self.include_representative = include_representative
        
        # Create output directory
        ensure_directory(self.output_dir)
    
    def export_cluster(self, cluster_info: Dict, tracks: List[Dict],
                      representative_track: Optional[Dict] = None) -> str:
        """Export a single cluster as a playlist.
        
        Args:
            cluster_info: Dictionary with cluster metadata
            tracks: List of track dictionaries
            representative_track: Optional representative track dict
            
        Returns:
            Path to created playlist file
        """
        # Generate filename
        cluster_idx = cluster_info.get('cluster_index', 0)
        filename = f"cluster_{cluster_idx:02d}.{self.playlist_format}"
        filepath = os.path.join(self.output_dir, filename)
        
        # Sort tracks by distance to centroid (if available)
        if tracks and 'distance_to_centroid' in tracks[0]:
            tracks = sorted(tracks, key=lambda t: t.get('distance_to_centroid', 0))
        
        # Add representative track first if requested
        if self.include_representative and representative_track:
            # Remove representative from tracks if it's already there
            tracks = [t for t in tracks if t['id'] != representative_track['id']]
            # Add at beginning
            tracks = [representative_track] + tracks
        
        # Export based on format
        if self.playlist_format in ['m3u', 'm3u8']:
            self._export_m3u(filepath, cluster_info, tracks, representative_track)
        elif self.playlist_format == 'json':
            self._export_json(filepath, cluster_info, tracks, representative_track)
        else:
            raise ValueError(f"Unsupported playlist format: {self.playlist_format}")
        
        return filepath
    
    def _export_m3u(self, filepath: str, cluster_info: Dict, tracks: List[Dict],
                   representative_track: Optional[Dict]) -> None:
        """Export as M3U playlist.
        
        Args:
            filepath: Output file path
            cluster_info: Cluster metadata
            tracks: List of tracks
            representative_track: Representative track
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("#EXTM3U\n")
            f.write(f"# Music Cluster Playlist - Cluster {cluster_info.get('cluster_index', 0)}\n")
            f.write(f"# Size: {len(tracks)} tracks\n")
            
            if representative_track:
                f.write(f"# Representative: {representative_track['filename']}\n")
            
            if 'silhouette_score' in cluster_info:
                f.write(f"# Quality Score: {cluster_info['silhouette_score']:.3f}\n")
            
            f.write("\n")
            
            # Write tracks
            for track in tracks:
                # Get track path
                track_path = track['filepath']
                if self.relative_paths:
                    track_path = get_relative_path(track_path, self.output_dir)
                
                # Write extended info if available
                duration = track.get('duration')
                if duration:
                    duration_sec = int(duration)
                    f.write(f"#EXTINF:{duration_sec},{track['filename']}\n")
                
                f.write(f"{track_path}\n")
    
    def _export_json(self, filepath: str, cluster_info: Dict, tracks: List[Dict],
                    representative_track: Optional[Dict]) -> None:
        """Export as JSON.
        
        Args:
            filepath: Output file path
            cluster_info: Cluster metadata
            tracks: List of tracks
            representative_track: Representative track
        """
        data = {
            "cluster": {
                "index": cluster_info.get('cluster_index', 0),
                "size": len(tracks),
                "id": cluster_info.get('id')
            },
            "representative_track": {
                "id": representative_track['id'],
                "filepath": representative_track['filepath'],
                "filename": representative_track['filename']
            } if representative_track else None,
            "tracks": [
                {
                    "id": track['id'],
                    "filepath": track['filepath'] if not self.relative_paths 
                               else get_relative_path(track['filepath'], self.output_dir),
                    "filename": track['filename'],
                    "duration": track.get('duration'),
                    "distance_to_centroid": track.get('distance_to_centroid')
                }
                for track in tracks
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def export_all_clusters(self, clustering_info: Dict, clusters_data: List[Dict]) -> List[str]:
        """Export all clusters from a clustering.
        
        Args:
            clustering_info: Clustering metadata
            clusters_data: List of dictionaries with cluster info, tracks, and representative
            
        Returns:
            List of created playlist file paths
        """
        created_files = []
        
        for cluster_data in clusters_data:
            cluster_info = cluster_data['cluster']
            tracks = cluster_data['tracks']
            representative = cluster_data.get('representative')
            
            filepath = self.export_cluster(cluster_info, tracks, representative)
            created_files.append(filepath)
        
        # Also create a summary file
        summary_path = self._create_summary(clustering_info, clusters_data)
        created_files.append(summary_path)
        
        return created_files
    
    def _create_summary(self, clustering_info: Dict, clusters_data: List[Dict]) -> str:
        """Create a summary file for the clustering.
        
        Args:
            clustering_info: Clustering metadata
            clusters_data: List of cluster data
            
        Returns:
            Path to summary file
        """
        summary_path = os.path.join(self.output_dir, "clustering_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Music Clustering Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Clustering Name: {clustering_info.get('name', 'Unnamed')}\n")
            f.write(f"Algorithm: {clustering_info.get('algorithm', 'kmeans')}\n")
            f.write(f"Number of Clusters: {clustering_info.get('num_clusters', len(clusters_data))}\n")
            
            if 'silhouette_score' in clustering_info and clustering_info['silhouette_score']:
                f.write(f"Silhouette Score: {clustering_info['silhouette_score']:.3f}\n")
            
            f.write(f"Created: {clustering_info.get('created_at', 'Unknown')}\n")
            f.write("\n")
            
            f.write("Clusters:\n")
            f.write("-" * 50 + "\n")
            
            for cluster_data in clusters_data:
                cluster_info = cluster_data['cluster']
                tracks = cluster_data['tracks']
                representative = cluster_data.get('representative')
                
                cluster_idx = cluster_info.get('cluster_index', 0)
                f.write(f"\nCluster {cluster_idx}:\n")
                f.write(f"  Size: {len(tracks)} tracks\n")
                
                if representative:
                    f.write(f"  Representative: {representative['filename']}\n")
                
                # Show a few sample tracks
                sample_count = min(5, len(tracks))
                if sample_count > 0:
                    f.write(f"  Sample tracks:\n")
                    for i, track in enumerate(tracks[:sample_count]):
                        f.write(f"    - {track['filename']}\n")
        
        return summary_path
