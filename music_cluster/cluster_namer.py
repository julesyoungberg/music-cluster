"""Cluster naming utilities for music-cluster."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .genre_classifier import GenreClassifier


class ClusterNamer:
    """Generate descriptive names for music clusters based on audio features."""
    
    # Feature indices (based on FeatureExtractor structure)
    # After aggregated frame features (mean+std), we have:
    # Rhythmic features at the end: BPM, onset strength stats, then high-level features
    BPM_INDEX = -10  # BPM is 10th from the end in feature vector
    
    def __init__(self, include_genre: bool = True, include_bpm: bool = True, 
                 use_bpm_range: bool = True, include_descriptors: bool = True):
        """Initialize cluster namer.
        
        Args:
            include_genre: Include genre classification in names
            include_bpm: Include BPM information in names
            use_bpm_range: Use BPM range instead of average (e.g., "120-130 BPM")
            include_descriptors: Include characteristic descriptors (Bass-Heavy, Dark, etc.)
        """
        self.cluster_stats = {}  # Store stats for comparative naming
        self.genre_classifier = GenreClassifier()
        self.include_genre = include_genre
        self.include_bpm = include_bpm
        self.use_bpm_range = use_bpm_range
        self.include_descriptors = include_descriptors
    
    def generate_name(self, centroid: np.ndarray, cluster_features: List[np.ndarray],
                     cluster_size: int) -> str:
        """Generate a descriptive name for a cluster.
        
        Args:
            centroid: Cluster centroid feature vector
            cluster_features: List of feature vectors in the cluster
            cluster_size: Number of tracks in cluster
            
        Returns:
            Generated cluster name
        """
        # Analyze centroid characteristics
        energy_level = self._classify_energy(centroid)
        spectral_char = self._classify_spectral_characteristics(centroid)
        rhythm_char = self._classify_rhythm(centroid)
        
        # Build name from characteristics
        name_parts = []
        
        # Energy descriptor
        if energy_level:
            name_parts.append(energy_level)
        
        # Spectral/timbral descriptor  
        if spectral_char:
            name_parts.append(spectral_char)
        
        # Rhythm descriptor
        if rhythm_char:
            name_parts.append(rhythm_char)
        
        # Fallback if no characteristics detected
        if not name_parts:
            name_parts = ["Electronic"]
        
        return " ".join(name_parts)
    
    def _classify_energy(self, centroid: np.ndarray) -> Optional[str]:
        """Classify energy level from features.
        
        Args:
            centroid: Feature vector
            
        Returns:
            Energy descriptor or None
        """
        # Assume first features are MFCCs, and later features include energy
        # This is a simplified heuristic based on typical feature ordering
        if len(centroid) < 20:
            return None
        
        # Use spectral features (typically after MFCCs)
        spectral_features = centroid[20:40] if len(centroid) > 40 else centroid[20:]
        energy = np.mean(np.abs(spectral_features))
        
        if energy > 0.3:
            return "High-Energy"
        elif energy > 0.15:
            return "Mid-Energy"
        elif energy > 0.05:
            return "Low-Energy"
        else:
            return "Minimal"
    
    def _classify_spectral_characteristics(self, centroid: np.ndarray) -> Optional[str]:
        """Classify spectral/timbral characteristics.
        
        Args:
            centroid: Feature vector
            
        Returns:
            Spectral descriptor or None
        """
        if len(centroid) < 40:
            return None
        
        # Analyze spectral content
        spectral_features = centroid[20:60] if len(centroid) > 60 else centroid[20:]
        
        # Brightness (high-frequency content)
        brightness = np.mean(spectral_features[:len(spectral_features)//2])
        
        # Spectral flatness/variation
        spectral_var = np.std(spectral_features)
        
        descriptors = []
        
        # Brightness-based descriptors
        if brightness > 0.2:
            descriptors.append("Bright")
        elif brightness < 0.05:
            descriptors.append("Dark")
        
        # Complexity-based descriptors  
        if spectral_var > 0.15:
            descriptors.append("Complex")
        elif spectral_var < 0.05:
            descriptors.append("Clean")
        
        return " ".join(descriptors) if descriptors else None
    
    def _classify_rhythm(self, centroid: np.ndarray) -> Optional[str]:
        """Classify rhythmic characteristics.
        
        Args:
            centroid: Feature vector
            
        Returns:
            Rhythm descriptor or None
        """
        if len(centroid) < 60:
            return None
        
        # Assume rhythm features are in the latter part of the vector
        rhythm_features = centroid[60:] if len(centroid) > 60 else centroid[-20:]
        
        # Rhythmic complexity
        rhythm_complexity = np.std(rhythm_features)
        
        if rhythm_complexity > 0.2:
            return "Driving"
        elif rhythm_complexity > 0.1:
            return "Groovy"
        elif rhythm_complexity < 0.05:
            return "Steady"
        
        return None
    
    def generate_names_for_clustering(self, clusters: List[Dict], 
                                     all_features: np.ndarray,
                                     all_labels: np.ndarray) -> Dict[int, str]:
        """Generate names for all clusters in a clustering using comparative analysis.
        
        Args:
            clusters: List of cluster dictionaries with centroids
            all_features: All feature vectors
            all_labels: Cluster labels for all features
            
        Returns:
            Dictionary mapping cluster_index to generated name
        """
        # First pass: compute statistics for all clusters
        cluster_stats = {}
        for cluster in clusters:
            cluster_idx = cluster['cluster_index']
            centroid = cluster.get('centroid')
            
            if centroid is None:
                continue
            
            cluster_mask = all_labels == cluster_idx
            cluster_features = all_features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            # Extract BPM from features (last rhythmic features section)
            # BPM_INDEX is -10, so we need at least 10 columns to access index -10
            bpms = cluster_features[:, self.BPM_INDEX] if cluster_features.shape[1] >= abs(self.BPM_INDEX) else None
            
            if bpms is not None and len(bpms) > 0:
                # Filter out outliers (use percentiles to avoid extreme values)
                bpm_p10 = np.percentile(bpms, 10)
                bpm_p90 = np.percentile(bpms, 90)
                bpms_filtered = bpms[(bpms >= bpm_p10) & (bpms <= bpm_p90)]
                
                if len(bpms_filtered) > 0:
                    bpm_mean = np.mean(bpms_filtered)
                    bpm_min = np.min(bpms_filtered)
                    bpm_max = np.max(bpms_filtered)
                    bpm_std = np.std(bpms_filtered)
                else:
                    bpm_mean = bpm_min = bpm_max = bpm_std = 0
            else:
                bpm_mean = bpm_min = bpm_max = bpm_std = 0
            
            # Extract energy (aggregated RMS features are in the middle section)
            # Slice [40:60] requires at least 60 elements (indices 0-59)
            # Fallback to [20:40] requires at least 40 elements
            # If still insufficient, use available elements or default to 0
            if len(centroid) >= 60:
                energy_features = centroid[40:60]
            elif len(centroid) >= 40:
                energy_features = centroid[20:40]
            elif len(centroid) >= 20:
                energy_features = centroid[10:len(centroid)]
            else:
                energy_features = np.array([])
            
            energy = np.mean(np.abs(energy_features)) if len(energy_features) > 0 else 0.0
            
            # Extract spectral characteristics
            # Slice [20:40] requires at least 40 elements (indices 0-39)
            # Fallback to [10:20] requires at least 20 elements
            if len(centroid) >= 40:
                spectral_features = centroid[20:40]
            elif len(centroid) >= 20:
                spectral_features = centroid[10:20]
            elif len(centroid) >= 10:
                spectral_features = centroid[5:len(centroid)]
            else:
                spectral_features = np.array([])
            
            if len(spectral_features) > 0:
                brightness = np.mean(spectral_features[:len(spectral_features)//2]) if len(spectral_features) > 1 else np.mean(spectral_features)
                spectral_complexity = np.std(spectral_features)
            else:
                brightness = 0.0
                spectral_complexity = 0.0
            
            # Bass presence (low MFCC coefficients typically represent low frequencies)
            # Slice [2:5] requires at least 5 elements (indices 0-4)
            if len(centroid) >= 5:
                bass_presence = np.mean(np.abs(centroid[2:5]))
            elif len(centroid) >= 3:
                bass_presence = np.mean(np.abs(centroid[2:len(centroid)]))
            else:
                bass_presence = 0.0
            
            cluster_stats[cluster_idx] = {
                'bpm': bpm_mean,
                'bpm_min': bpm_min,
                'bpm_max': bpm_max,
                'bpm_std': bpm_std,
                'energy': energy,
                'brightness': brightness,
                'complexity': spectral_complexity,
                'bass': bass_presence,
                'size': cluster['size']
            }
        
        # Compute global statistics for comparative naming
        all_bpms = [s['bpm'] for s in cluster_stats.values() if s['bpm'] > 0]
        all_energies = [s['energy'] for s in cluster_stats.values()]
        all_brightness = [s['brightness'] for s in cluster_stats.values()]
        all_bass = [s['bass'] for s in cluster_stats.values()]
        
        median_bpm = np.median(all_bpms) if all_bpms else 120
        median_energy = np.median(all_energies) if all_energies else 0.1
        median_brightness = np.median(all_brightness) if all_brightness else 0.1
        median_bass = np.median(all_bass) if all_bass else 0.1
        
        # Create cluster lookup dict
        clusters_dict = {c['cluster_index']: c for c in clusters}
        
        # Second pass: generate names with genre classification
        names = {}
        for cluster_idx, stats in cluster_stats.items():
            parts = []
            
            # Get genre classification
            if self.include_genre and stats['bpm'] > 0 and cluster_idx in clusters_dict:
                cluster_centroid = clusters_dict[cluster_idx].get('centroid')
                # Only classify if centroid is available and valid
                if cluster_centroid is not None and isinstance(cluster_centroid, np.ndarray) and len(cluster_centroid) > 0:
                    genre = self.genre_classifier.classify_genre(
                        cluster_centroid,
                        stats['bpm'],
                        stats['energy'],
                        stats['bass'],
                        stats['brightness']
                    )
                    parts.append(genre)
            
            # Add BPM
            if self.include_bpm and stats['bpm'] > 0:
                if self.use_bpm_range and stats['bpm_std'] > 3:
                    # Show range if there's significant variation
                    bpm_min_rounded = int(stats['bpm_min'] / 5) * 5  # Round to nearest 5
                    bpm_max_rounded = int((stats['bpm_max'] + 4) / 5) * 5  # Round up to nearest 5
                    if bpm_max_rounded - bpm_min_rounded >= 10:
                        parts.append(f"{bpm_min_rounded}-{bpm_max_rounded} BPM")
                    else:
                        parts.append(f"{int(stats['bpm'])} BPM")
                else:
                    bpm_val = int(stats['bpm'])
                    parts.append(f"{bpm_val} BPM")
            
            # Add distinctive characteristics (only if significantly different from median)
            if self.include_descriptors:
                descriptors = []
                
                # Bass descriptor (if significantly different)
                if stats['bass'] > median_bass * 1.4:
                    descriptors.append("Bass-Heavy")
                elif stats['bass'] < median_bass * 0.6:
                    descriptors.append("Light")
                
                # Brightness descriptor (if significantly different)
                if stats['brightness'] > median_brightness * 1.3:
                    descriptors.append("Bright")
                elif stats['brightness'] < median_brightness * 0.7:
                    descriptors.append("Dark")
                
                # Complexity descriptor (if significant)
                if stats['complexity'] > 0.18:
                    descriptors.append("Complex")
                elif stats['complexity'] < 0.04:
                    descriptors.append("Minimal")
                
                # Add descriptors to name
                if descriptors:
                    parts.extend(descriptors)
            
            # Build name
            if parts:
                names[cluster_idx] = " ".join(parts)
            else:
                names[cluster_idx] = f"Cluster {cluster_idx}"
        
        # Add names for clusters without stats
        for cluster in clusters:
            if cluster['cluster_index'] not in names:
                names[cluster['cluster_index']] = f"Cluster {cluster['cluster_index']}"
        
        # Make unique
        names = self._make_names_unique(names)
        
        return names
    
    def _make_names_unique(self, names: Dict[int, str]) -> Dict[int, str]:
        """Ensure all cluster names are unique by adding suffixes.
        
        Args:
            names: Dictionary of cluster_index to name
            
        Returns:
            Dictionary with unique names
        """
        name_counts = {}
        unique_names = {}
        
        for cluster_idx, name in names.items():
            if name not in name_counts:
                name_counts[name] = 0
                unique_names[cluster_idx] = name
            else:
                name_counts[name] += 1
                unique_names[cluster_idx] = f"{name} {name_counts[name]}"
        
        return unique_names
