"""Unit tests for genre_classifier module."""
import numpy as np
import pytest
from music_cluster.genre_classifier import GenreClassifier


class TestGenreClassifier:
    """Test genre classification logic."""
    
    @pytest.fixture
    def classifier(self):
        """Create a genre classifier instance."""
        return GenreClassifier()
    
    def test_techno_classification(self, classifier, sample_features):
        """Test classification of techno tracks."""
        # Techno: 128+ BPM, high bass
        bpm = 130.0
        energy = 0.15
        bass = 0.20
        brightness = 0.10
        genre = classifier.classify_genre(sample_features, bpm, energy, bass, brightness)
        assert genre == 'Techno', f"Expected Techno, got {genre}"
    
    def test_house_classification(self, classifier, sample_features):
        """Test classification of house tracks."""
        genre = classifier.classify_genre(sample_features, 124.0, 0.13, 0.10, 0.12)
        assert genre in ['House', 'Tech House', 'Deep House'], f"Got {genre}"
    
    def test_dnb_classification(self, classifier, sample_features):
        """Test classification of drum & bass tracks."""
        genre = classifier.classify_genre(sample_features, 174.0, 0.15, 0.15, 0.10)
        assert genre == 'Jungle', f"Expected Jungle, got {genre}"
    
    def test_dubstep_classification(self, classifier, sample_features):
        """Test classification of dubstep tracks."""
        genre = classifier.classify_genre(sample_features, 75.0, 0.10, 0.25, 0.05)
        assert genre == 'Dubstep', f"Expected Dubstep, got {genre}"
    
    def test_jungle_classification(self, classifier, sample_features):
        """Test classification of jungle tracks."""
        genre = classifier.classify_genre(sample_features, 170.0, 0.15, 0.15, 0.10)
        assert genre == 'Jungle', f"Expected Jungle, got {genre}"
    
    def test_hip_hop_classification(self, classifier, sample_features):
        """Test classification of hip-hop tracks."""
        genre = classifier.classify_genre(sample_features, 85.0, 0.10, 0.10, 0.08)
        assert genre == 'Hip-Hop', f"Expected Hip-Hop, got {genre}"
    
    def test_uk_garage_classification(self, classifier, sample_features):
        """Test classification of UK garage tracks."""
        genre = classifier.classify_genre(sample_features, 138.0, 0.12, 0.18, 0.10)
        assert genre == 'UK Garage', f"Expected UK Garage, got {genre}"
    
    def test_dnb_fast_classification(self, classifier, sample_features):
        """Test classification of fast drum & bass tracks."""
        genre = classifier.classify_genre(sample_features, 155.0, 0.15, 0.15, 0.12)
        assert genre == 'Drum & Bass', f"Expected Drum & Bass, got {genre}"
    
    def test_ambient_classification(self, classifier, sample_features):
        """Test classification of ambient tracks."""
        # BPM 75 falls in the 70-90 range where energy < 0.08 returns Ambient
        genre = classifier.classify_genre(sample_features, 75.0, 0.05, 0.10, 0.08)
        assert genre == 'Ambient', f"Expected Ambient, got {genre}"
    
    def test_hardcore_classification(self, classifier, sample_features):
        """Test classification of hardcore tracks."""
        genre = classifier.classify_genre(sample_features, 185.0, 0.20, 0.15, 0.12)
        assert genre == 'Hardcore', f"Expected Hardcore, got {genre}"
    
    def test_breaks_classification(self, classifier, sample_features):
        """Test classification of breaks tracks."""
        genre = classifier.classify_genre(sample_features, 138.0, 0.18, 0.12, 0.10)
        assert genre == 'Breaks', f"Expected Breaks, got {genre}"
    
    def test_edge_case_very_slow_bpm(self, classifier, sample_features):
        """Test classification with very slow BPM."""
        genre = classifier.classify_genre(sample_features, 40.0, 0.08, 0.10, 0.08)
        assert genre in ['Ambient', 'Experimental'], f"Got {genre}"
    
    def test_edge_case_very_fast_bpm(self, classifier, sample_features):
        """Test classification with very fast BPM."""
        genre = classifier.classify_genre(sample_features, 200.0, 0.20, 0.15, 0.12)
        assert genre == 'Hardcore', f"Expected Hardcore, got {genre}"
    
    def test_genre_consistency(self, classifier, sample_features):
        """Test that same features produce same genre."""
        genre1 = classifier.classify_genre(sample_features, 125.0, 0.15, 0.18, 0.10)
        genre2 = classifier.classify_genre(sample_features, 125.0, 0.15, 0.18, 0.10)
        assert genre1 == genre2, "Genre classification should be deterministic"
    
    def test_tech_house_classification(self, classifier, sample_features):
        """Test classification of tech house tracks."""
        genre = classifier.classify_genre(sample_features, 125.0, 0.15, 0.18, 0.10)
        assert genre == 'Tech House', f"Expected Tech House, got {genre}"
    
    def test_deep_house_classification(self, classifier, sample_features):
        """Test classification of deep house tracks."""
        genre = classifier.classify_genre(sample_features, 119.0, 0.12, 0.18, 0.08)
        assert genre == 'Deep House', f"Expected Deep House, got {genre}"
    
    def test_returns_string(self, classifier, sample_features):
        """Test that classification always returns a string."""
        for bpm in [70, 90, 120, 140, 170]:
            genre = classifier.classify_genre(sample_features, bpm, 0.12, 0.15, 0.10)
            assert isinstance(genre, str), f"Expected string, got {type(genre)}"
            assert len(genre) > 0, "Genre string should not be empty"
    
    def test_subgenre_modifier_bright(self, classifier):
        """Test bright subgenre modifier."""
        modifier = classifier.get_subgenre_modifier(0.10, 0.20, 0.10)
        assert modifier == 'Bright', f"Expected 'Bright', got '{modifier}'"
    
    def test_subgenre_modifier_dark(self, classifier):
        """Test dark subgenre modifier."""
        modifier = classifier.get_subgenre_modifier(0.10, 0.05, 0.10)
        assert modifier == 'Dark', f"Expected 'Dark', got '{modifier}'"
    
    def test_subgenre_modifier_minimal(self, classifier):
        """Test minimal subgenre modifier."""
        modifier = classifier.get_subgenre_modifier(0.10, 0.10, 0.03)
        assert modifier == 'Minimal', f"Expected 'Minimal', got '{modifier}'"
    
    def test_subgenre_modifier_industrial(self, classifier):
        """Test industrial subgenre modifier."""
        modifier = classifier.get_subgenre_modifier(0.10, 0.10, 0.20)
        assert modifier == 'Industrial', f"Expected 'Industrial', got '{modifier}'"
