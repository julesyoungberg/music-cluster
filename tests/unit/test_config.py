"""Configuration layering, persistence, and the paths derived from it."""

import os

import pytest
import yaml

from music_cluster.config import DEFAULT_CONFIG, Config


class TestDefaults:
    def test_an_absent_file_yields_the_defaults(self, temp_dir):
        config = Config(str(temp_dir / "nothing-here.yaml"))
        assert config.get("sorting", "neighbors") == DEFAULT_CONFIG["sorting"]["neighbors"]

    def test_defaults_are_not_shared_between_instances(self, temp_dir):
        first = Config(str(temp_dir / "a.yaml"))
        second = Config(str(temp_dir / "b.yaml"))

        first.set(["sorting", "neighbors"], 99)

        # A mutation through one Config leaking into DEFAULT_CONFIG would give
        # every later instance the wrong defaults.
        assert second.get("sorting", "neighbors") != 99
        assert DEFAULT_CONFIG["sorting"]["neighbors"] != 99

    def test_get_returns_the_supplied_default_for_an_unknown_key(self, temp_dir):
        config = Config(str(temp_dir / "a.yaml"))
        assert config.get("sorting", "no_such_key", default="fallback") == "fallback"
        assert config.get("no_such_section", default=None) is None


class TestUserOverrides:
    def test_user_values_override_defaults_key_by_key(self, temp_dir):
        path = temp_dir / "config.yaml"
        path.write_text(yaml.dump({"sorting": {"neighbors": 11}}))

        config = Config(str(path))

        assert config.get("sorting", "neighbors") == 11
        # Untouched keys in the same section keep their defaults rather than
        # disappearing along with the rest of the section.
        assert config.get("sorting", "knn_weight") == DEFAULT_CONFIG["sorting"]["knn_weight"]

    def test_nested_sections_merge_rather_than_replace(self, temp_dir):
        path = temp_dir / "config.yaml"
        path.write_text(yaml.dump({"sorting": {"feature_weights": {"rhythm": 2.0}}}))

        weights = Config(str(path)).get("sorting", "feature_weights")

        assert weights["rhythm"] == 2.0
        assert weights["timbre"] == 1.0

    def test_an_unreadable_file_falls_back_to_defaults(self, temp_dir, capsys):
        path = temp_dir / "config.yaml"
        path.write_text("{{{ not yaml")

        config = Config(str(path))

        assert config.get("sorting", "neighbors") == DEFAULT_CONFIG["sorting"]["neighbors"]
        assert "Warning" in capsys.readouterr().out

    def test_an_empty_file_is_treated_as_no_overrides(self, temp_dir):
        path = temp_dir / "config.yaml"
        path.write_text("")

        assert Config(str(path)).get("sorting", "neighbors")


class TestSetAndSave:
    def test_set_creates_intermediate_sections(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["brand", "new", "key"], 5)
        assert config.get("brand", "new", "key") == 5

    def test_set_replaces_a_scalar_standing_where_a_section_is_needed(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["thing"], "scalar")
        config.set(["thing", "nested"], 1)
        assert config.get("thing", "nested") == 1

    def test_saved_values_survive_a_reload(self, temp_dir):
        path = str(temp_dir / "nested" / "config.yaml")

        config = Config(path)
        config.set(["sorting", "auto_accept_confidence"], 0.9)
        config.save()

        assert Config(path).get("sorting", "auto_accept_confidence") == 0.9

    def test_save_creates_the_containing_directory(self, temp_dir):
        path = temp_dir / "a" / "b" / "config.yaml"
        Config(str(path)).save()
        assert path.is_file()

    def test_create_default_config_writes_the_file(self, temp_dir):
        path = temp_dir / "config.yaml"
        config = Config.create_default_config(str(path))
        assert path.is_file()
        assert config.get("database", "path")


class TestSection:
    def test_returns_a_copy_that_cannot_mutate_the_config(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))

        section = config.section("sorting")
        section["neighbors"] = 999

        assert config.get("sorting", "neighbors") != 999

    def test_an_unknown_section_is_empty(self, temp_dir):
        assert Config(str(temp_dir / "config.yaml")).section("nope") == {}


class TestPaths:
    def test_database_path_expands_a_tilde(self, temp_dir, monkeypatch):
        monkeypatch.delenv("MUSIC_CLUSTER_DB", raising=False)
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["database", "path"], "~/somewhere/library.db")

        assert "~" not in config.get_db_path()
        assert config.get_db_path().endswith("somewhere/library.db")

    def test_the_environment_overrides_the_configured_database(self, temp_dir, monkeypatch):
        monkeypatch.setenv("MUSIC_CLUSTER_DB", str(temp_dir / "override.db"))
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["database", "path"], "~/ignored.db")

        assert config.get_db_path() == str(temp_dir / "override.db")

    def test_the_cache_directory_sits_beside_the_database_by_default(self, temp_dir, monkeypatch):
        monkeypatch.delenv("MUSIC_CLUSTER_CACHE", raising=False)
        monkeypatch.setenv("MUSIC_CLUSTER_DB", str(temp_dir / "library.db"))
        config = Config(str(temp_dir / "config.yaml"))

        assert config.get_cache_dir() == os.path.join(str(temp_dir), "cache")

    def test_the_environment_overrides_the_cache_directory(self, temp_dir, monkeypatch):
        monkeypatch.setenv("MUSIC_CLUSTER_CACHE", str(temp_dir / "elsewhere"))
        config = Config(str(temp_dir / "config.yaml"))

        assert config.get_cache_dir() == str(temp_dir / "elsewhere")

    def test_the_config_path_itself_can_come_from_the_environment(self, temp_dir, monkeypatch):
        path = temp_dir / "from-env.yaml"
        path.write_text(yaml.dump({"sorting": {"neighbors": 3}}))
        monkeypatch.setenv("MUSIC_CLUSTER_CONFIG", str(path))

        assert Config().get("sorting", "neighbors") == 3


class TestExtractorConfig:
    def test_maps_settings_onto_the_extractor_keyword_arguments(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["feature_extraction", "mfcc_coefficients"], 13)
        config.set(["feature_extraction", "excerpt_seconds"], 30)

        kwargs = config.extractor_config()

        assert kwargs["n_mfcc"] == 13
        assert kwargs["excerpt_seconds"] == 30
        assert set(kwargs) == {
            "sample_rate",
            "frame_size",
            "hop_size",
            "n_mfcc",
            "excerpt_seconds",
            "max_seconds",
            "profile",
        }

    def test_the_sample_profile_analyses_one_shots_differently(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))

        music = config.extractor_config()
        sample = config.extractor_config("sample")

        assert sample["profile"] == "sample"
        # A one-shot's attack is at the very start and is over in milliseconds:
        # never excerpt, and step through time far more finely.
        assert sample["excerpt_seconds"] == 0
        assert sample["hop_size"] < music["hop_size"]
        assert sample["max_seconds"] is not None

    def test_a_user_override_beats_the_profile_default(self, temp_dir):
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["feature_extraction", "profiles", "sample", "hop_size"], 512)

        assert config.extractor_config("sample")["hop_size"] == 512

    @pytest.mark.parametrize("profile", [None, "music", "sample"])
    def test_the_result_can_be_passed_straight_to_the_extractor(self, temp_dir, profile):
        from music_cluster.extractor import FeatureExtractor

        config = Config(str(temp_dir / "config.yaml"))
        FeatureExtractor(**config.extractor_config(profile))
