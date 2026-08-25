"""Optional LLM naming: opt-in, and never load-bearing.

Every failure path must return an empty mapping so the caller falls back to the
heuristic labels rather than losing the discovery run.
"""

import sys
import types

import pytest

from music_cluster import llm
from music_cluster.config import Config


@pytest.fixture
def enabled_config(temp_dir, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = Config(str(temp_dir / "config.yaml"))
    config.set(["labeling", "llm_enabled"], True)
    return config


CANDIDATES = [
    {
        "candidate_index": 0,
        "size": 42,
        "stats": {
            "tag_bpm_median": 124,
            "descriptors": ["Deep", "Dark"],
            "top_genres": [{"name": "House", "count": 30}],
        },
        "exemplar_tracks": [{"artist": "Someone", "title": "A track"}],
    },
    {
        "candidate_index": 1,
        "size": 18,
        "stats": {"detected_bpm": 172, "descriptors": ["Percussive"]},
        "exemplar_tracks": [{"artist": None, "filename": "b.mp3"}],
    },
]


class FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text=self._text)
        return types.SimpleNamespace(content=[block])


def install_fake_anthropic(monkeypatch, text=None, error=None):
    """Install a stand-in `anthropic` module and return its messages stub."""
    messages = FakeMessages(text or "")

    class FakeAnthropic:
        def __init__(self, **kwargs):
            if error is not None:
                raise error
            self.messages = messages

    module = types.ModuleType("anthropic")
    module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return messages


class TestIsAvailable:
    def test_off_by_default(self, temp_dir, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert llm.is_available(Config(str(temp_dir / "config.yaml"))) is False

    def test_needs_a_key_as_well_as_the_flag(self, temp_dir, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["labeling", "llm_enabled"], True)

        assert llm.is_available(config) is False

    def test_available_when_enabled_with_a_key(self, enabled_config):
        assert llm.is_available(enabled_config) is True

    def test_honours_a_custom_key_variable(self, temp_dir, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("MY_KEY", "abc")
        config = Config(str(temp_dir / "config.yaml"))
        config.set(["labeling", "llm_enabled"], True)
        config.set(["labeling", "llm_api_key_env"], "MY_KEY")

        assert llm.is_available(config) is True


class TestSuggestNames:
    def test_returns_names_keyed_by_candidate_index(self, enabled_config, monkeypatch):
        install_fake_anthropic(
            monkeypatch,
            '{"names": [{"candidate_index": 0, "name": "Deep Rolling House"}, '
            '{"candidate_index": 1, "name": "Rollers"}]}',
        )

        assert llm.suggest_names(CANDIDATES, enabled_config) == {
            0: "Deep Rolling House",
            1: "Rollers",
        }

    def test_sends_only_metadata_and_statistics(self, enabled_config, monkeypatch):
        messages = install_fake_anthropic(monkeypatch, '{"names": []}')

        llm.suggest_names(CANDIDATES, enabled_config)

        prompt = messages.calls[0]["messages"][0]["content"]
        # The privacy promise in the README: tags and measurements, never audio.
        assert "Someone - A track" in prompt
        assert "House (30)" in prompt
        assert "124" in prompt

    def test_uses_the_configured_model(self, enabled_config, monkeypatch):
        messages = install_fake_anthropic(monkeypatch, '{"names": []}')
        enabled_config.set(["labeling", "llm_model"], "claude-opus-4-20250514")

        llm.suggest_names(CANDIDATES, enabled_config)

        assert messages.calls[0]["model"] == "claude-opus-4-20250514"

    def test_does_nothing_when_disabled(self, temp_dir, monkeypatch):
        called = install_fake_anthropic(monkeypatch, '{"names": [{"candidate_index": 0}]}')

        assert llm.suggest_names(CANDIDATES, Config(str(temp_dir / "config.yaml"))) == {}
        assert called.calls == []

    def test_does_nothing_without_candidates(self, enabled_config, monkeypatch):
        called = install_fake_anthropic(monkeypatch, '{"names": []}')
        assert llm.suggest_names([], enabled_config) == {}
        assert called.calls == []

    def test_a_missing_anthropic_package_is_not_an_error(self, enabled_config, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        assert llm.suggest_names(CANDIDATES, enabled_config) == {}

    def test_an_api_failure_falls_back_silently(self, enabled_config, monkeypatch):
        install_fake_anthropic(monkeypatch, error=RuntimeError("rate limited"))
        assert llm.suggest_names(CANDIDATES, enabled_config) == {}

    def test_an_unparseable_reply_falls_back(self, enabled_config, monkeypatch):
        install_fake_anthropic(monkeypatch, "Sorry, I cannot help with that.")
        assert llm.suggest_names(CANDIDATES, enabled_config) == {}


class TestParse:
    def test_reads_json_wrapped_in_prose(self):
        text = 'Here you go:\n{"names": [{"candidate_index": 2, "name": "Peak Time"}]}\nHope that helps.'
        assert llm._parse(text) == {2: "Peak Time"}

    def test_skips_entries_that_are_missing_fields(self):
        text = '{"names": [{"name": "No index"}, {"candidate_index": 1, "name": "Fine"}]}'
        assert llm._parse(text) == {1: "Fine"}

    def test_skips_empty_names(self):
        text = '{"names": [{"candidate_index": 0, "name": "   "}]}'
        assert llm._parse(text) == {}

    def test_truncates_a_runaway_name(self):
        text = '{"names": [{"candidate_index": 0, "name": "%s"}]}' % ("x" * 300)
        assert len(llm._parse(text)[0]) == 80

    def test_returns_nothing_for_text_with_no_json(self):
        assert llm._parse("no braces here") == {}
        assert llm._parse("") == {}

    def test_returns_nothing_for_malformed_json(self):
        assert llm._parse('{"names": [') == {}


class TestRender:
    def test_labels_each_candidate_with_its_index_and_size(self):
        rendered = llm._render(CANDIDATES)
        assert "## Candidate 0 (42 tracks)" in rendered
        assert "## Candidate 1 (18 tracks)" in rendered

    def test_falls_back_to_the_filename_for_an_untagged_track(self):
        assert "Unknown - b.mp3" in llm._render(CANDIDATES)

    def test_omits_sections_a_candidate_has_no_data_for(self):
        rendered = llm._render([{"candidate_index": 0, "size": 3}])
        assert "## Candidate 0 (3 tracks)" in rendered
        assert "genre tags" not in rendered
