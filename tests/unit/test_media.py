"""Waveform envelopes: what actually gets drawn under the playhead.

Every case here is one that used to draw wrongly — a flat block for a track with
an arrangement, a waveform that stopped seven minutes into a nine minute set, a
sliver of audio against an empty tail.
"""

import numpy as np
import pytest

from music_cluster import media
from music_cluster.media import compute_waveform, media_type_for, parse_range_header

pytest.importorskip("soundfile")
import soundfile as sf  # noqa: E402


SR = 22050


def write_audio(path, samples, rate=SR, subtype=None):
    sf.write(str(path), np.asarray(samples, dtype=np.float32), rate, subtype=subtype)
    return str(path)


def tone(seconds, freq=220.0, gain=1.0, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * freq * t) * gain


@pytest.fixture(autouse=True)
def clear_cache():
    media.clear_waveform_memory_cache()
    yield
    media.clear_waveform_memory_cache()


class TestComputeWaveform:
    def test_returns_the_requested_number_of_points(self, temp_dir):
        path = write_audio(temp_dir / "tone.wav", tone(5))
        result = compute_waveform(path, samples=137)
        assert len(result["peaks"]) == 137
        assert result["samples"] == 137

    def test_peaks_are_normalised(self, temp_dir):
        path = write_audio(temp_dir / "tone.wav", tone(5, gain=0.2))
        peaks = compute_waveform(path, samples=64)["peaks"]
        assert min(peaks) >= 0.0
        assert max(peaks) <= 1.0
        # A quiet recording still uses the full height: the envelope is relative
        # to the track, not to full scale.
        assert max(peaks) > 0.9

    def test_covers_the_whole_track(self, temp_dir):
        """A long set must not be cut off part way, or the playhead lies."""
        quiet = tone(400, gain=0.1)
        loud = tone(200, freq=180, gain=1.0)
        path = write_audio(temp_dir / "long.wav", np.concatenate([quiet, loud]))

        result = compute_waveform(path, samples=120)

        assert result["duration"] == pytest.approx(600, abs=1)
        # The loud third of the track is at the end of the envelope, not missing.
        peaks = result["peaks"]
        assert np.mean(peaks[-30:]) > np.mean(peaks[:60]) * 2

    def test_quiet_section_is_visibly_shorter_than_a_loud_one(self, temp_dir):
        loud = tone(10, gain=1.0)
        quiet = tone(10, gain=0.25)
        path = write_audio(temp_dir / "sections.wav", np.concatenate([loud, quiet]))

        peaks = compute_waveform(path, samples=100)["peaks"]

        assert np.mean(peaks[:40]) > 0.8
        assert np.mean(peaks[60:]) < 0.5

    def test_limited_master_still_shows_structure(self, temp_dir):
        """The 'every bar is the same height' case.

        Peaks alone cannot separate these sections — both hit the ceiling — so
        an envelope built only from them draws one solid block.
        """
        rng = np.random.default_rng(3)
        hot = np.tanh(rng.normal(0, 1, SR * 10) * 6)
        backed_off = np.tanh(rng.normal(0, 1, SR * 10) * 6) * 0.5
        path = write_audio(temp_dir / "limited.wav", np.concatenate([hot, backed_off]))

        peaks = np.asarray(compute_waveform(path, samples=100)["peaks"])

        assert peaks[:40].mean() - peaks[60:].mean() > 0.3
        assert peaks.std() > 0.15

    def test_steady_material_is_not_stretched_into_noise(self, temp_dir):
        """Contrast is added where there is something to show, not invented."""
        path = write_audio(temp_dir / "steady.wav", tone(20, gain=0.7))

        peaks = np.asarray(compute_waveform(path, samples=100)["peaks"])

        assert peaks.std() < 0.05
        assert peaks.mean() > 0.9

    def test_a_one_shot_keeps_its_decay(self, temp_dir):
        """A kick's tail must fall to the floor, not be stretched back up.

        The percentile expansion that pulls structure out of a limited master
        would lift a 300 ms decay off the baseline and draw a sound that keeps
        ringing when it does not.
        """
        t = np.arange(int(0.3 * SR)) / SR
        decay = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 14)
        path = write_audio(temp_dir / "kick.wav", decay)

        peaks = compute_waveform(path, samples=64)["peaks"]
        assert max(peaks) == pytest.approx(1.0, abs=0.05)
        # By the end of a decay this steep there is essentially nothing left.
        assert peaks[-1] < 0.15
        assert peaks[0] > peaks[len(peaks) // 2] > peaks[-1]

    def test_a_long_track_still_gets_its_structure_expanded(self, temp_dir):
        """The one-shot rule must not disable expansion for arrangements."""
        quiet = tone(4, gain=0.35)
        loud = tone(4, gain=0.4)
        path = write_audio(temp_dir / "long.wav", np.concatenate([quiet, loud]))

        peaks = compute_waveform(path, samples=64)["peaks"]
        first, second = peaks[:28], peaks[36:]
        # A 1.1x difference in the audio has to be visible as more than that.
        assert np.mean(second) - np.mean(first) > 0.15

    def test_silence_draws_nothing_rather_than_dividing_by_zero(self, temp_dir):
        path = write_audio(temp_dir / "silent.wav", np.zeros(SR * 3))
        result = compute_waveform(path, samples=50)
        assert result["peaks"] == [0.0] * 50
        assert result["duration"] == pytest.approx(3, abs=0.1)

    def test_short_file_spreads_across_the_whole_width(self, temp_dir):
        """A sub-second file used to be zero-padded into a mostly empty bar."""
        decay = tone(0.3) * np.linspace(1, 0, int(0.3 * SR))
        path = write_audio(temp_dir / "blip.wav", decay)

        peaks = compute_waveform(path, samples=200)["peaks"]

        assert len(peaks) == 200
        assert peaks[0] > 0.5
        # The decay reaches the far end of the drawing instead of stopping short.
        assert max(peaks[150:]) < peaks[0]
        assert any(value > 0 for value in peaks[100:150])

    def test_stereo_is_collapsed_to_one_envelope(self, temp_dir):
        left = tone(4, freq=200)
        right = tone(4, freq=300)
        path = write_audio(temp_dir / "stereo.wav", np.stack([left, right], axis=1))

        result = compute_waveform(path, samples=64)

        assert len(result["peaks"]) == 64
        assert max(result["peaks"]) > 0.9

    def test_resolution_is_clamped(self, temp_dir):
        path = write_audio(temp_dir / "tone.wav", tone(3))
        assert len(compute_waveform(path, samples=1)["peaks"]) == media.MIN_WAVEFORM_SAMPLES
        assert len(compute_waveform(path, samples=99999)["peaks"]) == media.MAX_WAVEFORM_SAMPLES

    def test_unreadable_file_raises_value_error(self, temp_dir):
        path = temp_dir / "not-audio.wav"
        path.write_bytes(b"this is not audio")
        with pytest.raises(ValueError):
            compute_waveform(str(path), samples=32)

    def test_flac_reads_the_same_shape_as_wav(self, temp_dir):
        audio = np.concatenate([tone(4, gain=1.0), tone(4, gain=0.2)])
        wav = write_audio(temp_dir / "a.wav", audio)
        flac = write_audio(temp_dir / "a.flac", audio)

        from_wav = compute_waveform(wav, samples=64)["peaks"]
        from_flac = compute_waveform(flac, samples=64)["peaks"]

        assert np.allclose(from_wav, from_flac, atol=0.02)


class TestDecodedFallback:
    """The path used for formats libsndfile will not open (m4a, opus, wma)."""

    def test_matches_the_streaming_path(self, temp_dir):
        pytest.importorskip("librosa")
        audio = np.concatenate([tone(5, gain=1.0), tone(5, gain=0.3)])
        path = write_audio(temp_dir / "a.wav", audio)

        streamed = media._windows_streaming(path, 50)
        decoded = media._windows_decoded(path, 50)

        assert streamed is not None
        assert decoded[2] == pytest.approx(streamed[2], abs=0.1)
        # Same shape, allowing for the resampling the decoded path does.
        assert np.corrcoef(streamed[0], decoded[0])[0, 1] > 0.95

    def test_used_when_streaming_cannot_open_the_file(self, temp_dir, monkeypatch):
        pytest.importorskip("librosa")
        path = write_audio(temp_dir / "a.wav", tone(3))
        monkeypatch.setattr(media, "_windows_streaming", lambda *args, **kwargs: None)

        result = compute_waveform(path, samples=32)

        assert len(result["peaks"]) == 32
        assert result["duration"] == pytest.approx(3, abs=0.1)


class TestWaveformCache:
    def test_second_call_is_served_from_cache(self, temp_dir, monkeypatch):
        path = write_audio(temp_dir / "a.wav", tone(4))
        cache_dir = str(temp_dir / "cache")

        first, key = media.cached_waveform(path, samples=64, cache_dir=cache_dir)

        calls = []
        monkeypatch.setattr(
            media, "compute_waveform", lambda *a, **k: calls.append(a) or {"peaks": [1.0]}
        )
        second, second_key = media.cached_waveform(path, samples=64, cache_dir=cache_dir)

        assert calls == []
        assert second == first
        assert second_key == key

    def test_disk_cache_survives_a_restart(self, temp_dir):
        path = write_audio(temp_dir / "a.wav", tone(4))
        cache_dir = str(temp_dir / "cache")

        first, _ = media.cached_waveform(path, samples=64, cache_dir=cache_dir)
        media.clear_waveform_memory_cache()  # as if the process restarted
        second, _ = media.cached_waveform(path, samples=64, cache_dir=cache_dir)

        assert second == first

    def test_editing_the_file_invalidates_the_cache(self, temp_dir):
        path = write_audio(temp_dir / "a.wav", tone(4, gain=1.0))
        cache_dir = str(temp_dir / "cache")
        before, before_key = media.cached_waveform(path, samples=64, cache_dir=cache_dir)

        write_audio(temp_dir / "a.wav", np.concatenate([tone(2, gain=1.0), np.zeros(SR * 2)]))
        after, after_key = media.cached_waveform(path, samples=64, cache_dir=cache_dir)

        assert after_key != before_key
        assert after["peaks"] != before["peaks"]

    def test_resolution_is_part_of_the_key(self, temp_dir):
        path = write_audio(temp_dir / "a.wav", tone(4))
        cache_dir = str(temp_dir / "cache")
        assert len(media.cached_waveform(path, 64, cache_dir)[0]["peaks"]) == 64
        assert len(media.cached_waveform(path, 128, cache_dir)[0]["peaks"]) == 128

    def test_corrupt_cache_entry_is_ignored(self, temp_dir):
        path = write_audio(temp_dir / "a.wav", tone(4))
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        key = media.waveform_cache_key(path, 64)
        (cache_dir / f"{key}.json").write_text("{ not json")

        result, _ = media.cached_waveform(path, samples=64, cache_dir=str(cache_dir))

        assert len(result["peaks"]) == 64


class TestRangeHeaders:
    """Seeking depends on these, and a bad range breaks playback outright."""

    @pytest.mark.parametrize(
        "header,size,expected",
        [
            ("bytes=0-99", 1000, (0, 99)),
            ("bytes=500-", 1000, (500, 999)),
            ("bytes=-100", 1000, (900, 999)),
            ("bytes=0-5000", 1000, (0, 999)),
            ("bytes=1000-", 1000, None),
            ("bytes=abc-def", 1000, None),
            ("items=0-99", 1000, None),
            (None, 1000, None),
        ],
    )
    def test_parses(self, header, size, expected):
        assert parse_range_header(header, size) == expected


def test_media_types_cover_the_formats_the_library_accepts():
    assert media_type_for("/music/a.mp3") == "audio/mpeg"
    assert media_type_for("/music/a.FLAC") == "audio/flac"
    assert media_type_for("/music/a.unknown") == "application/octet-stream"
