"""The evaluation harness: metrics on hand-made numbers, loaders on fake datasets."""

import json

import numpy as np
import pytest

from music_cluster.eval import datasets, metrics, report, sorting_sim
from tests.conftest import FEATURE_DIM


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_classification_metrics():
    truth = ["a", "a", "a", "b"]
    predicted = ["a", "a", "b", "b"]
    assert metrics.accuracy(truth, predicted) == pytest.approx(0.75)
    # a: P=1 R=2/3 F1=0.8; b: P=0.5 R=1 F1=2/3
    assert metrics.macro_f1(truth, predicted) == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics.balanced_accuracy(truth, predicted) == pytest.approx((2 / 3 + 1) / 2)
    assert metrics.top_k_accuracy(truth, [["b", "a"]] * 4, 1) == pytest.approx(0.25)
    assert metrics.top_k_accuracy(truth, [["b", "a"]] * 4, 2) == pytest.approx(1.0)
    assert metrics.confusion_pairs(truth, predicted) == [("a", "b", 1)]


def test_perfectly_calibrated_scores_have_no_calibration_error():
    # Ten items at 0.8 confidence, eight right.
    confidence = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert metrics.expected_calibration_error(confidence, correct) == pytest.approx(0.0)
    assert metrics.brier(confidence, correct) == pytest.approx(0.8 * 0.04 + 0.2 * 0.64)


def test_overconfident_scores_are_penalised():
    assert metrics.expected_calibration_error([0.95] * 4, [True, False, True, False]) == (
        pytest.approx(0.45)
    )


def test_risk_coverage_ranks_by_confidence():
    result = metrics.risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False], points=4)
    precisions = [row["precision"] for row in result["curve"]]
    assert precisions == pytest.approx([1.0, 1.0, 2 / 3, 0.5])


def test_auroc():
    assert metrics.auroc([3, 4], [1, 2]) == pytest.approx(1.0)
    assert metrics.auroc([1], [1]) == pytest.approx(0.5)
    assert metrics.auroc([], [1]) is None


def test_tempo_accuracy_forgives_octaves_only_in_accuracy2():
    result = metrics.tempo_accuracy([170, 128, 120], [172, 64, 150])
    assert result["accuracy1"] == pytest.approx(1 / 3)
    assert result["accuracy2"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Eb minor", (3, "minor")),
        ("D# min", (3, "minor")),
        ("f#m", (6, "minor")),
        ("C major", (0, "major")),
        ("B", (11, "major")),
        ("-", None),
        ("", None),
    ],
)
def test_parse_key(text, expected):
    assert metrics.parse_key(text) == expected


@pytest.mark.parametrize(
    "reference, estimate, score",
    [
        ("A minor", "A minor", 1.0),
        ("A minor", "E minor", 0.5),  # a fifth above
        ("A minor", "C major", 0.3),  # relative
        ("C major", "A minor", 0.3),
        ("A minor", "A major", 0.2),  # parallel
        ("A minor", "D minor", 0.0),  # a fifth below is not credited
        ("A minor", None, 0.0),
    ],
)
def test_key_score(reference, estimate, score):
    ref = metrics.parse_key(reference)
    assert ref is not None
    assert metrics.key_score(ref, metrics.parse_key(estimate)) == pytest.approx(score)


def test_key_accuracy_skips_unannotated_tracks():
    result = metrics.key_accuracy(["A minor", "-", "C major"], ["C major", "C major", "C major"])
    assert result["count"] == 2
    assert result["weighted"] == pytest.approx((0.3 + 1.0) / 2)
    assert result["relative"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Sorting simulation
# ----------------------------------------------------------------------


def _library(rng, groups=8, artists=12, per_artist=6, spread=1.0):
    x, y, who = [], [], []
    for g in range(groups):
        centre = rng.normal(0, 1, FEATURE_DIM) * 1.5
        for a in range(artists):
            artist_centre = centre + rng.normal(0, 0.5, FEATURE_DIM)
            for _ in range(per_artist):
                x.append(artist_centre + rng.normal(0, spread, FEATURE_DIM))
                y.append(f"genre{g}")
                who.append(f"artist{g}-{a}")
    return np.array(x), y, who


def test_sorter_beats_trivial_baselines_on_separable_groups(rng):
    x, y, who = _library(rng)
    result = sorting_sim.simulate(x, y, who, sorting_sim.SimulationSettings(trials=6, max_refs=60))
    methods = result["methods"]
    assert methods["sorter"]["top1"] > 0.9
    assert methods["sorter"]["top1"] > methods["majority"]["top1"] + 0.3
    assert {"majority", "tempo_only", "logistic", "knn"} <= set(methods)
    headline = result["headline"]
    assert headline["novelty_auroc"] is not None and headline["novelty_auroc"] > 0.8
    assert 0.0 <= headline["auto_coverage_current"] <= 1.0
    assert result["trials"] == 6


def test_simulation_keeps_each_artist_on_one_side(rng):
    indices = np.arange(40)
    artists = [f"a{i // 4}" for i in range(40)]
    refs, queries = sorting_sim._split_by_artist(indices, artists, 0.3, rng)
    assert {artists[i] for i in refs}.isdisjoint({artists[i] for i in queries})
    assert len(refs) + len(queries) == 40


def test_unlearnable_groups_score_near_chance(rng):
    x = rng.normal(0, 1, (480, FEATURE_DIM))
    y = [f"g{i % 6}" for i in range(480)]
    result = sorting_sim.simulate(x, y, None, sorting_sim.SimulationSettings(trials=4))
    assert result["methods"]["sorter"]["top1"] < 0.5


def test_simulation_needs_three_labels(rng):
    x = rng.normal(0, 1, (40, FEATURE_DIM))
    with pytest.raises(ValueError):
        sorting_sim.simulate(x, ["a", "b"] * 20, None, sorting_sim.SimulationSettings(trials=1))


# ----------------------------------------------------------------------
# Dataset loaders, on fake copies of each layout
# ----------------------------------------------------------------------


def test_fma_loader(temp_dir):
    meta = temp_dir / "fma" / "fma_metadata"
    meta.mkdir(parents=True)
    (meta / "tracks.csv").write_text(
        ",artist,set,set,track,track\n"
        ",id,split,subset,genre_top,genres\n"
        "track_id,,,,,\n"
        "2,10,training,small,Hip-Hop,[21]\n"
        '3,11,training,medium,Electronic,"[15, 181]"\n'
        "5,12,test,small,Electronic,[181]\n"
        "6,12,test,small,Electronic,[181]\n"
    )
    (meta / "genres.csv").write_text(
        "genre_id,#tracks,parent,title,top_level\n"
        "15,1,0,Electronic,15\n"
        "21,1,0,Hip-Hop,21\n"
        "181,1,15,Techno,15\n"
    )
    for tid in ("000002", "000005"):
        path = temp_dir / "fma" / "fma_small" / "000" / f"{tid}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    top = datasets.fma(str(temp_dir), subset="small", level="top")
    assert [(i.label, i.artist) for i in top] == [("Hip-Hop", "10"), ("Electronic", "12")]
    sub = datasets.fma(str(temp_dir), subset="small", level="sub")
    # Track 2 is top-level only; track 6 has no audio on disk.
    assert [i.label for i in sub] == ["Techno"]


def test_giantsteps_loader_prefers_corrected_annotations(temp_dir):
    repo = temp_dir / "giantsteps-tempo-dataset"
    for folder in ("annotations/tempo", "annotations_v2/tempo", "annotations/genre", "audio"):
        (repo / folder).mkdir(parents=True)
    (repo / "annotations/tempo/1.LOFI.bpm").write_text("85")
    (repo / "annotations_v2/tempo/1.LOFI.bpm").write_text("170\n")
    (repo / "annotations/genre/1.LOFI.genre").write_text("drum-and-bass")
    (repo / "audio/1.LOFI.mp3").write_bytes(b"")
    (repo / "annotations_v2/tempo/2.LOFI.bpm").write_text("128")  # no audio: skipped

    items = datasets.giantsteps_tempo(str(temp_dir))
    assert len(items) == 1
    assert items[0].label == "170"
    assert items[0].meta["genre"] == "drum-and-bass"


def test_folder_tree_labels_by_subfolder(temp_dir):
    for folder, names in (("Deep House", ["a.mp3", "b.wav"]), ("Techno", ["c.flac", "notes.txt"])):
        (temp_dir / folder / "sub").mkdir(parents=True)
        for name in names:
            (temp_dir / folder / "sub" / name).write_bytes(b"")
    items = datasets.folder_tree(str(temp_dir), read_artists=False)
    assert sorted((i.label, i.path.rsplit("/", 1)[-1]) for i in items) == [
        ("Deep House", "a.mp3"),
        ("Deep House", "b.wav"),
        ("Techno", "c.flac"),
    ]


def test_missing_dataset_raises_file_not_found(temp_dir):
    with pytest.raises(FileNotFoundError):
        datasets.giantsteps_key(str(temp_dir))


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------


def test_compare_flags_drops_in_the_wrong_direction(temp_dir):
    before = {"suites": {"s": {"headline": {"acc": 0.80, "ece": 0.10}}}}
    after = report.build(
        {"s": {"headline": {"acc": 0.70, "ece": 0.20}, "lower_is_better": ["ece"]}}
    )
    rows = {row["metric"]: row for row in report.compare(after, before, tolerance=0.02)}
    assert rows["acc"]["regressed"]
    assert rows["ece"]["regressed"]

    better = report.build(
        {"s": {"headline": {"acc": 0.90, "ece": 0.05}, "lower_is_better": ["ece"]}}
    )
    assert not any(row["regressed"] for row in report.compare(better, before))

    path = report.save(after, str(temp_dir / "r.json"))
    assert json.loads(path.read_text())["suites"]["s"]["headline"]["acc"] == 0.70
    text = report.markdown(after, report.compare(after, before))
    assert "| s | acc | 0.800 | 0.700 | -0.100 ⚠ |" in text
