from music_cluster import cli


def test_progress_bar_starts_a_new_bar_for_a_second_pass(monkeypatch):
    totals = []

    class FakeBar:
        def __init__(self, total, **_):
            totals.append(total)
            self.n = 0

        def refresh(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(cli, "tqdm", FakeBar)
    callback = cli.progress_bar("Embedding")
    for done, total in [(8, 10), (10, 10), (2, 3), (3, 3)]:
        callback(done, total, "x.mp3")

    assert totals == [10, 3]
