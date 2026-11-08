"""Automated evaluation: score each feature against labelled audio and baselines.

Layers, cheapest first:

* synthetic audio whose beat type, tempo and key are right by construction
  (:mod:`.tag_suites`, :mod:`.drums`);
* public labelled datasets per feature (:mod:`.datasets`, :mod:`.download`),
  analysed once and cached (:mod:`.analysis`);
* a simulated library for the sorter, with baselines and auto-file thresholds
  (:mod:`.sorting_sim`).

``cratewise eval`` drives it all; :mod:`.report` writes the results and
diffs them against a committed baseline. Nothing here reads or writes the
user's library database, and nothing moves a file.
"""
