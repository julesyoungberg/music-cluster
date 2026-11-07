# music-cluster evaluation

2026-11-07T05:33:45+00:00 · version 2.0.0 · commit 290ee44

## key-giantsteps

| Metric | Value |
| --- | --- |
| key_weighted | 0.513 |
| key_correct | 0.404 |
| key_tagged_correct | 0.485 |
| key_share_tagged | 0.508 |

| Method | weighted | correct |
| --- | --- | --- |
| ours_all | 0.513 | 0.404 |
| ours_tagged_only | 0.595 | 0.485 |
| most_common_key | 0.150 | 0.121 |
| essentia_edma | 0.658 | 0.591 |

## tempo-giantsteps

| Metric | Value |
| --- | --- |
| tempo_accuracy1 | 0.637 |
| tempo_accuracy2 | 0.829 |
| librosa_accuracy1 | 0.584 |

| Method | accuracy1 | accuracy2 |
| --- | --- | --- |
| ours | 0.637 | 0.829 |
| librosa_beat_track | 0.584 | 0.741 |
| essentia_rhythm2013 | 0.608 | 0.818 |

## sorting-fma

| Metric | Value |
| --- | --- |
| sorter_top1 | 0.449 |
| sorter_top3 | 0.809 |
| sorter_macro_f1 | 0.431 |
| sorter_ece | 0.027 |
| logistic_top1 | 0.426 |
| auto_precision_current | 0.554 |
| auto_coverage_current | 0.164 |
| coverage_at_95_precision | 0.000 |
| novelty_auroc | 0.501 |
| nowhere_auto_filed | 0.176 |

| Method | top1 | top3 | macro_f1 | ece | aurc |
| --- | --- | --- | --- | --- | --- |
| sorter | 0.449 | 0.809 | 0.431 | 0.027 | 0.429 |
| majority | 0.186 | 0.558 | 0.180 | - | - |
| tempo_only | 0.224 | 0.600 | 0.212 | - | - |
| logistic | 0.426 | 0.783 | 0.418 | 0.318 | 0.461 |
| knn | 0.386 | 0.744 | 0.371 | 0.272 | 0.499 |

Auto-file operating points (novel tracks count as errors):

- current (confidence ≥ 0.6, margin ≥ 0.08): precision 0.554, coverage 0.164
- precision 90: not reachable
- precision 95: not reachable

