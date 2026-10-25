# Upstream contract

CQTNet is MIT licensed, copyright (c) 2020 zhesong yu. The original
[LICENSE](../LICENSE) is retained.

- Upstream: <https://github.com/yzspku/CQTNet>
- Pinned revision: `c6411f41258b9cd5e8ce8f000d7f494391b4f6e2`
- Historical recovery **and actual implementation starting revision**:
  `4a636cf0944e7c6b60f838845b8073fafb06ee59`
- New branch: `refactor/cqtnet-toolkit`
- Authorization: user explicitly authorized the current Linux remote host.

The historical and pinned upstream model source have identical SHA-256:
`7b69cc83cb201089d3cafa3922de52f6f24f91a35dcc917b2d0eea96ba1fbf8f`.
The checksum is enforced on the independent, verbatim
[test reference](../tests/reference/CQTNet.py). License SHA-256:
`64344c188606a7768e71bc10afda8a454cc88ea7241ced800f60f16278ae0c9f`.

## Preserved and adapted behavior

[CQTNet](../src/csi/models/CQTNet.py) preserves the ordered convolution,
BatchNorm, activation, pooling and linear modules, state keys and forward math:
adaptive max pool -> 512-to-300 `fc0` embedding -> 10,000-class `fc1`.
Tests compare logits and embeddings exactly at 200, 300, 400 and 520 frames.
Unused imports and legacy disk-writing methods are excluded from the runtime
base class; checkpoint handling is centralized. This is architecture/numerical
compatibility, not a claim that the historical Python/CUDA environment survived.

[ExperimentModel](../src/csi/models/adapters.py) explicitly owns classifier
replacement, Conv0 freezing, BatchNorm mode, and reconstructed dropout after
`fc0`. With dropout disabled, or in evaluation mode, it delegates the original
forward unchanged. Dropout placement is **not recovered historical fact**.

| Behavior | `upstream.ce` | Report presets |
|---|---|---|
| CE inputs | Independent shuffled loaders at 200/300/400; three sequential optimizer steps per zipped batch | Configured fixed-length loader |
| Augmentation | Two frequency masks; quantized PIL bilinear speed resize 0.7-1.3; exclusive-upper-bound random crop | Off unless explicitly selected |
| Evaluation | Variable length, minimum 200 | Normalize full array, front crop/pad to 400 by default |
| Class IDs | Original numeric IDs 0-9999, 10,000-class head | Stable sorted training clique map |
| Adam | LR 0.001, decay 1e-5 | Stage-specific LR, decay 1e-4 |
| Scheduler | Plateau on sum(batch mean CE)/example count; factor 0.8, patience 2, minimum 5e-6 | Cosine, horizon equal to budget, minimum 1e-6 |
| Dropout/tonic/balanced | Not implicitly enabled | Selected explicitly |

PIL replaces the unused torchvision package dependency while matching the
upstream PIL resize operation. Augmentation and the three-loader path have CPU
tests. The old upstream evaluation benchmark-selection loop is replaced by
explicit manifests through the shared evaluator, not another legacy trainer.
Record which evaluation datasets were actually selected; do not call a
custom subset the full upstream benchmark.

CQT uses explicit modern librosa defaults, magnitude (not log), and
non-overlapping 20-frame means dropping remainder frames. Historical
library-dependent audio extraction parity still requires recovered reference
audio/features; the modern extraction recipe is versioned rather than called
historically identical.

## Preservation references

All links below are fixed to the source commit, not moving branch files:

- [Submitted paper](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/blob/4a636cf0944e7c6b60f838845b8073fafb06ee59/ISMIR_2025_cover_song%20%281%29.pdf)
- [Thesis/report source](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/blob/4a636cf0944e7c6b60f838845b8073fafb06ee59/final_report.tex)
- [Prerequisite results and their conflict](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/blob/4a636cf0944e7c6b60f838845b8073fafb06ee59/EXPERIMENT_AND_RESULTS_REPORT.md)
- [Earlier ablation matrix](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/blob/4a636cf0944e7c6b60f838845b8073fafb06ee59/novel_indian_pipeline/results/ablation_matrix.md)

The user-supplied 2026-10-25 rebuild plan overrides contradictory historical
prose. It is not a recovered final implementation. The user subsequently
authorized cleanup without waiting for a GPU. After passing the CPU baseline,
250 unchanged tracked legacy files were removed from this branch (637.9 MiB),
including duplicate trainers/loaders, reports, figures, logs and binary weights.
The original branch and Git history were not changed; no archive was created.
The user's untracked rebuild plan was retained.

The complete discarded tree remains available at the
[historical source commit](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/tree/4a636cf0944e7c6b60f838845b8073fafb06ee59).
For a specifically selected historical file, `git show
4a636cf0944e7c6b60f838845b8073fafb06ee59:<relative-path>` reads its original
content without restoring the legacy tree. Recovered weights belong under
the ignored data checkpoint directories, not in the source tree.

Cleanup and CPU smoke success do not establish GPU compatibility, original
benchmark completeness or historical report reproduction; those checks remain
outstanding.
