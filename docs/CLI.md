# CLI and configuration reference

Both `csi` and `python -m csi` use the same parser. `--help` is available at
every level. Commands print JSON on success. Configuration/runtime/input errors
print `csi: ...` on stderr and return 2; parser errors also return 2. Successful
execution returns 0. No success is reported after skipping a required test set.

## Command tree

| Command | Parameters and behavior |
|---|---|
| `list` | Presets, aliases, groups, components; no compute imports |
| `schema` | Every configuration field's type, default and allowed choices |
| `show` | Common configuration flags; resolved config, differences, evidence and dependency |
| `doctor` | Common flags, optional `--init` or `--resume`; validate inputs and initialization |
| `train` | Common flags, required `--run-id`, optional `--init` or `--resume`; one shared engine |
| `evaluate` | Common flags, required `--run-id` and `--init`; `--resume` is rejected |
| `suite` | Common flags, optional `--group`; exactly one of `--plan`/`--execute` |
| `data fixture` | Required `--output`; create a tiny synthetic CQT inventory in an empty directory |
| `data fetch` | Required `--asset dataset\|checkpoints`, repeatable exact `--file`, `--output` |
| `data prepare` | Common flags, required `--inventory`, optional `--features`; workers from config |
| `data tonic` | Common flags, required `--output`, optional `--check` |
| `data separate-vocals` | Repeatable required `--input`, required `--output`, `--model htdemucs`, `--device cpu`, opt-in `--allow-model-download` |
| `summarize` | `--runs runs`, optional `--output`; verify sealed run artifacts, aggregate actual results |
| `verify` | Required `--run`; optional `--reevaluate`, `--output-run-id`, `--atol 1e-8` |

Common flags: `--preset NAME_OR_ALIAS`, `--config FILE`, `--root DIRECTORY`,
`--dataset indian|shs100k`, `--modality mix|vocal`, repeatable `--set FIELD=VALUE`.
Suite `--group` choices are `upstream`, `prerequisites`, `phase1`, `phase2`,
`phase3`; `--preset` narrows to one row. Planning lists dependencies only.

Suite execution requires `--init-map FILE` (canonical preset -> checkpoint path
or null for an intentional scratch run) and `--run-prefix`. Relative initializer
paths in this map resolve against the map file. All selected initializers and
data are preflighted before the first run. Dependencies are not trained
automatically and screening/final rows never initialize from one another.
Full suite execution is an expensive explicit user action, not a planning step.

`--init` is a strict weights warm start, with compatible ArcFace transfer only
when class maps are verified equal. `--resume` requires the full toolkit
`last.pt`, identical resolved configuration/code/input manifests and restores
the epoch boundary. `best.pt` is explicitly an evaluation/initializer snapshot,
not an optimizer/RNG resume checkpoint. Paths are never selected by recency.

Verification checks sealed run membership/hashes, provenance presence, original
manifests, feature/audio provenance and initializer checksum. Optional
reevaluation creates a fresh run, comparing MAP/Top10/Rank1 using absolute
`--atol`; it never changes the original result. It is separate from comparing
training outcomes to historical targets.

Summary output contains histories' epoch count/best validation, actual test
metrics, descriptive report deltas and training-run mean/sample standard
deviation. Different numerical configurations are grouped separately (seed
excluded); evaluation-only exports do not count as extra training trials.
Statistics do not establish independent seeds, equal initialization, or equal
datasets. Original input revalidation belongs to `verify`, not summary.

## Precedence, types and paths

Precedence: built-in defaults -> preset inheritance -> JSON config file ->
repeatable `--set` in order -> dedicated `--dataset`/`--modality` flags.
Unknown fields, bad types, NaN/infinite floats, invalid choices and inconsistent
balanced batch structure are rejected. JSON booleans are lowercase
`true`/`false`; `null` means never refresh. Numeric strings are not numbers.
Unquoted enum/string override values are accepted as strings; shell-quote paths
containing spaces and the whole override when needed:

```bash
csi show --preset R9 --set 'paths.data=/remote/data with spaces' \
  --set 'evaluation.sets=["indian.mix","indian.vocal"]' \
  --set objectives.supcon.weight=0.15
```

A changed preset is labeled derived with differences. Dataset/modality
selection is independent and recorded as a difference from the preset default.
Resolved configs always mark historical reproduction false until externally
verified. Relative configured paths resolve against `--root`, otherwise the
configuration file's parent, otherwise current working directory. Explicit CLI
file paths (`--init`, `--inventory`, `--output`, etc.) are shell paths.

## Complete configuration fields

Defaults below are base defaults; use `show` for preset changes. Integer fields
reject booleans/floats. All numeric fields are finite and nonnegative. Unless
specified, positive quantities must be greater than zero. No arbitrary Python
expressions or dynamic imports are evaluated.

| Field | Base default, type, choices/constraints |
|---|---|
| `dataset` | string `indian`; `indian`, `shs100k` |
| `modality` | string `mix`; `mix`, `vocal` |
| `profile` | string `report`; `report`, `upstream` |
| `paths.data` | string `data` |
| `paths.runs` | string `runs` |
| `paths.manifest` | string empty; optional directory overriding dataset manifest discovery |
| `paths.tonic` | string empty; required cache path when tonic enabled |
| `preprocess.frames` | integer 400, minimum 200; report crop/pad length |
| `preprocess.tonic.enabled` | boolean false |
| `preprocess.augmentation` | boolean false; upstream masks/speed/random crop when training |
| `preprocess.sample_rate` | positive integer 22050 |
| `preprocess.hop_length` | positive integer 512 |
| `preprocess.fmin` | positive number 32.70319566257483 Hz |
| `preprocess.bins` | integer 84; fixed by backbone |
| `preprocess.bins_per_octave` | integer 12; fixed |
| `preprocess.aggregate` | positive integer 20 |
| `preprocess.dtype` | string `float64`; only supported storage dtype |
| `model.dropout` | number 0, `[0,1)`; adapter after fc0, only in training |
| `model.freeze_conv0` | boolean false; freeze convolution parameters only |
| `model.batchnorm` | string `train`; `train`, `frozen` (all running stats frozen) |
| `model.replace_classifier` | boolean false; explicit initializer fc1 reset, even when shapes match |
| `objectives.ce.weight` | number 0; enabled when positive |
| `objectives.triplet.weight` | number 0 |
| `objectives.triplet.margin` | positive number 0.3 |
| `objectives.triplet.distance` | string `euclidean`; `euclidean`, `cosine` |
| `objectives.triplet.normalize` | boolean false; normalize before ordinary triplet |
| `objectives.modified_triplet.weight` | number 1 |
| `objectives.modified_triplet.margin` | number 0.3 |
| `objectives.modified_triplet.positive` | string `random`; `random`, `hard`, `mean`, `hard_mean` |
| `objectives.supcon.weight` | number 0 |
| `objectives.supcon.temperature` | positive number 0.07 |
| `objectives.arcface.weight` | number 0 |
| `objectives.arcface.subcenters` | positive integer 3 |
| `objectives.arcface.margin` | number 0.3 radians, less than pi |
| `objectives.arcface.scale` | positive number 30 |
| `negative.selection_refresh_epochs` | positive integer 3 or null; identity refresh |
| `negative.embedding_refresh_epochs` | positive integer 3 or null; independent embedding refresh |
| `negative.embedding_policy` | string `cached`; `cached`, `live` |
| `mining.anchors` | string `originals`; `originals`, `all`; metric anchors only |
| `mining.duplicates` | string `exclude`; first distinct track occurrence in metric losses |
| `mining.skip_batch_without_triplet` | boolean false; optional historical batch gate |
| `sampler.kind` | string `random`; `random`, `balanced` |
| `sampler.cliques_per_batch` | positive integer 16 |
| `sampler.tracks_per_clique` | positive integer 4 |
| `sampler.shortage` | string `error`; `error`, `replace` |
| `sampler.tail` | string `drop`; `drop`, `error`; leftover balanced cliques |
| `sampler.include_original` | boolean true; require/select an original per selected clique |
| `train.epochs` | integer 75; zero requires an initializer and evaluates without optimization |
| `train.lr` | positive number 0.0005 |
| `train.weight_decay` | number 0.0001; Adam |
| `train.batch_size` | positive integer 64; must equal balanced P x K |
| `train.scheduler` | string `cosine`; `cosine`, `plateau` |
| `train.min_lr` | number 0.000001; no greater than LR |
| `train.plateau_factor` | number 0.8, `[0,1)` |
| `train.plateau_patience` | integer 2 |
| `train.patience` | integer 15; zero disables early stopping |
| `train.min_improvement` | number 0.002; resets patience, not best selection |
| `train.seed` | integer 42; Python/NumPy/Torch/CUDA seeds |
| `train.device` | string `cpu`; `cpu`, `cuda`, `cuda:<index>` |
| `train.workers` | integer 0; loader/feature worker processes |
| `split.validation_fraction` | number 0.1, `[0,1)`; explicit val overrides generation |
| `split.seed` | integer 42 |
| `split.rounding` | string `ceil`; only supported policy |
| `evaluation.no_positive` | string `zero`; `zero`, `error` |
| `evaluation.sets` | nonempty string list; default all four canonical dataset.modality keys |

Upstream requires random batches, no tonic, CE weight 1 and all other objective
weights zero. At least one objective must be active for nonzero budgets.
Changing objective components requires no engine edits. Ordinary triplet uses
random distinct in-batch positives and negatives; modified triplet uses the
configured positive miner and the training-partition negative cache.

Random batching shuffles all tracks, keeps the final partial batch, and has
ceil(N/batch_size) batches. Balanced epoch length is
floor(number_of_cliques/P); shuffled leftover cliques are dropped or cause
an explicit error. These policies are fixed and documented rather than hidden
or claimed historically recovered. Loader workers are epoch-boundary seeded;
mid-epoch resume and distributed training are not supported.
