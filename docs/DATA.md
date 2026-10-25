# Data and recovery

No benchmark assets ship in the package. Downloads are explicit, selected,
pinned, and remote-only. Never infer experimental membership from the HF viewer.

| Asset | Repository | Immutable revision |
|---|---|---|
| Audio/metadata | `rojin254/PP4-indian-music-csi` | `973478c5b13ee5dcd153222289d867a4cc182abd` |
| Earlier weights | `rojin254/PP4-CQTNet-checkpoints` | `5ae4cbfb60fb798c0c5c1854503f23bc543bc7f8` |

`data fetch --file` requires exact filenames (repeat the flag); no glob or
implicit full snapshot is allowed. Each subset gets a checksum/byte-count ledger
with requested files, revision, successes and HTTP failures. Partial failure
returns an error, not success. Recorded checksums establish downloaded bytes,
not experimental provenance or audio completeness. No uploads are implemented.

## Folder mapping

Fetch dataset paths into `data/audio`, preserving their remote relative paths:

| Canonical dataset/modality | Training folder | Test folder |
|---|---|---|
| `indian.mix` | `train/indian` | `test/indian` |
| `indian.vocal` | `train/indian_vocal_train` | `test/indian_vocal` |
| `shs100k.mix` | `train/shs100k_train` | `test/shs100k-test` |
| `shs100k.vocal` | `train/shs100k_train_vocal` | `test/shs100k_test_vocal` |

Do not copy audio to normalize names. Metadata fetched outside those folders
can be placed under `data/metadata` with a separate explicit fetch invocation.
Stored vocal stems are preferable to regenerating different stems.

```text
data/
  audio/{train,test}/<preserved-folder>/
  metadata/tracks.csv
  manifests/<dataset>/{base_train,inventory,train,val,test}.csv
  manifests/<dataset>/{split,label_map}.json
  features/<dataset>/<modality>/<recipe-hash>/<track-id>.{npy,json}
  cache/tonic/<requested-index>.json
  checkpoints/{pretrained,recovered}/
runs/<run-id>/
```

All large data, feature, cache, checkpoint and run paths are ignored by Git.
Small manifests can be versioned after independently confirming membership.
Synthetic fixture roots additionally contain `fixture.json`; fixture outputs
must never be compared with report targets.

## Inventory schema

CSV headers are mandatory. Strings are explicit; IDs are not guessed from audio
filenames:

| Column | Meaning |
|---|---|
| `dataset` | `indian` or `shs100k` |
| `modality` | `mix` or `vocal` |
| `track_id` | Stable ID, identical across modalities; no path separators |
| `clique_id` | Stable cover clique; no path separators |
| `is_original` | Literal `true` or `false` |
| `source_track_id` | Original source identity shared by augmented siblings |
| `audio_path` | Relative to configured data root; no parent traversal |
| `partition` | Existing `train`, explicit `val`, or original `test` |
| `feature_path` | Prepared relative feature path |
| `feature_sha256` | Prepared array checksum |
| `feature_recipe` | Prepared recipe SHA-256 |

The last three can be empty on audio import; `prepare --features` populates
them. Import is an explicit CSV contract, not an unreliable filename-to-clique
converter. Legacy list membership must be translated into this schema and
verified before use.

```bash
uv run csi data prepare --inventory data/metadata/tracks.csv --features \
  --set train.workers=4
```

Without `--features`, preparation only persists metadata; doctor will reject
missing feature provenance. With it, outputs and `metadata/prepared_tracks.csv`
are created without overwriting the input inventory.

## Splits and guarantees

For each dataset, use explicit validation membership if supplied. Otherwise
sort training clique strings, shuffle using split seed 42 and select
`ceil(0.1 * number_of_cliques)`. Explicit `val` takes precedence over generating
another holdout. This is a **reconstructed**, deterministic policy, not proof of
the lost validation identities. Preserve original test membership.

Persist base training order, inventory checksum, seed, rounding, validation
cliques, per-modality partition identities, original test IDs, and a sorted
report label map. Later preparation must match saved split provenance exactly;
use a new root for an intentional change. Do not overwrite an experiment's split.
Upstream CE retains numeric clique IDs instead of the report's dense label map;
each run exports its actual label map.

Reject duplicate track identities, cross-modality identity inconsistencies,
cross-partition clique or augmentation-family leakage, missing/corrupt arrays,
modified saved membership, wrong dtype/shape, non-finite features, stale recipes,
and changed audio checksums. A feature sidecar is required. Doctor validates
train, val and every requested test set before run allocation.

Cross-modality IDs must agree wherever both modalities are present; partial
source availability is not proof of completeness. Compare manifests against the
original inventory and reference counts before claiming benchmark recovery:
Indian 2,813/709 tracks, 402/139 cliques; SHS subset 1,826/591 tracks, 92/24
cliques. The 2,149-track plot remains unexplained; do not delete data to match it.

## Recipes and tonic caches

Audio is loaded mono float32 at 22,050 Hz with `soxr_hq`, offset 0, no duration
limit. CQT has 84 bins, 12/octave, C1 minimum frequency, hop 512, tuning 0,
filter scale 1, norm 1, sparsity 0.01, Hann window, scale enabled, constant
padding, complex64 output. Magnitudes are averaged in complete 20-frame
windows into float64 arrays; remainder frames are dropped. Inputs under 1,000
samples or without a complete window fail. Silence is valid zero magnitude.

Recipe fingerprints include every relevant setting, loader/CQT constants,
audio checksum and NumPy/librosa/SciPy/soundfile/soxr versions. Sidecars record
shape/dtype and array checksum. Existing assets must match, not silently be
replaced. Modern extraction is tested; historical librosa parity is unverified.

Report loading divides by the maximum of the **full** array plus `1e-6` before
front cropping/padding, then adds a channel axis. Upstream loader behavior is
documented in [UPSTREAM](UPSTREAM.md).

Tonic is the argmax of time-mean magnitude folded into 12 pitch classes.
Rotate within each octave, never across octave boundaries. Silence yields 0
by deterministic argmax. Build an index only when needed:

```bash
uv run csi data tonic --output data/cache/tonic/indian-mix.json
uv run csi data tonic --output data/cache/tonic/indian-mix.json --check
uv run csi doctor --preset R2 --init /absolute/dynamic/best.pt \
  --set preprocess.tonic.enabled=true \
  --set paths.tonic=data/cache/tonic/indian-mix.json
```

The index covers selected training/validation modality and all requested test
sets. Entries include feature checksum, recipe and estimator ID; missing/stale
entries fail. Different training modalities may need separate index files.

## Optional vocal separation

`data separate-vocals` uses an explicitly installed Demucs, explicit inputs,
device/output/model, and requires `--allow-model-download` because Demucs may
retrieve weights. It records source/output hashes and fails on missing output.
Demucs and its weights were **not** installed or exercised in this rebuild.
Separation does not rewrite metadata or silently replace recovered vocal audio.
