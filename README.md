# Modular CQTNet experiment toolkit

An installable research toolkit for cover-song identification in Indian popular
music. One configurable engine supports cross-entropy, ordinary triplet, static
and dynamic modified triplet, all 23 report configurations, pretrained
evaluation, Indian-domain fine-tuning and custom compositions.
ByteCover and TPPNet are outside the implementation scope.

## Setup

Run on an authorized remote Linux host. A GPU is **not required** for the
synthetic smoke tests; the locked environment uses CPU PyTorch.

```bash
uv sync --frozen --extra audio --extra test
uv run csi list
uv run csi suite --group phase3 --plan
uv run --extra audio --extra test pytest tests -q
```

The installed `csi` command and `python -m csi` expose the same interface.
Listing, configuration inspection and suite planning do not download assets
or start training. Installation downloads only declared package dependencies.

## First CPU smoke run

```bash
uv run csi data fixture --output data/fixture
uv run csi doctor --preset prerequisites.ce \
  --set paths.data=data/fixture --set train.batch_size=4
uv run csi train --preset prerequisites.ce --run-id fixture-ce \
  --set paths.data=data/fixture --set train.device=cpu \
  --set train.batch_size=4 --set preprocess.frames=200 --set train.epochs=2
uv run csi verify --run runs/fixture-ce --reevaluate \
  --output-run-id fixture-ce-parity
```

Use a fresh fixture directory and run IDs. The fixture is synthetic CQT,
**not** a scientific benchmark. Training, checkpoint reload, exact
epoch-boundary resume, four-set evaluation and integrity verification are
covered by CPU tests. For a smaller focused test invocation:

```bash
uv run --extra audio --extra test pytest tests/test_lifecycle.py tests/test_workflows.py -q
```

## Package layout

```text
src/csi/          Shared CLI, training, evaluation and scientific components
src/csi/presets/  Upstream/prerequisite profiles and all report configurations
tests/           CPU contracts, lifecycle checks and pinned upstream reference
docs/            Data, commands, evidence and reproduction instructions
.github/         CPU CI
```

Manifests, features, downloaded audio/weights and run outputs are prepared
explicitly under the documented data/output roots. Generated assets do not
belong in source control.

## Documentation

- [Upstream architecture, licensing and historical references](docs/UPSTREAM.md)
- [Data layout, manifests and pinned asset recovery](docs/DATA.md)
- [Complete CLI and configuration reference](docs/CLI.md)
- [Experiment rows, components, aliases and uncertainties](docs/EXPERIMENTS.md)
- [Execution, CPU verification, resume and troubleshooting](docs/REPRODUCE.md)

## Cleanup and scientific status

Work is isolated on `refactor/cqtnet-toolkit`, starting at
`4a636cf0944e7c6b60f838845b8073fafb06ee59`. At the user's request, cleanup
proceeded after CPU validation without waiting for a GPU: 250 unchanged tracked
legacy files (637.9 MiB) were removed, including duplicate pipelines, old
launchers, reports, logs, figures and binary weights. The toolkit does not
import those files. The original branch, Git history and user's untracked plan
are preserved; nothing was moved into an archive.

Historical material remains available at the
[recovery-source commit](https://github.com/rojin254/cover_song_identification_for_indian_popular_music/tree/4a636cf0944e7c6b60f838845b8073fafb06ee59).
The unchanged upstream backbone and independent test reference retain the
original [MIT license](LICENSE).

CPU smoke success is not proof of historical reproduction. GPU execution,
complete dataset/checkpoint recovery, historical audio-runtime parity and
full report reruns remain unverified. Reference metrics and unresolved settings
are labeled explicitly. No full dataset download, paid compute, full training
suite, upload, commit or push is performed automatically.

Nine selected historical checkpoints have been recovered locally, renamed and
CPU-checked. See the [checkpoint mapping and test-exposure warnings](docs/DATA.md#recovered-historical-checkpoints).
