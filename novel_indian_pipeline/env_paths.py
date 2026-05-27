import os


def _load_env_file(path):
    """Tiny .env loader (no dependency on python-dotenv).

    Lines that are blank or start with '#' are ignored. Surrounding quotes are
    stripped from values. Existing process env wins over .env.
    """
    if not os.path.exists(path):
        return
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_HERE = os.path.dirname(os.path.abspath(__file__))
_load_env_file(os.path.join(_HERE, '.env'))


def _req(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f'Required path {key} is not set. '
            f'Copy novel_indian_pipeline/.env.example to .env and edit it.'
        )
    return val


# ---------------------------------------------------------------------------
# Per-dataset bundles: (cqt_dir, train_txt|None, test_txt, ckpt|None, fmt).
# fmt 'indian'  -> "<filename>,<clique>" with is_original inferred from suffix
# fmt 'shs100k' -> "<filename>,<clique>,<is_original_int>"
# ---------------------------------------------------------------------------
DATASETS = {
    'indian': {
        'cqt_dir': lambda: _req('INDIAN_CQT_DIR'),
        'train_txt': lambda: _req('INDIAN_TRAIN_TXT'),
        'test_txt': lambda: _req('INDIAN_TEST_TXT'),
        'ckpt': lambda: _req('INDIAN_CKPT'),
        'fmt': 'indian',
    },
    'indian_vocal': {
        'cqt_dir': lambda: _req('INDIAN_VOCAL_CQT_DIR'),
        'train_txt': lambda: _req('INDIAN_VOCAL_TRAIN_TXT'),
        'test_txt': lambda: _req('INDIAN_VOCAL_TEST_TXT'),
        'ckpt': lambda: _req('INDIAN_VOCAL_CKPT'),
        'fmt': 'indian',
    },
    'shs100k': {
        'cqt_dir': lambda: _req('SHS100K_CQT_DIR'),
        'train_txt': lambda: None,  # eval-only in this pipeline
        'test_txt': lambda: _req('SHS100K_TEST_TXT'),
        'ckpt': lambda: _req('SHS100K_CKPT'),
        'fmt': 'shs100k',
    },
    'shs100k_vocal': {
        'cqt_dir': lambda: _req('SHS100K_VOCAL_CQT_DIR'),
        'train_txt': lambda: None,
        'test_txt': lambda: _req('SHS100K_VOCAL_TEST_TXT'),
        'ckpt': lambda: _req('SHS100K_VOCAL_CKPT'),
        'fmt': 'shs100k',
    },
}


def dataset_paths(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name!r}. Choices: {list(DATASETS)}')
    spec = DATASETS[name]
    return {
        'cqt_dir': spec['cqt_dir'](),
        'train_txt': spec['train_txt'](),
        'test_txt': spec['test_txt'](),
        'ckpt': spec['ckpt'](),
        'fmt': spec['fmt'],
    }


def results_dir():
    return _req('RESULTS_DIR')


def logs_dir():
    return _req('LOGS_DIR')


def new_ckpt_dir():
    return _req('NEW_CKPT_DIR')


def tonic_index_dir():
    return _req('TONIC_INDEX_DIR')
