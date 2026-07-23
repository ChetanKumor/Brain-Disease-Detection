# Contributing

Thanks for your interest in improving Brain Disease Detection! This guide covers
the local workflow and the quality bar for changes.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # no TensorFlow needed for tests/linters
```

To run the actual web app or train models you also need the full runtime:

```bash
pip install -r requirements.txt
```

## Before you open a pull request

Run the full quality gate and make sure it passes:

```bash
make check      # ruff (lint) + mypy (types) + pytest (tests)
```

Individually:

```bash
make lint       # ruff check .
make format     # ruff format .
make typecheck  # mypy
make test       # pytest
```

## Guidelines

- **Keep the layers decoupled.** The `web` and `inference` layers must not
  import TensorFlow at module import time — heavy ML imports stay lazy so the
  test suite runs without them. Depend on the `ModelProvider` protocol, not on a
  concrete model.
- **Add tests** for new behaviour. Prefer injecting stub models via fixtures over
  loading real weights.
- **Type hints & docstrings** on new public functions; keep `mypy` clean.
- **Configuration over constants.** New tunables belong in `configs/config.yaml`
  and the typed config in `src/brain_disease_detection/config.py`.
- **No fabricated results.** Don't hard-code or claim accuracy numbers; metrics
  come from an actual training run.

## Commit messages

Use short, imperative, [Conventional Commits](https://www.conventionalcommits.org/)
style prefixes where sensible (`feat:`, `fix:`, `docs:`, `test:`, `build:`,
`chore:`, `refactor:`). Group related changes into atomic commits.

## Reporting issues

Please include the steps to reproduce, the expected vs. actual behaviour, and
your Python / OS versions.
