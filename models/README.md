# Models

Trained model weights live here as `*.keras` files. They are **not** committed
to version control (see `.gitignore`) because they are large binaries — treat
them as build artifacts distributed via releases or a model registry.

Expected files (names come from `configs/config.yaml`):

| Disease        | File                     |
| -------------- | ------------------------ |
| Alzheimer's    | `alzheimer_model.keras`  |
| Brain Stroke   | `brainstroke_model.keras`|
| Brain Tumor    | `brain_tumor_model.keras`|

## Producing the models

Train them from a labelled dataset with the training CLI:

```bash
python scripts/train.py --disease brain_tumor --data-dir data/brain_tumor
```

Each run also writes a `<disease>_metrics.json` file next to the model with the
test-set accuracy/loss and the training history. The application loads whatever
`*.keras` files are present and reports which models are available on its
status endpoint.
