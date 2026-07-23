# Data

Datasets are **not** committed to this repository — they are large and often
licensed. This directory documents the layout the training pipeline expects and
is otherwise ignored by git.

## Expected layout

The training CLI reads an image-classification directory with one sub-folder per
class. Sub-folder names are read **alphabetically**, which must match the
`class_labels` order in `configs/config.yaml`.

```
data/
├── brain_tumor/
│   ├── Glioma/        *.jpg
│   ├── Meningioma/    *.jpg
│   ├── NoTumor/       *.jpg
│   └── Pituitary/     *.jpg
├── brain_stroke/
│   ├── Haemorrhagic/  *.jpg
│   ├── Ischemic/      *.jpg
│   └── Normal/        *.jpg
└── alzheimers/
    ├── MildDemented/      *.jpg
    ├── ModerateDemented/  *.jpg
    ├── NonDemented/       *.jpg
    └── VeryMildDemented/  *.jpg
```

The pipeline creates reproducible train / validation / test splits from this
single directory (ratios configured under `training:` in the config).

## Suggested public datasets

- **Brain Tumor MRI** — multi-class glioma / meningioma / pituitary / no-tumor MRI scans.
- **Brain Stroke CT** — haemorrhagic / ischemic / normal CT scans.
- **Alzheimer's MRI** — four-stage dementia MRI scans.

Download a dataset, arrange it as above, then run `scripts/train.py`.
