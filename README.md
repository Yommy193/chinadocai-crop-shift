# ChinaDocAI 2026: detector-aware crop alignment

This repository contains the core code for the CSIG/ChinaDocAI 2026
study on detector-aware crop alignment for oracle-bone document recognition.
It supports the controlled cross-evaluation of Anchor, Faster R-CNN, and FCOS
crop sources, CNN recognizer training, target evaluation, validation summaries,
and the corresponding aggregate figures.

## Repository contents

- `src/oracle_vit/`: model, dataset, inference, and evaluation package code.
- `scripts/`: crop alignment construction, training, cross-detector evaluation,
  aggregation, and portable figure generation scripts.
- `configs/`: the CNN baseline configuration used by the study.
- `tests/`: focused tests for the crop-shift and CNN components.

## Deliberately excluded

Raw oracle-bone/HWOBC images, detector outputs, recognizer checkpoints, cached
arrays, JSONL manifests, logs, generated figures, paper files, and local machine
paths are excluded. The `.gitignore` blocks these files. Supply them locally and
pass their paths explicitly when running an experiment.

## Quick start

```bash
python -m pip install -e .
python -m pytest tests
```

The main workflow is: (1) build detector-aware alignment manifests with
`build_crop_shift_alignment_data.py`, (2) train the recognizer with
`train_crop_shift_cnn_cached.py` or `train_crop_shift_cnn.py`, (3) evaluate each
checkpoint with `evaluate_crop_shift_targets.py`, and (4) aggregate the three
detector cross-evaluations with `aggregate_fcos_cross_evals.py`. All output
locations are supplied by command-line arguments or are ignored generated
directories.

The HWOBC training script defaults to `data/hwobc/sample` as a placeholder.
The dataset itself is not part of this repository.

## Reproducibility boundary

The scripts preserve the CSIG supplement's validation-only protocol and keep
training crop source separate from evaluation detector. Re-running an experiment
requires the same local data split, detector crop manifests, checkpoint files,
and random seeds used in the paper's experiment record.
