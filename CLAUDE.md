# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research code for a Master's thesis (Mémoire) on automatic breast density classification from
mammography DICOM images (4 classes: DENSITY A/B/C/D, BI-RADS-style). Models are trained on the
VinDr-Mammo dataset and evaluated on DDSM. All code lives under `Codes/`; everything outside it
(PDF report, LaTeX slides) is not source code.

This is experimental research code developed iteratively for a thesis, not a production library —
expect duplicated scripts across variant directories (see Architecture below) rather than shared
abstractions.

## Environment

- A pre-existing venv at `../.venv` (repo root) targets **Python 3.9** — use it, not the system
  `python3` (which may be a different version).
- Dependencies: `pip install -r Codes/requirements.txt` (torch, torchvision, timm, transformers,
  pydicom, opencv-python, scikit-image, scikit-learn, pandas).
- GPU training is normally run on a SLURM cluster (see `run_*_slurm.sh` scripts); `python3 -u` is
  used for unbuffered output in job logs.
- No linter/formatter or automated test runner (pytest, etc.) is configured. "Test" scripts
  (`test_*.py`) are standalone scripts with `if __name__ == '__main__':` blocks meant to be run
  directly (e.g. `python3 test_models.py`), not collected by pytest.

## Data layout expected by the code

```
<IMAGE_ROOT>/
├── train/density_A|B|C|D/<study_id>/<image_id>.dicom
└── test/density_A|B|C|D/<study_id>/<image_id>.dicom
```
with a fallback to the raw VinDr-Mammo layout (`<IMAGE_ROOT>/images/<study_id>/<image_id>.dicom` or
`<IMAGE_ROOT>/<study_id>/<image_id>.dicom`) when the organized layout isn't found — see
`FeatureDumper.dump_features` in [Codes/models.py](Codes/models.py) and `MammographyDataset` in
[Codes/training.py](Codes/training.py) for the exact lookup order.

An annotations CSV (VinDr `breast-level_annotations.csv` or a filtered `output_annotations.csv`,
see `Codes/DDSM/`) provides `study_id`, `image_id`, `breast_density`, `laterality`, `split`
columns. Class mapping is always `{"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}`.

Dataset paths are hardcoded absolute paths at the top of each script (`CONFIG` dict / `VINDR_ROOT`
in [Codes/training.py](Codes/training.py), `--data_csv`/`--image_root` args elsewhere, cluster paths
under `/home_nfs/...` in the SLURM scripts, local paths under `/Volumes/...` for interactive runs).
When running locally vs. on the cluster, expect to override these via CLI args or edit `CONFIG`.

## Architecture

There are two parallel model families sharing the same DICOM preprocessing and feature-extraction
core, plus several near-duplicate variant directories.

### 1. Transformer/CNN backbone + feature-based MLPs (`Codes/training.py`, `Codes/models.py`)

`training.py` is a unified CLI (`--mode {augment, finetune, finetune_augmented, dump_features,
train_on_features, train}`) driving a pipeline:

1. **augment** — builds a class-balanced augmented dataset CSV (`*_augmented.csv`).
2. **finetune / finetune_augmented** — fine-tunes a backbone classifier end-to-end
   (`FineTunedFeatureExtractor` in `models.py`), optionally on the augmented set.
3. **dump_features** — runs a frozen or fine-tuned backbone (`FeatureExtractor`) over all images and
   caches `.npy` feature/label arrays (`FeatureDumper`).
4. **train_on_features / train** — trains lightweight MLP heads on the cached features:
   `MLPClassifier` (4-class) and/or `BinaryMLPClassifier` for every class pair, via
   `FeatureMLPTrainer`.

`HierarchicalClassifier` composes the 4-class MLP with all pairwise binary MLPs: it predicts the
top-2 candidate classes with the 4-class head, then refines with the corresponding binary
classifier. `Codes/three_stage_pipeline.py` implements an alternative cascade of exactly 3 binary
models: (A,B) vs (C,D), then A vs B, then C vs D.

Backbones (`BACKBONES` / `CVT_CONFIGS` dicts in `models.py`) come from two libraries:
`timm.create_model` for ViT/DeiT/Swin/PVT/Twins/ResNet/EfficientNet, and Hugging Face
`transformers` (`AutoModel`) for CVT variants (`cvt`, `cvt-13`, `cvt-21`, `cvt-w24`) — CVT models
run a per-image PIL round-trip through an `AutoImageProcessor` inside `forward()`, which is far
slower than the batched timm path.

### 2. Hybrid two-branch model (`hybrid_model.py` / `hybrid_finetuning.py`, present in several
   variant directories — see below)

Combines an `ImageBranch` (CNN or ViT feature extractor over the mammogram, `feature_dim=512`) with
a second tabular branch — either `HistogramMLP` (256-bin grayscale histogram → 64-D) or
`GLCMDescriptorMLP` (5 GLCM texture descriptors: contrast, homogeneity, energy, correlation,
entropy → 16-D), concatenated and passed through a final MLP to the 4 density classes. Trained
end-to-end with `hybrid_finetuning.py` (early stopping, LR scheduling, checkpoint resume via
`--resume_from`), evaluated with `test_hybrid_model.py`.

### Variant directories

`hybrid 2 branches images/`, `hybrid train mlo resnet/`, `hybrid train mlo rexnet/`,
`hybrid_train_cc_rexnet/`, `hybridemodel avec stages train mlo resnet/` each contain their **own
copies** of `hybrid_model.py`, `hybrid_finetuning.py`, `test_hybrid_model.py`, `preprocessing.py`
tuned for a specific backbone/view combination (ResNet50 vs RexNet150; CC view vs MLO view; with or
without GLCM). These are not imported from each other — they are independently editable snapshots.
**When fixing a bug in shared logic (e.g. DICOM reading, GLCM extraction), check whether the same
bug exists in the sibling copies under the other variant directories** before assuming a fix in one
location is complete.

### DICOM preprocessing (`Codes/preprocessing.py`)

`read_dicom` handles `MONOCHROME1` vs `MONOCHROME2` photometric interpretation (inverting
MONOCHROME1 so all images share a black background). `preprocess_image` crops 700px based on
`laterality` (L/R), resizes to 224×224, normalizes intensity to `[0,255]`, and replicates the
single channel to RGB for backbones pretrained on 3-channel inputs. `analyze_dicom_metadata` /
`batch_analyze_photometry` are diagnostic helpers for photometry inconsistencies across the dataset
(see also `Codes/diagnose_dicom.py`, `Codes/test_dicom_photometry.py`).

### Other entry points

- [Codes/predict_gui.py](Codes/predict_gui.py) — GUI for predicting density from a single image.
- [Codes/browse_mammograms.py](Codes/browse_mammograms.py) — interactive DICOM browsing.
- [Codes/eval_image_branches.py](Codes/eval_image_branches.py) — evaluates image-only branches.
- [Codes/generate_augmented_dataset.py](Codes/generate_augmented_dataset.py) — standalone
  augmented-dataset generator (also reachable via `training.py --mode augment`).
- [Codes/inspect_checkpoint.py](Codes/inspect_checkpoint.py) — inspects a saved `.pth` checkpoint's
  keys/shapes.
- [Codes/DDSM/filtrage_images.py](Codes/DDSM/filtrage_images.py) /
  [Codes/DDSM/analyze_study_consistency.py](Codes/DDSM/analyze_study_consistency.py) — dataset
  filtering/consistency scripts for DDSM annotations.
- `run_*_slurm.sh` at the repo root and inside variant directories — SLURM batch launchers, one per
  training/eval job; read the matching script to see the exact CLI flags used for a given run.

## Common commands

Run from `Codes/` (or the relevant variant directory) with the project venv active.

```bash
# Feature-based pipeline (Codes/training.py)
python training.py --mode augment --target_samples 2000
python training.py --mode finetune --backbone cvt-w24 --epochs 100 --batch_size 10 --use_augmentation
python training.py --mode dump_features --backbone cvt-w24 --use_finetuned
python training.py --mode train_on_features --backbone cvt-w24 --train_mlp4 --all_binaries --use_finetuned

# Hybrid two-branch model (from within a variant directory, e.g. "hybrid 2 branches images/")
python hybrid_finetuning.py --epochs 50 --batch_size 8 --lr 1e-4 --use_augmentation
python test_hybrid_model.py

# Standalone diagnostic/test scripts (no pytest — run directly)
python test_models.py
python test_dicom_photometry.py
python test_section623.py
```

See [Codes/UNIFIED_TRAINING_GUIDE.md](Codes/UNIFIED_TRAINING_GUIDE.md) for the full set of
`training.py` modes, recommended per-backbone hyperparameters, and multi-step workflows (e.g.
augment → finetune_augmented → dump_features → train_on_features).
