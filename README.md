# SAMIX

Official codebase in progress for:

**SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy**

CVPR 2026

`SAMIX` studies mixed-supervision segmentation with a SAM2-centered design. The
full method contains two major parts:

- `SA-SAM2`: a SAM2-based in-context segmentor with semantic adapters inserted
  into the image encoder
- `SPNet`: a reference selecting policy for retrieving useful support examples

This repository currently focuses on the first part and the training scaffold
around it:

- SAM2 image encoder with SANSA-style semantic adapters
- Polyp-PVT segmentation model with EMA teacher
- warmup + co-training pipeline
- mixed-supervision dataset format for `mask / box / scribble / point / class / unlabeled`

`SPNet` is not included yet.

## Overview

The current training framework has two stages.

### 1. Warmup stage

Only `SA-SAM2` is trained. The original SAM2 parameters stay frozen, and only
the semantic adapter parameters are updated. Training follows a few-shot /
in-context segmentation style:

- support images provide high-quality mask supervision
- query images are segmented with SAM2 memory and prompt conditioning
- the adapter learns to inject task-specific semantics into the SAM2 encoder

### 2. Co-training stage

After warmup, three modules are optimized together:

- `SA-SAM2`
- `Polyp-PVT` segmentation model
- `EMA` copy of `Polyp-PVT` as teacher

In this stage, `SA-SAM2` acts as an in-context segmentor that uses support
images as semantic references, while `Polyp-PVT` learns from a mix of:

- dense supervision on mask-labeled samples
- weak supervision on box / scribble / point / class labels
- pseudo-label and consistency signals on weak or unlabeled samples

## Repository Status

This repository has already been brought to a runnable baseline:

- `SA-SAM2` can be built on top of the official `SAM2`
- `Polyp-PVT` can be instantiated with `timm`
- `warmup_step` and `cotrain_step` have passed remote smoke tests on the target server

At the same time, this is still an active research codebase rather than a final
camera-ready release. In particular:

- `SPNet` is not implemented yet
- some weak-supervision losses are practical defaults and may still be refined
- training scripts are ready for real data, but final benchmark configs and
  dataset-specific recipes still need to be polished

## Code Structure

```text
SAMIX/
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- data_format.md
|-- examples/
|   `-- dataset_manifest.example.json
|-- samix/
|   |-- adapters.py          # semantic adapter modules
|   |-- data.py              # mixed-supervision dataset and episode sampling
|   |-- ema.py               # EMA teacher
|   |-- framework.py         # top-level training container
|   |-- losses.py            # dense and weak supervision losses
|   |-- model_utils.py       # dataclasses and utility structures
|   |-- polyp_pvt.py         # Polyp-PVT segmentation model
|   |-- prompts.py           # annotation -> SAM prompt conversion
|   |-- sa_hiera.py          # SAM2 Hiera trunk with semantic adapters
|   |-- sa_sam2.py           # SAM2-based in-context segmentor
|   |-- training.py          # warmup and co-training logic
|   `-- __init__.py
`-- scripts/
    |-- build_manifest.py    # build dataset manifest json
    |-- demo_random.py       # quick random-tensor smoke test
    |-- train_cotrain.py     # warmup + co-training entrypoint
    `-- upload_to_4090_6_hq.py
```

## Installation

### 1. Create an environment

Recommended baseline:

- Python `3.10`
- PyTorch `2.5.1`
- CUDA `12.4` or compatible driver/runtime

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install the official SAM2 package

This repository expects the official `SAM2` codebase to be available. One
practical setup is:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

If you are working in an environment where building custom CUDA ops is not
desired, install SAM2 in the same way you validated locally for your platform.

### 3. Prepare SAM2 checkpoints

`train_cotrain.py` requires:

- `--sam2-root`: path to the official `segment-anything-2` repo
- `--sam2-ckpt`: path to a SAM2 checkpoint such as `sam2_hiera_tiny.pt`,
  `sam2_hiera_base_plus.pt`, or `sam2_hiera_large.pt`
- `--sam2-config`: matching config name such as `sam2_hiera_t.yaml` or
  `sam2_hiera_l.yaml`

## Mixed-Supervision Dataset Format

The repository uses one shared dataset layout for all supervision types.

```text
dataset_root/
|-- images/
|-- masks/
|-- boxes/
|-- scribbles/
|-- points/
|-- classes/
|-- unlabeled/
`-- splits.json
```

The key idea is simple:

- image files live in `images/`
- each supervision type has a parallel annotation folder
- `splits.json` declares train / val / test membership and the available label
  types for each image

See:

- [docs/data_format.md](C:\Users\HuQiang\Documents\SAMIX_Github\docs\data_format.md)
- [examples/dataset_manifest.example.json](C:\Users\HuQiang\Documents\SAMIX_Github\examples\dataset_manifest.example.json)

To generate a manifest automatically:

```bash
python scripts/build_manifest.py \
  --dataset-root /path/to/dataset \
  --dataset-name my_dataset \
  --output /path/to/dataset/splits.json
```

## Training

The main training entrypoint is:

```bash
python scripts/train_cotrain.py \
  --manifest /path/to/dataset/splits.json \
  --sam2-root /path/to/segment-anything-2 \
  --sam2-ckpt /path/to/checkpoints/sam2_hiera_tiny.pt \
  --sam2-config sam2_hiera_t.yaml \
  --image-size 512 \
  --shots 1 \
  --warmup-epochs 5 \
  --joint-epochs 20 \
  --batch-size 2 \
  --num-workers 4 \
  --device cuda
```

### What the script does

1. loads the mixed-supervision training split
2. builds warmup few-shot episodes from mask-labeled samples
3. builds co-training episodes with support masks and mixed query supervision
4. initializes `SA-SAM2`, `Polyp-PVT`, and `EMA`
5. runs warmup training
6. switches to joint co-training

## Current Supervision Support

The current codebase includes losses or handling for:

- `mask`: dense segmentation supervision
- `box`: projection-style `M2B` box supervision
- `scribble`: sparse stroke supervision
- `point`: sparse point supervision
- `class`: image-level presence supervision
- `unlabeled`: pseudo-label and EMA consistency learning

These are implemented mainly in:

- [samix/losses.py](C:\Users\HuQiang\Documents\SAMIX_Github\samix\losses.py)
- [samix/training.py](C:\Users\HuQiang\Documents\SAMIX_Github\samix\training.py)

## Quick Sanity Checks

### Python syntax check

```bash
python -m py_compile samix/*.py scripts/train_cotrain.py
```

### Minimal import / build check

```python
from samix import build_sa_sam2, PolypPVT

seg_model = PolypPVT(pretrained_backbone=False)
print(type(seg_model).__name__)
```

## Reproducibility Notes

If you want to reproduce experiments cleanly, we recommend tracking the
following in your run logs:

- SAM2 checkpoint and config name
- image size
- support shot count
- warmup epochs
- joint training epochs
- optimizer settings
- supervision-type sampling ratios

Because this repository is still being reconstructed and polished, we recommend
pinning:

- the exact `SAM2` commit
- the exact conda / pip environment
- dataset manifest version

## Planned Updates

The next high-priority items are:

1. add `SPNet`
2. align the remaining weak-supervision objectives with the final paper version
3. add benchmark-specific configs and scripts
4. add evaluation and inference entrypoints
5. add released checkpoints and final reproduction recipes

## Acknowledgements

This implementation is built around:

- the official `SAM2` codebase
- the semantic adapter idea used in `SANSA`
- `Polyp-PVT` as the auxiliary segmentation model family

## Citation

If you use this repository, please cite the SAMIX paper. A BibTeX entry will be
added here once the final release metadata is ready.
