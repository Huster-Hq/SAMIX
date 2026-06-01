# SAMIX Lite

`SAMIX Lite` is a concise re-implementation of the core idea behind SAMIX:

- a lightweight semantic adapter inspired by SANSA / AdaptFormer
- reference retrieval from a support bank
- retrieval-conditioned segmentation on the query image

This repository intentionally does **not** include `SPNet` yet. The current goal is
to recover a clean, hackable baseline after the original code was lost.

## What is included

- `samix/adapters.py`: semantic adapter blocks
- `samix/backbones.py`: a small convolutional image encoder with adapter injection
- `samix/retrieval.py`: support prototype extraction and top-k retrieval
- `samix/model.py`: an end-to-end few-shot segmentation model
- `docs/data_format.md`: mixed-supervision dataset storage format
- `scripts/build_manifest.py`: automatic split/manifest generator
- `scripts/demo_random.py`: smoke demo on random tensors
- `tests/test_smoke.py`: basic shape and gradient checks

## Method sketch

1. Encode support and query images with a frozen-or-mostly-frozen visual backbone.
2. Insert semantic adapters in the late stages of the encoder.
3. Build support prototypes with masked average pooling over support features.
4. Retrieve the most relevant support prototypes for each query by cosine similarity.
5. Fuse retrieved prototypes back into the query feature map and predict a mask.

This is not a line-by-line reproduction of the CVPR 2026 paper. It is a compact
research scaffold that preserves the main modeling logic while staying easy to
extend toward a fuller SAM2-based version later.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python scripts/demo_random.py
python -m unittest discover -s tests
```

## Mixed-supervision dataset format

`SAMIX Lite` now includes a recommended storage format for mixed annotations:

- one `images/` folder
- parallel `masks/`, `boxes/`, `scribbles/`, `points/`, `classes/`, `unlabeled/`
- one `splits.json` manifest describing train / val / test and label types

See:

- [docs/data_format.md](C:/Users/HuQiang/Documents/SAMIX_Github/docs/data_format.md)
- [examples/dataset_manifest.example.json](C:/Users/HuQiang/Documents/SAMIX_Github/examples/dataset_manifest.example.json)

You can auto-generate a manifest with:

```bash
python scripts/build_manifest.py --dataset-root /path/to/dataset --dataset-name my_dataset --output /path/to/dataset/splits.json
```

## Example

```python
import torch

from samix import build_samix_lite

model = build_samix_lite()

batch = {
    "support_images": torch.randn(2, 3, 3, 256, 256),
    "support_masks": torch.randint(0, 2, (2, 3, 1, 256, 256)).float(),
    "query_images": torch.randn(2, 3, 256, 256),
}

out = model(**batch)
print(out["logits"].shape)
print(out["retrieval_indices"].shape)
```

## Repository layout

```text
SAMIX_Github/
|-- README.md
|-- requirements.txt
|-- scripts/
|   `-- demo_random.py
|-- samix/
|   |-- __init__.py
|   |-- adapters.py
|   |-- backbones.py
|   |-- model.py
|   `-- retrieval.py
`-- tests/
    `-- test_smoke.py
```

## Next steps

Recommended next upgrades:

1. Replace the toy backbone with a real SAM2 image encoder wrapper.
2. Add prompt encoders for point / box / scribble / mask support.
3. Add multi-scale retrieval and temporal memory.
4. Re-introduce `SPNet` as a learned retrieval policy on top of the current bank.
