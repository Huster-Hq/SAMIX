# Data Format

This document defines the recommended storage format for mixed-supervision
training data in `SAMIX Lite`.

The design goal is:

- one image folder
- parallel annotation folders for each supervision type
- one JSON manifest describing split membership and available labels

## Directory layout

```text
dataset_root/
|-- images/
|   |-- img_000001.jpg
|   |-- img_000002.jpg
|   `-- ...
|-- masks/
|   |-- img_000001.png
|   `-- ...
|-- boxes/
|   |-- img_000003.json
|   `-- ...
|-- scribbles/
|   |-- img_000004.json
|   `-- ...
|-- points/
|   |-- img_000005.json
|   `-- ...
|-- classes/
|   |-- img_000006.json
|   `-- ...
|-- unlabeled/
|   |-- img_000007.json
|   `-- ...
`-- splits.json
```

All annotation folders are parallel to `images/`.

## Naming rule

Every annotation file uses the same stem as the image file:

- image: `images/img_000123.jpg`
- mask: `masks/img_000123.png`
- box: `boxes/img_000123.json`
- scribble: `scribbles/img_000123.json`
- point: `points/img_000123.json`
- class: `classes/img_000123.json`
- unlabeled marker: `unlabeled/img_000123.json`

If one image has multiple supervision types, it can appear in multiple folders.
For example, the same image may have both `mask` and `class` annotations.

## Annotation file formats

### 1. Mask

Store a binary or indexed PNG mask:

```text
masks/img_000123.png
```

Recommended:

- background = `0`
- foreground / class ids > `0`

### 2. Box

Store boxes as JSON:

```json
{
  "image_id": "img_000123",
  "boxes": [
    { "xyxy": [45, 61, 220, 310], "class_name": "dog", "class_id": 17 }
  ]
}
```

### 3. Scribble

Store each scribble as a polyline:

```json
{
  "image_id": "img_000123",
  "scribbles": [
    {
      "points": [[41, 55], [48, 62], [53, 74]],
      "class_name": "dog",
      "class_id": 17
    }
  ]
}
```

### 4. Point

Store points as sparse supervision:

```json
{
  "image_id": "img_000123",
  "points": [
    { "xy": [88, 120], "label": 1, "class_name": "dog", "class_id": 17 },
    { "xy": [15, 20], "label": 0, "class_name": "background", "class_id": 0 }
  ]
}
```

Suggested convention:

- `label = 1` foreground point
- `label = 0` background point

### 5. Class

Store image-level labels:

```json
{
  "image_id": "img_000123",
  "classes": [
    { "class_name": "dog", "class_id": 17 }
  ]
}
```

### 6. Unlabeled

Store a tiny marker JSON so the sample can be included explicitly:

```json
{
  "image_id": "img_000123",
  "status": "unlabeled"
}
```

You may omit unlabeled images from annotation folders entirely, but keeping an
explicit `unlabeled/` folder makes curation and semi-supervised training easier.

## Split manifest

The file `splits.json` stores:

- dataset metadata
- train / val / test image ids
- available annotation types per image
- optional class information

Example:

```json
{
  "dataset_name": "samix_mixed_supervision_v1",
  "root": ".",
  "annotation_types": ["mask", "box", "scribble", "point", "class", "unlabeled"],
  "splits": {
    "train": ["img_000001", "img_000002", "img_000003"],
    "val": ["img_000101"],
    "test": ["img_000201"]
  },
  "images": {
    "img_000001": {
      "file_name": "images/img_000001.jpg",
      "split": "train",
      "width": 1280,
      "height": 720,
      "labels": {
        "mask": "masks/img_000001.png",
        "class": "classes/img_000001.json"
      }
    },
    "img_000002": {
      "file_name": "images/img_000002.jpg",
      "split": "train",
      "width": 1280,
      "height": 720,
      "labels": {
        "box": "boxes/img_000002.json",
        "point": "points/img_000002.json"
      }
    },
    "img_000003": {
      "file_name": "images/img_000003.jpg",
      "split": "train",
      "width": 1280,
      "height": 720,
      "labels": {
        "unlabeled": "unlabeled/img_000003.json"
      }
    }
  }
}
```

## Why this format works well

It makes three things easy:

1. Supervision-type sampling:
   the loader can select only `mask` samples, or blend `mask/box/point/...`.
2. Multi-task training:
   the same image can carry multiple annotation types.
3. Semi-supervised training:
   unlabeled images stay inside the same dataset and split logic.

## Recommended loader behavior

At training time, your dataset loader should:

1. read `splits.json`
2. choose image ids from `splits.train`
3. inspect `images[image_id]["labels"]`
4. dispatch to the correct parser based on available label types
5. optionally sample one supervision type if multiple are present

## Recommended priority when multiple labels exist

If you need a default priority:

1. `mask`
2. `scribble`
3. `box`
4. `point`
5. `class`
6. `unlabeled`

That keeps dense supervision preferred over sparse or weak supervision, while
still preserving all metadata in the manifest.
