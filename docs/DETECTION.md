# Object detection data

The detection feature is under development. The prepared dataset and weights are local artifacts and are not committed to Git. The configured raw and processed paths can be changed per machine.

## Sources and attribution

- [Football Players Detection v10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 312 original 1920 × 1080 frames.
- [Football Object Detection v1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 1,150 640 × 640 images with provider-generated crop variants. All 242 distinct frame identifiers in this export also occur in v10.

Both export class mappings are verified against their metadata. The output mapping is `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`. Goal polygons and corner boxes are removed. Images and boxes are validated, with sub-pixel edge overflows clipped only within a one-pixel tolerance. Conflicting labels on byte-identical images stop preparation.

## Split and limitations

SHA-256 identifies exact duplicates; a 64-bit DCT perceptual hash flags similar images. The filename prefix groups related sequences and crops across both exports. Some visually similar source prefixes are grouped conservatively in [the source configuration](../configs/detection_sources.json); cross-prefix perceptual-hash candidates are grouped as well. The grouping is a precaution, not proof of original match identity. Different prefixes can still come from the same match, so residual match leakage remains possible.

With seed 42, held-out groups are selected using the original-frame counts and class representation, before training. The 80/10/10 target applies to original frames: 251/31/30. The training set has 1,203 images after adding variants and removing two duplicates; validation and test use only originals. The third exact duplicate is part of the held-out augmentations that are excluded. There are only two source groups each in validation and test, and adjacent images may be highly correlated. These sets are therefore a limited estimate of performance on new matches.

The sample of annotated training images reviewed was generally plausible, but this is not a complete annotation-quality audit. At 640 input size, 270 of 272 original ball boxes are smaller than 8 × 8 pixels after aspect-preserving resize; this and class imbalance make ball detection difficult.

## Prepare and verify

In PowerShell, from the project root, change the paths to suit the local machine. The output directory must not already exist and must not be inside or contain the raw directory.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_detection_dataset.py `
  --raw E:\PitchLensStorage\datasets\raw `
  --output data\processed\detection-v1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The output contains YOLO image and label folders, `data.yaml`, a per-image `manifest.json` with source IDs and SHA-256 hashes, and `report.json` with grouping, class counts and audit results. Neither the raw data nor the processed data should be committed. Use a new output directory for each preparation instead of overwriting an earlier one.
