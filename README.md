# AoPC-Seg v1.1

Automated segmentation of the **ascending aorta** and **descending aorta** in
two-dimensional phase-contrast cardiovascular magnetic resonance (2D-PC CMR)
**magnitude** images.

AoPC-Seg consists of two single-vessel models, one per vessel, distributed as
ONNX files that run with ONNX Runtime. The reference script uses the CPU
execution provider; compatible GPU providers can be selected in custom code.

> **Research use only.** AoPC-Seg is not a medical device and has not been
> approved for clinical use. Outputs must be reviewed by a qualified person.

## Contents

| Path | Description |
|---|---|
| `weights/ascending.onnx` | Ascending-aorta model |
| `weights/descending.onnx` | Descending-aorta model |
| `profiles/*.json` | Preprocessing and input/output specification for each model |
| `infer.py` | Minimal reference inference script |
| `requirements.txt` | Python dependencies |
| `SHA256SUMS` | Checksums for the release files |
| `.gitattributes` | Git LFS tracking for the ONNX weights |
| `LICENSE`, `NOTICE` | Apache License 2.0 and attribution |

## Quick start

```bash
git lfs install
git clone https://github.com/ciaranpgc-a11y/AoPC-Seg.git
cd AoPC-Seg
pip install -r requirements.txt
python infer.py --vessel ascending  --input frame.dcm   --output aao_mask.npy
python infer.py --vessel descending --input dicom_dir/  --output dao_masks/
python infer.py --vessel ascending  --input frame.npy --spacing 1.40 1.40 --output mask.npy
```

The two ONNX weights are stored with Git LFS. Clone with Git LFS installed so
that `weights/` contains the model files rather than pointer files.

Each call segments one 2D frame (or every `.dcm` file in a folder, frame by
frame) and saves a binary mask (`uint8`, 1 = vessel) on the input image grid.

From Python:

```python
import infer
session, profile = infer.load_model("ascending")
mask = infer.segment(session, profile, image_2d, spacing_yx_mm=(1.4, 1.4))
```

## Input requirements

- A single 2D **magnitude** frame (not the phase/velocity image).
- DICOM input: `RescaleSlope` and `RescaleIntercept` are applied, and
  `PixelSpacing` is read automatically.
- NumPy input: supply the pixel spacing in mm as (row, column).
- **Ascending aorta:** acquisitions planned orthogonal to the ascending aorta,
  or transverse/near-axial planes through the ascending aorta.
- **Descending aorta:** transverse or near-axial planes only. The descending
  aorta is not intersected in cross-section in planes orthogonal to the
  ascending aorta, so this model should not be applied to them.

## Preprocessing (implemented in `infer.py`)

1. Resample to 1.25 x 1.25 mm (bilinear), if the source spacing differs by
   more than 2%.
2. Z-score normalise each image using its own mean and standard deviation.
3. Pad to the fixed model input size (ascending 224 x 256; descending
   192 x 256), or, for larger images, use a sliding window with 50% overlap
   and Gaussian-weighted averaging of logits.
4. Take the arg-max over the two output channels (0 = background, 1 = vessel)
   and map the label back to the source grid (nearest neighbour).

The ONNX models take `image` of shape `(1, 1, H, W)` (float32) and return
`logits` of shape `(1, 2, H, W)`.

## Citation

If you use these model weights or the inference code, please cite the release:

> Grafton-Clarke C, Garg P. AoPC-Seg: aortic 2D-PC CMR segmentation models
> and inference code. Version 1.1. GitHub, 2026.
> https://github.com/ciaranpgc-a11y/AoPC-Seg

## Licence

The model weights and code are released under the Apache License 2.0
(see `LICENSE` and `NOTICE`). Training images and datasets are not included.
