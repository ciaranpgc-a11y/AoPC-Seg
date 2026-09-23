"""Minimal inference for AoPC-Seg v1.1 (2D phase-contrast CMR magnitude images).

Usage
-----
    python infer.py --vessel ascending --input frame.dcm --output mask.npy
    python infer.py --vessel descending --input frame.npy --spacing 1.40 1.40 --output mask.npy
    python infer.py --vessel ascending --input dicom_folder/ --output masks/

Input is one 2D magnitude frame (DICOM, or a NumPy array with --spacing in mm
as row, column). Output is a binary mask (uint8, 1 = vessel) on the input grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent


def load_model(vessel: str):
    import onnxruntime as ort
    profile = json.loads((HERE / "profiles" / f"{vessel}.json").read_text())
    session = ort.InferenceSession(str(HERE / profile["weights"]),
                                   providers=["CPUExecutionProvider"])
    return session, profile


def _preprocess(image, spacing_yx_mm, profile):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a single 2D frame (rows x columns).")
    ty, tx = profile["target_spacing_yx_mm"]
    zy, zx = float(spacing_yx_mm[0]) / ty, float(spacing_yx_mm[1]) / tx
    if abs(zy - 1.0) > 0.02 or abs(zx - 1.0) > 0.02:
        arr = zoom(arr, (zy, zx), order=1).astype(np.float32)
    finite = np.isfinite(arr)
    vals = arr[finite]
    mean = float(vals.mean()) if vals.size else 0.0
    std = float(vals.std()) if vals.size else 1.0
    std = std if std > 1e-8 else 1.0
    arr = (np.where(finite, arr, mean) - mean) / std
    return arr.astype(np.float32)


def _gaussian(ph, pw, sigma=0.25):
    yy = np.linspace(-1, 1, ph)[:, None]
    xx = np.linspace(-1, 1, pw)[None, :]
    return np.maximum(np.exp(-(yy**2 + xx**2) / (2 * sigma**2)), 1e-3).astype(np.float32)


def _logits(session, arr, profile):
    ph, pw = profile["patch_size_yx"]
    name_in = session.get_inputs()[0].name
    run = lambda p: session.run(None, {name_in: p[None, None].astype(np.float32)})[0][0]
    h, w = arr.shape
    if h <= ph and w <= pw:
        tile = np.zeros((ph, pw), np.float32)
        tile[:h, :w] = arr
        return run(tile)[:, :h, :w]
    ys = list(range(0, max(1, h - ph + 1), max(1, ph // 2))) or [0]
    xs = list(range(0, max(1, w - pw + 1), max(1, pw // 2))) or [0]
    if ys[-1] != max(0, h - ph):
        ys.append(max(0, h - ph))
    if xs[-1] != max(0, w - pw):
        xs.append(max(0, w - pw))
    gw = _gaussian(ph, pw)
    acc, wsum = None, np.zeros((h, w), np.float32)
    for y in ys:
        for x in xs:
            yh, xw = min(ph, h - y), min(pw, w - x)
            tile = np.zeros((ph, pw), np.float32)
            tile[:yh, :xw] = arr[y:y + yh, x:x + xw]
            out = run(tile)
            if acc is None:
                acc = np.zeros((out.shape[0], h, w), np.float32)
            acc[:, y:y + yh, x:x + xw] += out[:, :yh, :xw] * gw[:yh, :xw]
            wsum[y:y + yh, x:x + xw] += gw[:yh, :xw]
    return acc / np.maximum(wsum, 1e-6)[None]


def segment(session, profile, image, spacing_yx_mm):
    """Return a uint8 mask (1 = vessel) on the grid of `image`."""
    image = np.asarray(image)
    arr = _preprocess(image, spacing_yx_mm, profile)
    label = np.argmax(_logits(session, arr, profile), axis=0).astype(np.uint8)
    if label.shape != image.shape:
        label = zoom(label, (image.shape[0] / label.shape[0], image.shape[1] / label.shape[1]), order=0)
        out = np.zeros(image.shape, np.uint8)
        h, w = min(out.shape[0], label.shape[0]), min(out.shape[1], label.shape[1])
        out[:h, :w] = label[:h, :w]
        label = out
    return label


def read_dicom(path):
    import pydicom
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    img = img * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    return img, [float(v) for v in ds.PixelSpacing]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vessel", required=True, choices=["ascending", "descending"])
    ap.add_argument("--input", required=True, help=".dcm file, .npy file, or folder of .dcm files")
    ap.add_argument("--spacing", nargs=2, type=float, help="pixel spacing (row, column) in mm for .npy input")
    ap.add_argument("--output", required=True, help=".npy file, or folder when --input is a folder")
    a = ap.parse_args()
    session, profile = load_model(a.vessel)
    src = Path(a.input)
    if src.is_dir():
        out_dir = Path(a.output); out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.dcm")):
            img, sp = read_dicom(f)
            np.save(out_dir / f"{f.stem}_mask.npy", segment(session, profile, img, sp))
        return
    if src.suffix.lower() == ".npy":
        if not a.spacing:
            ap.error("--spacing is required for .npy input")
        img, sp = np.load(src), a.spacing
    else:
        img, sp = read_dicom(src)
    np.save(a.output, segment(session, profile, img, sp))


if __name__ == "__main__":
    main()
