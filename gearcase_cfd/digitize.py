"""Digitize a side-view photo of a gearcase (or hull section) into a 2D profile.

Usage:
    python digitize.py photo.jpg --scale-px 1200 --scale-mm 500 --out profile.csv

--scale-px / --scale-mm: a known length in the photo (e.g. the bottom of a
gearcase to the top of the anti-cavitation plate, measured with a tape). The
pixel distance between two clicked/known points and the real distance give
mm per pixel.

If the automatic edge detection is poor, pass --points-csv with hand-picked
x,y pixel coordinates tracing the outline in order; they are used directly.

Outputs:
    profile.csv        x_mm, y_mm  (origin = first calibration point, y up)
    profile_overlay.png  the extracted outline drawn on the photo for checking

Limitations (read before trusting numbers):
    * The camera must be square to the gearcase. Perspective skews lengths.
    * Shadows and water on the surface can fool thresholding; check the overlay.
    * The profile is one side silhouette, so the 2D model assumes the section
      is symmetric about the centerline.
"""

from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np


def load_gray(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_outline(gray: np.ndarray, invert: bool, min_area_frac: float) -> np.ndarray:
    """Return the largest external contour as an (N, 2) array of pixel coords."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flags = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, flags + cv2.THRESH_OTSU)
    # Close small gaps (water streaks, glare) then fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SystemExit("no contour found; try --invert or --points-csv")
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area_frac * gray.size:
        raise SystemExit("largest contour is too small; check threshold or photo framing")
    # Light smoothing to remove pixel stair-steps, keep geometry
    pts = best[:, 0, :].astype(float)
    k = 5
    kernel1d = np.ones(k) / k
    xs = np.convolve(np.r_[pts[-k:, 0], pts[:, 0], pts[:k, 0]], kernel1d, "same")[k:-k]
    ys = np.convolve(np.r_[pts[-k:, 1], pts[:, 1], pts[:k, 1]], kernel1d, "same")[k:-k]
    return np.column_stack([xs, ys])


def to_mm(px_pts: np.ndarray, origin_px: np.ndarray, mm_per_px: float) -> np.ndarray:
    d = (px_pts - origin_px) * mm_per_px
    d[:, 1] *= -1.0  # image y points down; flip so y is up
    return d


def write_csv(path: str, pts_mm: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_mm"])
        for x, y in pts_mm:
            w.writerow([f"{x:.3f}", f"{y:.3f}"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--scale-px", type=float, required=True, help="known length in pixels")
    ap.add_argument("--scale-mm", type=float, required=True, help="same length in mm")
    ap.add_argument("--origin-px", type=float, nargs=2, default=None,
                    metavar=("X", "Y"), help="pixel origin (default: centroid of outline)")
    ap.add_argument("--invert", action="store_true", help="gearcase is darker than background")
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--points-csv", help="hand-traced pixel outline (x,y per row) instead of auto")
    ap.add_argument("--out", default="profile.csv")
    args = ap.parse_args()

    img, gray = load_gray(args.image)
    mm_per_px = args.scale_mm / args.scale_px

    if args.points_csv:
        data = np.loadtxt(args.points_csv, delimiter=",", skiprows=1)
        outline = data[:, :2]
    else:
        outline = extract_outline(gray, args.invert, args.min_area_frac)

    origin = np.array(args.origin_px) if args.origin_px else outline.mean(axis=0)
    pts_mm = to_mm(outline, origin, mm_per_px)
    write_csv(args.out, pts_mm)

    overlay = img.copy()
    cv2.polylines(overlay, [outline.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.circle(overlay, tuple(int(v) for v in origin), 6, (0, 255, 0), -1)
    overlay_path = args.out.rsplit(".", 1)[0] + "_overlay.png"
    cv2.imwrite(overlay_path, overlay)

    span_x = pts_mm[:, 0].max() - pts_mm[:, 0].min()
    span_y = pts_mm[:, 1].max() - pts_mm[:, 1].min()
    print(f"wrote {len(pts_mm)} points to {args.out}")
    print(f"overlay: {overlay_path}  (check it before using the numbers)")
    print(f"extent: {span_x:.1f} mm x {span_y:.1f} mm, mm/px = {mm_per_px:.4f}")


if __name__ == "__main__":
    main()
