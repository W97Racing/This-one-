"""Generate matched test profiles: same length and max thickness, different nose shape.

    rounded: elliptical thickness distribution (blunt-ish, rounded nose)
    pointed: (1-u^2)^0.7 thickness distribution (sharper nose and tail, no corners)

Writes CSVs in mm to samples/ when run directly. Used by the tests too.
"""

import csv
import os

import numpy as np

LENGTH = 0.40       # m
THICK_HALF = 0.05   # m, max half-thickness
HERE = os.path.dirname(os.path.abspath(__file__))


def half_thickness(kind: str, x: np.ndarray) -> np.ndarray:
    u = np.clip(2 * x / LENGTH, -1.0, 1.0)  # -1 front, +1 tail
    if kind == "rounded":
        return THICK_HALF * np.sqrt(np.maximum(0.0, 1 - u ** 2))
    if kind == "pointed":
        # Exponent 0.7 gives a sharper nose and tail than the ellipse but no
        # corners. A linear (diamond) shape has convex corners at max thickness
        # where inviscid Cp is singular, so it cannot be compared fairly.
        return THICK_HALF * np.maximum(0.0, 1 - u ** 2) ** 0.7
    raise ValueError(kind)


def profile(kind: str, n_half: int = 300) -> np.ndarray:
    # Cosine spacing clusters points near both ends where the curvature changes fastest
    th = np.linspace(0, np.pi, n_half)
    x = -LENGTH / 2 * np.cos(th)
    # Avoid exact zero-thickness tips; keep a tiny offset so the polygon stays closed
    h = half_thickness(kind, x)
    top = np.column_stack([x, h])
    bot = np.column_stack([x[::-1], -h[::-1]])
    pts = np.vstack([top, bot[1:-1]])
    return pts


def write_mm(path: str, pts_m: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_mm"])
        for x, y in pts_m:
            w.writerow([f"{x * 1000:.3f}", f"{y * 1000:.3f}"])


if __name__ == "__main__":
    out = os.path.join(HERE, "samples")
    os.makedirs(out, exist_ok=True)
    for kind in ("rounded", "pointed"):
        write_mm(os.path.join(out, f"{kind}_nose.csv"), profile(kind))
        print("wrote", os.path.join(out, f"{kind}_nose.csv"))
