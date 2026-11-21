"""Checks for the digitizer and the panel solver. Run: python test_gearcase_cfd.py"""

import os
import tempfile

import cv2
import numpy as np

import digitize
import panel


def circle(n=200, r=0.1):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def test_circle_matches_analytic_cp():
    """Inviscid flow past a cylinder: Cp = 1 - 4 sin^2(theta), theta from the front stagnation point."""
    pts = circle()
    res = panel.analyse(pts, speed_mps=10.0, depth_m=1.0, flow_dir="+x")
    # Front stagnation point is at x = -r (flow comes from -x moving +x)
    theta = np.arctan2(res.y, res.x)          # angle of control point from centre
    # angle from the upstream stagnation point (pi), measured around the body
    phi = np.abs(np.angle(np.exp(1j * (theta - np.pi))))
    cp_exact = 1 - 4 * np.sin(phi) ** 2
    err = np.max(np.abs(res.cp - cp_exact))
    print(f"circle max |Cp error| = {err:.4f}")
    assert err < 0.02, err


def test_ccw_and_orientation():
    pts = circle()[::-1]  # clockwise input
    fixed = panel.ensure_closed_ccw(pts)
    x, y = fixed[:, 0], fixed[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area > 0


def test_rounded_nose_lower_suction_peak_than_pointed():
    """Sharper leading edge gives a higher suction peak at the same speed."""
    import make_samples
    rounded = make_samples.profile("rounded")
    pointed = make_samples.profile("pointed")
    v = 30.0
    cp_round = panel.analyse(panel.ensure_closed_ccw(rounded), v, 1.0).cp_min
    cp_point = panel.analyse(panel.ensure_closed_ccw(pointed), v, 1.0).cp_min
    print(f"Cp_min rounded = {cp_round:.3f}, pointed = {cp_point:.3f}")
    assert cp_point < cp_round


def test_digitizer_roundtrip():
    """Draw a known outline, digitize it, and compare extent in mm."""
    scale = 4.0  # px per mm
    w, h = 800, 400
    img = np.full((h, w, 3), 255, np.uint8)
    # Rectangle-ish body 120 mm long x 60 mm tall, drawn 100 px from edges
    x0, y0 = 100, 100
    length_mm, height_mm = 120.0, 60.0
    body = np.array([
        [x0, y0],
        [x0 + int(length_mm * scale), y0],
        [x0 + int(length_mm * scale) - 40, y0 + int(height_mm * scale)],
        [x0 + 40, y0 + int(height_mm * scale)],
    ], np.int32)
    cv2.fillPoly(img, [body], (30, 30, 30))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "synthetic.png")
    cv2.imwrite(path, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dark body on white background, so the body is the inverted foreground
    outline = digitize.extract_outline(gray, invert=True, min_area_frac=0.001)
    mm = digitize.to_mm(outline, outline.mean(axis=0), 1.0 / scale)
    extent_x = mm[:, 0].max() - mm[:, 0].min()
    extent_y = mm[:, 1].max() - mm[:, 1].min()
    print(f"digitized extent {extent_x:.1f} x {extent_y:.1f} mm (expected ~120 x 60)")
    assert abs(extent_x - length_mm) < 2.0
    assert abs(extent_y - height_mm) < 2.0


if __name__ == "__main__":
    test_circle_matches_analytic_cp()
    test_ccw_and_orientation()
    test_rounded_nose_lower_suction_peak_than_pointed()
    test_digitizer_roundtrip()
    print("all tests passed")
