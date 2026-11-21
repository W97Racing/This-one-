"""2D inviscid source-panel flow solver with cavitation and friction estimates.

Solves potential flow past a closed 2D profile (constant-strength source panels,
Hess-Smith style, zero circulation) for a freestream of speed V. Output is the
surface pressure coefficient Cp along the profile, the minimum Cp (where
cavitation starts), and a cavitation index for the operating depth.

What this is:
    * A fast way to see where suction peaks sit on a nose or gearcase section
      and how that changes when the nose is rounded or pointed.
    * A cavitation-onset check: cavitation starts where -Cp_min exceeds the
      cavitation number sigma = (p0 - pv) / (0.5 rho V^2).

What this is NOT:
    * Not a 3D CFD solution. A side-view silhouette treated as a 2D section is
      an infinite-span approximation. A gearcase nose is closer to a body of
      revolution about the prop shaft, which needs an axisymmetric solver.
    * No free surface, no propeller, no ventilation physics.
    * Inviscid pressure has zero drag (d'Alembert). The friction term below is
      a separate turbulent flat-plate estimate with a form factor, per unit span.

Usage:
    python panel.py profile.csv --speed-mph 100 --depth-mm 100
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

RHO = 1000.0          # kg/m^3 fresh water
NU = 1.0e-6           # m^2/s
G = 9.81
P_ATM = 101_325.0     # Pa
P_VAPOR = 2_339.0     # Pa at 20 C


@dataclass
class Result:
    x: np.ndarray          # panel control points (m)
    y: np.ndarray
    cp: np.ndarray         # surface pressure coefficient at control points
    cp_min: float
    x_cp_min: float        # location of minimum Cp (m)
    sigma: float           # cavitation number at operating depth
    cavitation_risk: bool  # True if -cp_min > sigma
    perimeter: float       # m per unit span
    chord: float           # profile length in flow direction (m)
    friction_per_span: float  # N per m of span (turbulent flat-plate estimate)


def load_profile(path: str) -> np.ndarray:
    """Load x_mm,y_mm CSV and return (N,2) in metres, closed and CCW."""
    with open(path) as f:
        rdr = csv.reader(f)
        next(rdr)
        pts = np.array([[float(a), float(b)] for a, b in rdr]) / 1000.0
    return ensure_closed_ccw(pts)


def ensure_closed_ccw(pts: np.ndarray) -> np.ndarray:
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    # Shoelace area; reverse to counterclockwise if needed
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts = pts[::-1]
    return pts


def solve_source_panels(pts: np.ndarray, v_inf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return control points, tangential surface speed, and Cp for freestream v_inf."""
    n = len(pts)
    p1 = pts
    p2 = np.roll(pts, -1, axis=0)
    d = p2 - p1
    s = np.linalg.norm(d, axis=1)
    phi = np.arctan2(d[:, 1], d[:, 0])
    cp_pts = 0.5 * (p1 + p2)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    # CCW traversal: outward normal is to the right of the direction
    nrm = np.column_stack([sin_p, -cos_p])
    tan = np.column_stack([cos_p, sin_p])

    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        x = cp_pts[i, 0]
        y = cp_pts[i, 1]
        # Panel j frame for every control point i (vectorised over j)
        dx = x - p1[:, 0]
        dy = y - p1[:, 1]
        xl = dx * cos_p + dy * sin_p
        yl = -dx * sin_p + dy * cos_p
        r1 = xl ** 2 + yl ** 2
        r2 = (xl - s) ** 2 + yl ** 2
        ul = np.log(r1 / np.maximum(r2, 1e-30)) / (4 * np.pi)
        vl = (np.arctan2(yl, xl - s) - np.arctan2(yl, xl)) / (2 * np.pi)
        # Global velocity from unit-strength source on each panel
        u = ul * cos_p - vl * sin_p
        v = ul * sin_p + vl * cos_p
        A[i, :] = u * nrm[i, 0] + v * nrm[i, 1]
        B[i, :] = u * tan[i, 0] + v * tan[i, 1]
        A[i, i] = 0.5          # outward-side self-influence of own source sheet
        B[i, i] = 0.0          # no tangential self-influence

    rhs = -(nrm @ v_inf)
    sigma = np.linalg.solve(A, rhs)
    vt = tan @ v_inf + B @ sigma
    v_mag = np.abs(vt)
    speed = np.linalg.norm(v_inf)
    cp = 1.0 - (v_mag / speed) ** 2
    return cp_pts, v_mag, cp


def friction_drag_per_span(perimeter: float, chord: float, speed: float, thickness_ratio: float) -> float:
    """Turbulent flat-plate skin friction with a 2D form-factor, per unit span."""
    re = speed * chord / NU
    cf = 0.455 / (np.log10(re) ** 2.58)   # Schlichting, turbulent
    ff = 1.0 + 2.0 * thickness_ratio + 60.0 * thickness_ratio ** 4
    return 0.5 * RHO * speed ** 2 * perimeter * cf * ff


def analyse(pts_m: np.ndarray, speed_mps: float, depth_m: float, flow_dir: str = "+x") -> Result:
    pts_m = ensure_closed_ccw(np.asarray(pts_m, dtype=float))  # solver needs CCW; always enforce
    direction = np.array([1.0, 0.0]) if flow_dir == "+x" else np.array([-1.0, 0.0])
    v_inf = speed_mps * direction
    cp_pts, _, cp = solve_source_panels(pts_m, v_inf)

    i_min = int(np.argmin(cp))
    cp_min = float(cp[i_min])
    x_cp_min = float(cp_pts[i_min, 0])

    p0 = P_ATM + RHO * G * depth_m
    q = 0.5 * RHO * speed_mps ** 2
    sigma = (p0 - P_VAPOR) / q
    cavitation_risk = -cp_min > sigma

    seg = np.roll(pts_m, -1, axis=0) - pts_m
    perimeter = float(np.sum(np.linalg.norm(seg, axis=1)))
    along = pts_m @ direction
    chord = float(along.max() - along.min())
    thickness = float((pts_m[:, 1].max() - pts_m[:, 1].min()) / max(chord, 1e-9))
    fric = friction_drag_per_span(perimeter, chord, speed_mps, thickness)

    return Result(cp_pts[:, 0], cp_pts[:, 1], cp, cp_min, x_cp_min, sigma,
                  cavitation_risk, perimeter, chord, fric)


def report(res: Result, speed_mph: float, depth_mm: float) -> str:
    lines = [
        f"speed {speed_mph:.0f} mph, depth {depth_mm:.0f} mm",
        f"chord (flow direction): {res.chord*1000:.1f} mm, perimeter: {res.perimeter*1000:.1f} mm",
        f"min Cp: {res.cp_min:.3f} at x = {res.x_cp_min*1000:.1f} mm",
        f"cavitation number sigma: {res.sigma:.3f}",
        f"cavitation onset predicted: {'YES' if res.cavitation_risk else 'no'} (needs -Cp_min > sigma)",
        f"friction drag (estimate): {res.friction_per_span:.1f} N per m of span",
        "NOTE: 2D infinite-span section; ventilation and prop effects not modelled.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="CSV with x_mm,y_mm from digitize.py")
    ap.add_argument("--speed-mph", type=float, default=100.0)
    ap.add_argument("--depth-mm", type=float, default=0.0,
                    help="depth of the section below the free surface (0 = surface-running)")
    ap.add_argument("--flow", choices=["+x", "-x"], default="+x",
                    help="direction the water moves past the body in profile coordinates")
    ap.add_argument("--plot", help="optional PNG of Cp vs x")
    args = ap.parse_args()

    pts = load_profile(args.profile)
    speed = args.speed_mph * 0.44704
    res = analyse(pts, speed, args.depth_mm / 1000.0, args.flow)
    print(report(res, args.speed_mph, args.depth_mm))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(res.x * 1000, -res.cp, "-o", ms=2)
        ax.axhline(res.sigma, color="r", ls="--", label="-sigma (cavitation threshold)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("-Cp")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
