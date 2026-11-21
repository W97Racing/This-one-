"""Axisymmetric (body of revolution) source-panel solver for a gearcase nose.

A gearcase nose seen from the side is close to the meridian profile of a body of
revolution about the propeller shaft. This module takes that meridian, builds the
surface from ring-shaped constant-strength source panels, and solves for the
surface pressure at zero incidence (no circulation).

Coordinates: z is along the shaft axis (flow goes in +z, nose at the smallest z),
r is radial distance from the axis.

What it gives:
    * Cp along the meridian, and minimum Cp (suction peak).
    * Cavitation-onset screen: -Cp_min > sigma, sigma = (p0 - pv)/(0.5 rho U^2).
    * Total friction drag: turbulent flat-plate Cf times wetted area times the
      Hoerner body-of-revolution form factor 1 + 1.5(d/l)^1.5 + 7(d/l)^3.

What it does NOT give:
    * Free surface, ventilation, propeller, or the lower-unit leg and skeg.
    * Viscous separation. Inviscid Cp is fine for the attached-flow nose region.

STATUS: NOT YET VALIDATED. The sphere check (Cp = 1 - 2.25 sin^2 theta) gives
3-15% max Cp error, and the error does not fall cleanly with panel count. The
self-panel quadrature needs work before this is used for numbers.

Usage:
    python axisym.py meridian.csv --speed-mph 100 --depth-mm 100
    python axisym.py outline.csv --from-outline --speed-mph 100 --depth-mm 100
    (--from-outline reads the x_mm,y_mm file from digitize.py, takes the upper
     half of the silhouette as r about the shaft line, and uses x as z.)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

RHO = 1000.0
NU = 1.0e-6
G = 9.81
P_ATM = 101_325.0
P_VAPOR = 2_339.0

N_GAUSS = 6        # Gauss points per panel
N_AZIM = 48        # azimuthal points on each ring


@dataclass
class AxiResult:
    z: np.ndarray            # control points along meridian (m)
    r: np.ndarray
    cp: np.ndarray
    cp_min: float
    z_cp_min: float
    sigma: float
    cavitation_risk: bool
    wetted_area: float       # m^2
    length: float            # m
    diameter: float          # m
    drag_n: float            # N, friction estimate


def cosine_meridian(length: float, max_radius: float, n: int, shape) -> np.ndarray:
    """Build a meridian with cosine spacing from z=-L/2 to +L/2. shape(u) gives r for u in [-1,1]."""
    th = np.linspace(0, np.pi, n + 1)
    z = -length / 2 * np.cos(th)
    u = np.clip(2 * z / length, -1.0, 1.0)
    r = max_radius * shape(u)
    r[0] = 0.0
    r[-1] = 0.0
    return np.column_stack([z, r])


def meridian_from_outline(path: str, axis_y_mm: float | None = None) -> np.ndarray:
    """Take a digitize.py outline (x_mm,y_mm), keep the upper half about the axis, return (z,r) in m."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    if axis_y_mm is None:
        axis_y_mm = 0.5 * (y.max() + y.min())
    keep = y >= axis_y_mm
    z = x[keep]
    r = y[keep] - axis_y_mm
    order = np.argsort(z)
    z, r = z[order], r[order]
    # Collapse duplicate z values by keeping the largest radius (the silhouette edge)
    zu = np.unique(np.round(z, 3))
    ru = np.array([r[np.round(z, 3) == zv].max() for zv in zu])
    # Close on the axis at both ends
    zu = np.r_[zu[0], zu, zu[-1]]
    ru = np.r_[0.0, ru, 0.0]
    return np.column_stack([zu, ru]) / 1000.0


def solve(meridian: np.ndarray, speed_mps: float, depth_m: float) -> AxiResult:
    """Solve the axisymmetric source-panel problem for a meridian (z,r) in metres."""
    z = meridian[:, 0]
    r = meridian[:, 1]
    dz = np.diff(z)
    dr = np.diff(r)
    ds = np.hypot(dz, dr)
    tz = dz / np.maximum(ds, 1e-30)
    tr = dr / np.maximum(ds, 1e-30)
    nz = -tr                    # outward normal in (z, r) for a meridian running nose -> tail
    nr = tz
    zc = 0.5 * (z[1:] + z[:-1])
    rc = 0.5 * (r[1:] + r[:-1])
    n = len(ds)

    # Gauss points along each panel, azimuth around each ring
    xg, wg = np.polynomial.legendre.leggauss(N_GAUSS)
    frac = 0.5 * (xg + 1.0)
    wfrac = 0.5 * wg
    zq = z[:-1, None] + frac[None, :] * dz[:, None]          # (n, G)
    rq = r[:-1, None] + frac[None, :] * dr[:, None]
    warc = wfrac[None, :] * ds[:, None]                       # arc-length weights (n, G)
    theta = np.arange(N_AZIM) * 2 * np.pi / N_AZIM
    dtheta = 2 * np.pi / N_AZIM
    # Area weight per (panel, gauss, azimuth) point, excluding the singular self panel below
    Wq = warc * rq * dtheta                                   # (n, G)
    Qx = rq[:, :, None] * np.cos(theta)[None, None, :]        # (n, G, M)
    Qy = rq[:, :, None] * np.sin(theta)[None, None, :]
    Qz = np.broadcast_to(zq[:, :, None], Qx.shape)

    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i in range(n):
        dz_i = zc[i] - Qz
        dx_i = rc[i] - Qx
        dy_i = 0.0 - Qy
        R3 = (dz_i ** 2 + dx_i ** 2 + dy_i ** 2) ** 1.5
        f = Wq[:, :, None] / (4 * np.pi * R3)
        vz = dz_i * f
        vx = dx_i * f
        an = (vz * nz[i] + vx * nr[i]).sum(axis=(1, 2))
        bt = (vz * tz[i] + vx * tr[i]).sum(axis=(1, 2))
        A[i, :] = an
        B[i, :] = bt
        A[i, i] = 0.5          # outward-side self term of the source sheet
        B[i, i] = 0.0          # principal-value tangential self term is zero

    U = speed_mps
    sigma = np.linalg.solve(A, -nz * U)
    q = np.abs(tz * U + B @ sigma)
    cp = 1.0 - (q / U) ** 2

    i_min = int(np.argmin(cp))
    cp_min = float(cp[i_min])
    z_cp_min = float(zc[i_min])

    p0 = P_ATM + RHO * G * depth_m
    sig = (p0 - P_VAPOR) / (0.5 * RHO * U ** 2)
    cav = -cp_min > sig

    wetted = float(np.sum(2 * np.pi * rc * ds))
    length = float(z.max() - z.min())
    diameter = float(2 * r.max())
    re = U * length / NU
    cf = 0.455 / (np.log10(re) ** 2.58)
    ff = 1.0 + 1.5 * (diameter / length) ** 1.5 + 7.0 * (diameter / length) ** 3
    drag = 0.5 * RHO * U ** 2 * wetted * cf * ff

    return AxiResult(zc, rc, cp, cp_min, z_cp_min, sig, cav, wetted, length, diameter, drag)


def report(res: AxiResult, speed_mph: float, depth_mm: float) -> str:
    return "\n".join([
        f"speed {speed_mph:.0f} mph, depth {depth_mm:.0f} mm",
        f"length {res.length*1000:.1f} mm, max diameter {res.diameter*1000:.1f} mm",
        f"wetted area {res.wetted_area*1e4:.1f} cm^2",
        f"min Cp: {res.cp_min:.3f} at z = {res.z_cp_min*1000:.1f} mm",
        f"cavitation number sigma: {res.sigma:.3f}",
        f"cavitation onset predicted: {'YES' if res.cavitation_risk else 'no'} (needs -Cp_min > sigma)",
        f"friction drag estimate: {res.drag_n:.1f} N",
        "NOTE: no free surface, ventilation, propeller, or leg/skeg modelled.",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="meridian CSV with z_mm,r_mm, or an outline from digitize.py with --from-outline")
    ap.add_argument("--from-outline", action="store_true")
    ap.add_argument("--axis-y-mm", type=float, default=None, help="shaft line in the outline's y coordinates")
    ap.add_argument("--speed-mph", type=float, default=100.0)
    ap.add_argument("--depth-mm", type=float, default=0.0)
    args = ap.parse_args()

    if args.from_outline:
        mer = meridian_from_outline(args.csv, args.axis_y_mm)
    else:
        data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
        mer = data[:, :2] / 1000.0
    res = solve(mer, args.speed_mph * 0.44704, args.depth_mm / 1000.0)
    print(report(res, args.speed_mph, args.depth_mm))


if __name__ == "__main__":
    main()
