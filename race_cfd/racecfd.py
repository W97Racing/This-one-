"""racecfd: flow-testing tool for developing race boat and gearcase characteristics.

Subcommands
-----------
sweep    Inviscid panel analysis of one section over a range of speeds and depths.
         Reports suction peak, cavitation onset, and friction estimate per speed.
compare  Same sweep for several sections (e.g. rounded vs pointed nose) side by side.
rans     Full 2D RANS (k-omega SST, OpenFOAM) of one section at one speed.

Every run writes files to an output folder: CSV tables, PNG plots, and a
Markdown report, so the results can be uploaded or shared.

Scope and limits (see README.md): 2D infinite-span sections, no free surface, no
propeller, no ventilation. Use the numbers to compare shapes, not as absolute drag.

Examples
--------
python racecfd.py sweep ../gearcase_cfd/samples/rounded_nose.csv \
    --speeds 40 60 80 100 --depths 50 150 --out out/rounded
python racecfd.py compare ../gearcase_cfd/samples/rounded_nose.csv \
    ../gearcase_cfd/samples/pointed_nose.csv --speed 100 --depth 100 --out out/nose
python racecfd.py rans ../gearcase_cfd/gearcase_side/gearcase_side_smooth.csv --speed 5 --out out/gearcase_rans
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.abspath(os.path.join(HERE, "..", "gearcase_cfd"))
sys.path.insert(0, GC)

import panel  # noqa: E402  (from gearcase_cfd)

MPH = 0.44704


def load(path: str) -> np.ndarray:
    return panel.load_profile(path)


def run_sweep(pts: np.ndarray, speeds_mph: list[float], depths_mm: list[float], flow: str):
    rows = []
    for d in depths_mm:
        for v in speeds_mph:
            r = panel.analyse(pts, v * MPH, d / 1000.0, flow)
            rows.append({
                "speed_mph": v, "depth_mm": d, "min_cp": r.cp_min,
                "x_min_cp_mm": r.x_cp_min * 1000, "sigma": r.sigma,
                "cavitation": "YES" if r.cavitation_risk else "no",
                "friction_N_per_m_span": r.friction_per_span,
            })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_sweep(path: str, rows: list[dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in sorted({r["depth_mm"] for r in rows}):
        rr = [r for r in rows if r["depth_mm"] == d]
        sp = [r["speed_mph"] for r in rr]
        ax.plot(sp, [-r["min_cp"] for r in rr], "-o", label=f"-Cp_min, depth {d:.0f} mm")
        ax.plot(sp, [r["sigma"] for r in rr], "--", label=f"sigma, depth {d:.0f} mm")
    ax.set_xlabel("speed (mph)")
    ax.set_ylabel("suction peak -Cp_min vs cavitation number sigma")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path: str, title: str, rows: list[dict], notes: list[str]) -> None:
    lines = [f"# {title}", "", "| speed (mph) | depth (mm) | min Cp | sigma | cavitation | friction (N/m span) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['speed_mph']:.0f} | {r['depth_mm']:.0f} | {r['min_cp']:.3f} | {r['sigma']:.3f} | "
                     f"{r['cavitation']} | {r['friction_N_per_m_span']:.0f} |")
    lines += ["", "## Notes"] + [f"- {n}" for n in notes]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_sweep(a) -> None:
    pts = load(a.profile)
    os.makedirs(a.out, exist_ok=True)
    rows = run_sweep(pts, a.speeds, a.depths, a.flow)
    write_csv(os.path.join(a.out, "sweep.csv"), rows)
    plot_sweep(os.path.join(a.out, "sweep.png"), rows, os.path.basename(a.profile))
    write_report(os.path.join(a.out, "report.md"), f"Sweep: {os.path.basename(a.profile)}", rows, NOTES)
    print(f"wrote {a.out}/sweep.csv, sweep.png, report.md")


def cmd_compare(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    all_rows = []
    for prof in a.profiles:
        r = run_sweep(load(prof), [a.speed], [a.depth], a.flow)[0]
        r["profile"] = os.path.basename(prof)
        all_rows.append(r)
    cols = ["profile"] + [k for k in all_rows[0] if k != "profile"]
    with open(os.path.join(a.out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    write_report(os.path.join(a.out, "report.md"), f"Compare at {a.speed} mph, {a.depth} mm",
                 all_rows, NOTES)
    print(f"wrote {a.out}/compare.csv and report.md")


def cmd_rans(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    rans = os.path.join(GC, "rans2d.py")
    subprocess.run([sys.executable, rans, a.profile, "--speed", str(a.speed), "--out", a.out], check=True)
    print(f"RANS outputs in {a.out} (cp_rans.csv, log.simpleFoam, VTK/)")


NOTES = [
    "Inviscid 2D source-panel pressure; zero drag by itself. Friction is a turbulent flat-plate estimate.",
    "Cavitation screen: -Cp_min > sigma, with sigma = (p0 - pv)/(0.5 rho V^2). Screening rule only.",
    "No free surface, ventilation, propeller, or leg/skeg effects.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep")
    s.add_argument("profile")
    s.add_argument("--speeds", type=float, nargs="+", default=[40, 60, 80, 100])
    s.add_argument("--depths", type=float, nargs="+", default=[50.0])
    s.add_argument("--flow", choices=["+x", "-x"], default="+x")
    s.add_argument("--out", default="out/sweep")
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("compare")
    c.add_argument("profiles", nargs="+")
    c.add_argument("--speed", type=float, default=100.0)
    c.add_argument("--depth", type=float, default=50.0)
    c.add_argument("--flow", choices=["+x", "-x"], default="+x")
    c.add_argument("--out", default="out/compare")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("rans")
    r.add_argument("profile")
    r.add_argument("--speed", type=float, default=5.0, help="m/s")
    r.add_argument("--out", default="out/rans")
    r.set_defaults(func=cmd_rans)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
