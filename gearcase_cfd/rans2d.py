"""2D RANS (k-omega SST, OpenFOAM simpleFoam) of a gearcase section.

Pipeline:
    1. gmsh meshes the profile (x_mm,y_mm CSV) inside a rectangular far field,
       extruded one cell thick so OpenFOAM sees a 2D case (empty front/back).
    2. gmshToFoam converts the mesh; this script edits the patch types and writes
       the OpenFOAM case dictionaries.
    3. simpleFoam solves steady incompressible RANS.
    4. foamToVTK exports the body patch; Cp = p / (0.5 U^2) is computed and written.

Assumptions (read before trusting numbers):
    * Infinite-span 2D section. Side silhouette != cross-section (see README).
    * Wall functions on the body (nutkWallFunction). Check y+ in the output; values
      far above ~300 mean the first cell is too coarse for the wall model.
    * Steady RANS, attached flow. No free surface, no propeller, no ventilation.
    * Inlet turbulence intensity 1% and length scale 10% of chord (assumed).

Usage:
    python rans2d.py samples/rounded_nose.csv --speed 5 --out runs/rounded
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import textwrap

import numpy as np

OF_ENV = "export WM_PROJECT_DIR=/usr/share/openfoam; . /usr/share/openfoam/etc/bashrc >/dev/null 2>&1"
RHO = 1000.0
NU = 1.0e-6
CMU = 0.09


def run_of(cmd: str, cwd: str, log: str) -> None:
    with open(log, "w") as lf:
        r = subprocess.run(["bash", "-c", f"{OF_ENV}; {cmd}"], cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {cmd} (see {log})")


def read_profile_m(path: str) -> np.ndarray:
    with open(path) as f:
        rdr = csv.reader(f)
        next(rdr)
        pts = np.array([[float(a), float(b)] for a, b in rdr]) / 1000.0
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x, y = pts[:, 0], pts[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pts = pts[::-1]
    return pts


def build_mesh(pts: np.ndarray, msh_path: str, body_size: float, far_factor: float, thickness: float) -> dict:
    import gmsh

    chord = pts[:, 0].max() - pts[:, 0].min()
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    yc = 0.5 * (ymin + ymax)
    xa = xmin - 5 * chord
    xb = xmax + 10 * chord
    ya = yc - 5 * chord
    yb = yc + 5 * chord

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.add("gearcase")
    geo = gmsh.model.geo

    pids = [geo.addPoint(float(x), float(y), 0.0, body_size) for x, y in pts]
    body = geo.addSpline(pids + [pids[0]])
    body_loop = geo.addCurveLoop([body])

    p1 = geo.addPoint(xa, ya, 0, far_factor * chord)
    p2 = geo.addPoint(xb, ya, 0, far_factor * chord)
    p3 = geo.addPoint(xb, yb, 0, far_factor * chord)
    p4 = geo.addPoint(xa, yb, 0, far_factor * chord)
    l_bot = geo.addLine(p1, p2)
    l_out = geo.addLine(p2, p3)
    l_top = geo.addLine(p3, p4)
    l_in = geo.addLine(p4, p1)
    outer = geo.addCurveLoop([l_bot, l_out, l_top, l_in])
    surf = geo.addPlaneSurface([outer, body_loop])
    geo.synchronize()

    # Refine toward the body
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [body])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 400)
    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", body_size)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", far_factor * chord)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.02 * chord)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", 0.8 * chord)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, thickness, [1], [], True)
    geo.synchronize()

    groups = {"inlet": [], "outlet": [], "top": [], "bottom": [], "body": [], "frontAndBack": []}
    eps = 1e-9 + 1e-6 * chord
    for dim, tag in ext:
        if dim == 2:
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
            if abs(z1 - z0) < eps:
                groups["frontAndBack"].append(tag)
            elif abs(x0 - xa) < eps and abs(x1 - xa) < eps:
                groups["inlet"].append(tag)
            elif abs(x0 - xb) < eps and abs(x1 - xb) < eps:
                groups["outlet"].append(tag)
            elif abs(y0 - ya) < eps and abs(y1 - ya) < eps:
                groups["bottom"].append(tag)
            elif abs(y0 - yb) < eps and abs(y1 - yb) < eps:
                groups["top"].append(tag)
            else:
                groups["body"].append(tag)
        elif dim == 3:
            vol = tag
    # The base surface at z=0 is also front/back
    groups["frontAndBack"].append(surf)
    for k, tags in groups.items():
        pg = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, pg, k)
    pv = gmsh.model.addPhysicalGroup(3, [vol])
    gmsh.model.setPhysicalName(3, pv, "fluid")

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()
    return {"chord": chord, "xa": xa, "xb": xb, "ya": ya, "yb": yb}


def write_dict(path: str, header_class: str, obj: str, body: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = textwrap.dedent(f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {header_class};
            object      {obj};
        }}
        """)
    with open(path, "w") as f:
        f.write(hdr + "\n" + body)


def fix_boundary(case: str) -> None:
    """Set patch types: frontAndBack empty, body wall, everything else patch."""
    path = os.path.join(case, "constant", "polyMesh", "boundary")
    text = open(path).read()
    for name, typ in (("frontAndBack", "empty"), ("body", "wall")):
        text = re.sub(rf"({re.escape(name)}\s*\{{[^}}]*?type\s+)\w+(;)", rf"\g<1>{typ}\2", text, flags=re.S)
    open(path, "w").write(text)


def write_case(case: str, U: float, chord: float, length_scale: float) -> None:
    k_in = 1.5 * (0.01 * U) ** 2
    omega_in = np.sqrt(k_in) / (CMU ** 0.25 * length_scale)
    write_dict(f"{case}/constant/transportProperties", "dictionary", "transportProperties",
               f"transportModel  Newtonian;\nnu  [0 2 -1 0 0 0 0]  {NU};\n")
    write_dict(f"{case}/constant/turbulenceProperties", "dictionary", "turbulenceProperties",
               "simulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n")
    write_dict(f"{case}/system/controlDict", "dictionary", "controlDict", textwrap.dedent(f"""\
        application     simpleFoam;
        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         4000;
        deltaT          1;
        writeControl    timeStep;
        writeInterval   4000;
        purgeWrite      0;
        writeFormat     ascii;
        writePrecision  8;
        writeCompression off;
        timeFormat      general;
        timePrecision   6;
        runTimeModifiable true;
        """))
    write_dict(f"{case}/system/fvSchemes", "dictionary", "fvSchemes", textwrap.dedent("""\
        ddtSchemes      { default steadyState; }
        gradSchemes     { default Gauss linear; }
        divSchemes
        {
            default         none;
            div(phi,U)      bounded Gauss linearUpwind grad(U);
            div(phi,k)      bounded Gauss upwind;
            div(phi,omega)  bounded Gauss upwind;
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
        }
        laplacianSchemes { default Gauss linear corrected; }
        interpolationSchemes { default linear; }
        snGradSchemes  { default corrected; }
        wallDist { method meshWave; }
        """))
    write_dict(f"{case}/system/fvSolution", "dictionary", "fvSolution", textwrap.dedent("""\
        solvers
        {
            p { solver GAMG; tolerance 1e-6; relTol 0.05; smoother GaussSeidel; }
            "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1; }
        }
        SIMPLE
        {
            nNonOrthogonalCorrectors 1;
            consistent no;
            residualControl { p 1e-4; U 1e-5; "(k|omega)" 1e-5; }
        }
        relaxationFactors
        {
            fields { p 0.3; }
            equations { U 0.7; "(k|omega)" 0.7; }
        }
        """))
    write_dict(f"{case}/0/U", "volVectorField", "U", textwrap.dedent(f"""\
        dimensions      [0 1 -1 0 0 0 0];
        internalField   uniform ({U} 0 0);
        boundaryField
        {{
            inlet {{ type fixedValue; value uniform ({U} 0 0); }}
            outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({U} 0 0); }}
            top {{ type slip; }}
            bottom {{ type slip; }}
            body {{ type noSlip; }}
            frontAndBack {{ type empty; }}
        }}
        """))
    write_dict(f"{case}/0/p", "volScalarField", "p", textwrap.dedent("""\
        dimensions      [0 2 -2 0 0 0 0];
        internalField   uniform 0;
        boundaryField
        {
            inlet { type zeroGradient; }
            outlet { type fixedValue; value uniform 0; }
            top { type slip; }
            bottom { type slip; }
            body { type zeroGradient; }
            frontAndBack { type empty; }
        }
        """))
    write_dict(f"{case}/0/k", "volScalarField", "k", textwrap.dedent(f"""\
        dimensions      [0 2 -2 0 0 0 0];
        internalField   uniform {k_in};
        boundaryField
        {{
            inlet {{ type fixedValue; value uniform {k_in}; }}
            outlet {{ type inletOutlet; inletValue uniform {k_in}; value uniform {k_in}; }}
            top {{ type zeroGradient; }}
            bottom {{ type zeroGradient; }}
            body {{ type kqRWallFunction; value uniform {k_in}; }}
            frontAndBack {{ type empty; }}
        }}
        """))
    write_dict(f"{case}/0/omega", "volScalarField", "omega", textwrap.dedent(f"""\
        dimensions      [0 0 -1 0 0 0 0];
        internalField   uniform {omega_in};
        boundaryField
        {{
            inlet {{ type fixedValue; value uniform {omega_in}; }}
            outlet {{ type inletOutlet; inletValue uniform {omega_in}; value uniform {omega_in}; }}
            top {{ type zeroGradient; }}
            bottom {{ type zeroGradient; }}
            body {{ type omegaWallFunction; value uniform {omega_in}; }}
            frontAndBack {{ type empty; }}
        }}
        """))
    write_dict(f"{case}/0/nut", "volScalarField", "nut", textwrap.dedent("""\
        dimensions      [0 2 -1 0 0 0 0];
        internalField   uniform 0;
        boundaryField
        {
            inlet { type calculated; value uniform 0; }
            outlet { type calculated; value uniform 0; }
            top { type calculated; value uniform 0; }
            bottom { type calculated; value uniform 0; }
            body { type nutkWallFunction; value uniform 0; }
            frontAndBack { type empty; }
        }
        """))


def read_cp_from_vtp(vtp_path: str, U: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (centres, Cp) for body faces from an ASCII XML polydata (.vtp) with cell data p."""
    import xml.etree.ElementTree as ET

    root = ET.parse(vtp_path).getroot()
    piece = root.find(".//Piece")

    def arr(parent, name=None):
        for da in parent.iter("DataArray"):
            if name is None or da.get("Name") == name:
                return np.array(da.text.split(), dtype=float)
        raise KeyError(name)

    pts = arr(piece.find("Points")).reshape(-1, 3)
    polys = piece.find("Polys")
    conn = arr(polys, "connectivity").astype(int)
    offs = arr(polys, "offsets").astype(int)
    starts = np.r_[0, offs[:-1]]
    cells = [conn[a:b] for a, b in zip(starts, offs)]
    p = arr(piece.find("CellData"), "p")
    centres = np.array([pts[c].mean(axis=0) for c in cells])
    return centres, p / (0.5 * U ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--speed", type=float, default=5.0, help="m/s")
    ap.add_argument("--body-size", type=float, default=None, help="m, body mesh size (default chord/400)")
    ap.add_argument("--far-factor", type=float, default=0.3, help="far-field size as multiple of chord")
    ap.add_argument("--out", default="runs/case")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    case = os.path.abspath(args.out)
    pts = read_profile_m(args.profile)
    chord = pts[:, 0].max() - pts[:, 0].min()
    body_size = args.body_size or chord / 400
    thickness = 0.01 * chord
    msh = os.path.join(case, "mesh.msh")
    dims = build_mesh(pts, msh, body_size, args.far_factor, thickness)

    write_case(case, args.speed, chord, length_scale=0.1 * chord)
    run_of("gmshToFoam mesh.msh", case, os.path.join(case, "log.gmshToFoam"))
    fix_boundary(case)
    run_of("checkMesh", case, os.path.join(case, "log.checkMesh"))
    run_of("simpleFoam", case, os.path.join(case, "log.simpleFoam"))
    run_of("foamToVTK -ascii -latestTime -patches '(body)' -fields '(p)'", case, os.path.join(case, "log.foamToVTK"))

    vtk_dirs = sorted(d for d in os.listdir(os.path.join(case, "VTK")) if os.path.isdir(os.path.join(case, "VTK", d)))
    vtp = os.path.join(case, "VTK", vtk_dirs[-1], "boundary", "body.vtp")
    xc, cp = read_cp_from_vtp(vtp, args.speed)
    out_csv = os.path.join(case, "cp_rans.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "y_m", "cp"])
        for (x, y, _), c in zip(xc, cp):
            w.writerow([f"{x:.6f}", f"{y:.6f}", f"{c:.5f}"])
    print(f"chord {chord*1000:.1f} mm, U {args.speed} m/s, Re {args.speed*chord/NU:.2e}")
    print(f"min Cp (RANS) {cp.min():.3f} at x = {xc[np.argmin(cp),0]*1000:.1f} mm")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
