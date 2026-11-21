# racecfd: flow testing for race boat and gearcase development

A command-line tool that runs the flow analyses in `../gearcase_cfd` over speed and depth
sweeps, compares shapes, and writes CSV, PNG and Markdown outputs.

## Commands

| Command | What it does | Speed | Use it for |
|---|---|---|---|
| `sweep` | Panel analysis over speeds × depths | seconds | cavitation onset vs speed and depth |
| `compare` | Same, for several sections side by side | seconds | nose or gearcase shape choices |
| `rans` | Full 2D RANS, OpenFOAM k-ω SST | tens of minutes | detailed Cp and drag on one shape |

```bash
python racecfd.py sweep  profile.csv --speeds 40 60 80 100 --depths 50 150 --out out/run1
python racecfd.py compare a.csv b.csv --speed 100 --depth 50 --out out/cmp
python racecfd.py rans   profile.csv --speed 5 --out out/rans
```

Profiles are CSV files with `x_mm,y_mm` columns, traced from photos with
`../gearcase_cfd/digitize.py`. Flow goes in +x by default. Put the nose at low x.

## Outputs

- `sweep.csv` / `compare.csv`: min Cp, cavitation number, cavitation flag, friction per metre of span
- `sweep.png`: suction peak against cavitation number for each depth
- `report.md`: summary table and scope notes
- `rans`: `cp_rans.csv`, OpenFOAM logs, and VTK output

## What the numbers mean

- **Cp is non-dimensional.** Min Cp depends on the shape only. Cavitation depends on speed
  and depth through sigma = (p0 - pv)/(0.5 rho V^2). So a shape with a deep suction peak is
  flagged first as speed increases.
- **Cavitation flag is a screen.** It says the suction peak is deep enough to cavitate if
  nuclei are present. It does not predict real cavitation or blowout.
- **Friction is an estimate** (turbulent flat plate with form factor), per metre of span.

## Limits

- 2D infinite-span sections. Gearcase noses are closer to bodies of revolution (`../gearcase_cfd/axisym.py`,
  not yet validated).
- No free surface, no ventilation, no propeller. Blowout and surface-running prop behaviour
  are not modelled. These are the main design drivers for surface drives.
- RANS on blunt, separated shapes needs care: the gearcase silhouette diverged at default
  relaxation and was stable with upwind momentum and relaxation 0.3 (U) / 0.1 (p).
- Digitized profiles carry scale uncertainty; see the notes in each profile folder.
