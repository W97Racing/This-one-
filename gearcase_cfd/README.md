# Gearcase flow tools (photo → profile → flow screen)

A small Python toolchain for looking at surface-running gearcase shapes:

1. `digitize.py` turns a side-view photo into a calibrated outline (mm).
2. `panel.py` runs a 2D inviscid source-panel solver on that outline and reports
   the surface pressure distribution, the cavitation screen at a given speed and
   depth, and a friction estimate.
3. `make_samples.py` generates matched rounded and pointed noses for comparison.
4. `test_gearcase_cfd.py` checks the solver against an analytic cylinder and the
   digitizer against a synthetic image of known size.

## Scope: what this is and is not

| Item | Status |
|---|---|
| Photo digitizing (calibrated, with overlay for checking) | Built, tested on a synthetic image |
| 2D inviscid pressure (Cp) on a closed section | Built, validated on a cylinder (exact match) |
| Cavitation-onset screen (−Cp_min vs. cavitation number) | Built. Screening rule only |
| Friction estimate (turbulent flat plate, form factor) | Built. Rough, per metre of span |
| Full 3D RANS CFD (OpenFOAM or similar) | **Not run.** See below |
| Propeller / blade-face pressure / blowout | **Not modelled.** A 2D panel code cannot capture it |
| Free surface, ventilation, air entrainment | **Not modelled** |
| Hull (hydroplane or runabout) flow | **Not modelled.** Only the appendage section is covered |

### Why no full 3D CFD yet

OpenFOAM is not installed in this environment. Installing it needs a third-party apt
repository and sudo, and the install was blocked. Nothing here depends on it, so the
pipeline is ready for a solver once it is installed, but no RANS result has been produced.

### Modelling assumptions to keep in mind

- **Side silhouette ≠ cross-section.** A photo gives the side outline. Treating that
  outline as a 2D section means an infinite-span shape. A gearcase nose is closer to a
  body of revolution about the prop shaft, so the honest next step is an axisymmetric
  solver. The 2D result is for comparing noses, not for absolute drag.
- **Sharp corners are singular.** Inviscid Cp goes unbounded at convex corners. A diamond
  nose gave Cp ≈ −1248 at a corner, which is a discretization artifact. Use smooth
  shapes for comparisons. Real flow at a sharp edge is also set by the boundary layer
  and separation, which this model ignores.
- **Cavitation screen.** Cavitation is predicted when −Cp_min > σ, with
  σ = (p₀ − p_v) / (½ρV²). At 100 mph, σ at the surface is about 0.10, so the screen
  flags almost any section. Treat it as a conservative warning, not a verdict. Nuclei,
  free-surface effects and transient loading matter in practice.

## Using it on a photo

```bash
# 1. Digitize. Measure a known length (e.g. cavitation plate to gearcase foot) on the
#    boat, and find its pixel length in the photo.
python digitize.py photo.jpg --scale-px 1200 --scale-mm 500 --out profile.csv
#    Check profile_overlay.png. If the outline is wrong, pass --invert (dark body)
#    or hand-trace the outline with --points-csv.

# 2. Analyse at speed and depth.
python panel.py profile.csv --speed-mph 100 --depth-mm 100 --flow +x --plot cp.png
```

`--flow` sets the direction water moves past the body in profile coordinates. Check
that the nose is where the water arrives. Photos must be taken square-on to the
gearcase. Perspective errors go straight into the scale.

## Verified results (synthetic noses, same length 400 mm, same max thickness 100 mm, 100 mph, 100 mm depth)

Generate with `python make_samples.py`, then run `python panel.py samples/<name>.csv ...`.

| Nose | Min Cp | Location | Cavitation screen (σ = 0.100) |
|---|---|---|---|
| Rounded (ellipse) | −0.563 | mid-body (x ≈ 0) | Flagged |
| Pointed (1−u²)^0.7 | −0.647 | mid-body | Flagged |

The pointed nose has about 15% higher suction peak at this resolution. This is
consistent with the article's point that a sharper shape raises suction peaks.
The 2D model cannot show the article's real-world blowout behaviour, which comes from
the free surface and the propeller.

## Tests

```bash
python test_gearcase_cfd.py
```

Passes: circle Cp matches analytic 1 − 4sin²θ exactly; orientation is normalized;
pointed nose has higher suction than rounded; digitizer recovers a 120 × 60 mm
synthetic body to within 0.2 mm.
