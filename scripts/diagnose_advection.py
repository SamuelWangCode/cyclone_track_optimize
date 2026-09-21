"""Separate wind, vorticity-pattern and nonlinear parts of advection error.

Exact relative-to-analysis algebra in a resolved horizontal diagnostic; it does
not isolate a network module or latent-heating mechanism.
"""

from pathlib import Path
import json
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_budget import CFG, R, DT, SCALE, SOURCES, raw_kinematics

INPUT = ROOT / "inputs/fields"
OUT = ROOT / "runs/budget"
from typhoon.diagnostics import smooth

COMPONENTS = [
    "wind_error_on_reference",
    "reference_wind_on_vorticity_error",
    "nonlinear_error_advection",
]


def main():
    full = json.loads((OUT / "vorticity_budget.json").read_text(encoding="utf-8"))
    original = {
        (r["source"], r["region"], r["level"], r["start_hour"]): r
        for r in full["phases"]
    }
    with xr.open_dataset(INPUT / "control_tianxing/000.nc") as d:
        lat, lon, sp = d.lat.values, d.lon.values, d.sp.values
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    co = np.cos(phi)[:, None]
    dx = lambda a: np.gradient(a, lam, axis=-1, edge_order=2) / (R * co)
    dy = lambda a: np.gradient(a, phi, axis=-2, edge_order=2) / R
    valid = [binary_erosion(sp > p * 100, iterations=2) for p in CFG["levels_hpa"]]
    den = [smooth(v.astype(float), lat, lon, 100) for v in valid]
    sf = lambda a, k: smooth(np.where(valid[k], a, 0), lat, lon, 100) / np.maximum(
        den[k], 1e-12
    )
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    ref = []
    for h in range(0, 169, 6):
        with xr.open_dataset(INPUT / "era5t" / f"{h:03d}.nc") as d:
            levs = []
            for k, p in enumerate(CFG["levels_hpa"]):
                u, v = [
                    d[key].sel(level=p).values.astype("float64") for key in ["u", "v"]
                ]
                z = raw_kinematics(u, v, lat, lon)["zeta"]
                levs.append((u, v, z, dx(z), dy(z)))
            ref.append(levs)
    phases, closure = [], []
    for source in SOURCES:
        if source == "era5t":
            continue
        times, errors = [], []
        for i, h in enumerate(range(0, 169, 6)):
            terms, err = [], []
            with xr.open_dataset(INPUT / source / f"{h:03d}.nc") as d:
                for k, p in enumerate(CFG["levels_hpa"]):
                    u, v = [
                        d[key].sel(level=p).values.astype("float64")
                        for key in ["u", "v"]
                    ]
                    z = raw_kinematics(u, v, lat, lon)["zeta"]
                    ur, vr, zr, gx, gy = ref[i][k]
                    du, dv, dz = u - ur, v - vr, z - zr
                    ex, ey = dx(dz), dy(dz)
                    raw = [-du * gx - dv * gy, -ur * ex - vr * ey, -du * ex - dv * ey]
                    terms.append(np.stack([sf(vv, k) * DT / SCALE for vv in raw]))
                    err.append(sf(dz, k) / SCALE)
            times.append(terms)
            errors.append(err)
        terms, errors = np.array(times), np.array(errors)
        for level in ["850", "700", "low_mean"]:
            a = (
                terms.mean(axis=1)
                if level == "low_mean"
                else terms[:, CFG["levels_hpa"].index(int(level))]
            )
            e = (
                errors.mean(axis=1)
                if level == "low_mean"
                else errors[:, CFG["levels_hpa"].index(int(level))]
            )
            projections = 0.5 * (e[1:] + e[:-1])[:, None] * 0.5 * (a[1:] + a[:-1])
            coverage = (
                np.minimum(*den)
                if level == "low_mean"
                else den[CFG["levels_hpa"].index(int(level))]
            )
            for region, (s, w, n, ee) in CFG["regions_SWNE"].items():
                mask = (
                    (yy >= s) & (yy <= n) & (xx >= w) & (xx <= ee) & (coverage >= 0.95)
                )
                weight = co * mask
                weight /= weight.sum()
                values = (projections * weight).sum(axis=(-2, -1))
                for start, end in CFG["phase_hours"]:
                    val = values[start // 6 : end // 6].sum(axis=0)
                    target = original[source, region, level, start][
                        "error_growth_projection"
                    ]["relative_advection"]
                    closure.append(abs(float(val.sum()) - target))
                    phases.append(
                        dict(
                            source=source,
                            level=level,
                            region=region,
                            start_hour=start,
                            end_hour=end,
                            components=dict(zip(COMPONENTS, map(float, val))),
                            total=float(val.sum()),
                        )
                    )
        print("ADVECTION_DECOMPOSED", source, flush=True)
    assert max(closure) < 1e-10, max(closure)
    (OUT / "advection.json").write_text(
        json.dumps(
            dict(
                phases=phases,
                checks=dict(max_closure=max(closure)),
                definitions={
                    "wind_error_on_reference": "-delta_V dot grad(zeta_reference)",
                    "reference_wind_on_vorticity_error": "-V_reference dot grad(delta_zeta)",
                    "nonlinear_error_advection": "-delta_V dot grad(delta_zeta)",
                },
                meaning="Each tendency difference projected onto midpoint forecast-minus-ERA5T vorticity error. Positive amplifies squared pattern error. Finite-amplitude bookkeeping, not causal fractions.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print("ADVECTION_COMPLETE", max(closure), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder containing vorticity_budget.json",
    )
    args = p.parse_args()
    INPUT = args.fields
    OUT = args.output
    main()
