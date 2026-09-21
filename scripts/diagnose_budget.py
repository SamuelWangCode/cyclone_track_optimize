"""Resolved horizontal vorticity tendencies on identical fixed domains.

Uses existing forecasts only. The remainder is deliberately not a diabatic term.
Error-growth projections are bookkeeping identities, not causal percentages.
"""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typhoon.diagnostics import smooth

OUT = ROOT / "runs/budget"
INPUT = ROOT / "inputs/fields"
CFG_PATH = ROOT / "config/budget.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
R = 6371000.0
OMEGA = 7.292115e-5
DT = 21600.0
SCALE = CFG["vorticity_scale_s1"]
KEYS = ["zeta", "divergence", "relative_advection", "planetary_advection", "stretching"]
TERMS = KEYS[2:]
SOURCES = {"era5t": None}
for model in ["tianxing", "pangu_6h"]:
    for state in ["control", "optimal"]:
        SOURCES[state + "_" + model] = None
    SOURCES["era5init_" + model] = None


def raw_kinematics(u, v, lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    co = np.cos(phi)[:, None]
    dx = lambda a: np.gradient(a, lam, axis=-1, edge_order=2) / (R * co)
    dy = lambda a: np.gradient(a, phi, axis=-2, edge_order=2) / R
    zeta = dx(v) - dy(u * co) / co
    div = dx(u) + dy(v * co) / co
    f = 2 * OMEGA * np.sin(phi)[:, None]
    beta = 2 * OMEGA * co / R
    return dict(
        zeta=zeta,
        divergence=div,
        relative_advection=-u * dx(zeta) - v * dy(zeta),
        planetary_advection=-v * beta,
        stretching=-(zeta + f) * div,
    )


def source_fields(source, lat, lon, valid, den):
    all_times = []
    for h in range(0, 169, 6):
        with xr.open_dataset(INPUT / source / f"{h:03d}.nc") as d:
            assert np.array_equal(d.lat.values, lat) and np.array_equal(
                d.lon.values, lon
            )
            by_level = []
            for k, p in enumerate(CFG["levels_hpa"]):
                a = raw_kinematics(
                    d.u.sel(level=p).values.astype("float64"),
                    d.v.sel(level=p).values.astype("float64"),
                    lat,
                    lon,
                )
                by_level.append(
                    np.stack(
                        [
                            smooth(np.where(valid[k], a[key], 0), lat, lon, 100)
                            / np.maximum(den[k], 1e-12)
                            for key in KEYS
                        ]
                    )
                )
        # All stored quantities nondimensionalized: zeta/div in 1e-5 s-1,
        # physical tendencies in that unit per six hours.
        arr = np.stack(by_level)
        arr[:, :2] /= SCALE
        arr[:, 2:] *= DT / SCALE
        all_times.append(arr)
    print("SOURCE_LOADED", source, flush=True)
    return np.stack(all_times)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(INPUT / "control_tianxing/000.nc") as d:
        lat, lon, sp = d.lat.values, d.lon.values, d.sp.values
    valid = [binary_erosion(sp > p * 100, iterations=2) for p in CFG["levels_hpa"]]
    den = [smooth(v.astype(float), lat, lon, 100) for v in valid]
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    weight0 = np.cos(np.deg2rad(yy))
    masks = {}
    for region, (s, w, n, e) in CFG["regions_SWNE"].items():
        box = (yy >= s) & (yy <= n) & (xx >= w) & (xx <= e)
        for lev in ["850", "700", "low_mean"]:
            coverage = (
                np.minimum(*den)
                if lev == "low_mean"
                else den[CFG["levels_hpa"].index(int(lev))]
            )
            mask = box & (coverage >= 0.95)
            weights = weight0 * mask
            masks[region, lev] = weights / weights.sum()
    ref = source_fields("era5t", lat, lon, valid, den)
    rows, intervals, phases, initial = [], [], [], []
    closure_max = 0.0
    phase_maps = {}
    for source in SOURCES:
        data = ref if source == "era5t" else source_fields(source, lat, lon, valid, den)
        for lev in ["850", "700", "low_mean"]:
            a = (
                data.mean(axis=1)
                if lev == "low_mean"
                else data[:, CFG["levels_hpa"].index(int(lev))]
            )
            b = (
                ref.mean(axis=1)
                if lev == "low_mean"
                else ref[:, CFG["levels_hpa"].index(int(lev))]
            )
            err = a[:, 0] - b[:, 0]
            actual = np.diff(a[:, 0], axis=0)
            actual_ref = np.diff(b[:, 0], axis=0)
            rhs = 0.5 * (a[:-1, 2:] + a[1:, 2:])
            rhs_ref = 0.5 * (b[:-1, 2:] + b[1:, 2:])
            residual = actual - rhs.sum(axis=1)
            residual_ref = actual_ref - rhs_ref.sum(axis=1)
            err_mid = 0.5 * (err[1:] + err[:-1])
            delta = dict(zip(TERMS, np.moveaxis(rhs - rhs_ref, 1, 0)))
            delta["residual"] = residual - residual_ref
            delta["actual"] = actual - actual_ref
            for region in CFG["regions_SWNE"]:
                w = masks[region, lev]
                avg = lambda x: np.sum(x * w, axis=(-2, -1))
                for i, h in enumerate(range(0, 169, 6)):
                    row = dict(
                        source=source,
                        level=lev,
                        region=region,
                        lead_hours=h,
                        zeta_mean=float(avg(a[i, 0])),
                        zeta_rms=float(np.sqrt(avg(a[i, 0] ** 2))),
                        divergence_mean=float(avg(a[i, 1])),
                        error_rmse=float(np.sqrt(avg(err[i] ** 2))),
                    )
                    rows.append(row)
                egrowth = 0.5 * np.diff(avg(err**2))
                proj = {k: avg(err_mid * v) for k, v in delta.items()}
                closure_max = max(
                    closure_max,
                    float(np.max(np.abs(egrowth - proj["actual"]))),
                    float(
                        np.max(
                            np.abs(
                                proj["actual"]
                                - sum(proj[k] for k in TERMS + ["residual"])
                            )
                        )
                    ),
                )
                for i in range(28):
                    intervals.append(
                        dict(
                            source=source,
                            level=lev,
                            region=region,
                            start_hour=i * 6,
                            end_hour=(i + 1) * 6,
                            actual_tendency_mean=float(avg(actual[i])),
                            term_means={
                                k: float(avg(rhs[i, j])) for j, k in enumerate(TERMS)
                            },
                            residual_mean=float(avg(residual[i])),
                            tendency_rms=float(np.sqrt(avg(actual[i] ** 2))),
                            residual_rms=float(np.sqrt(avg(residual[i] ** 2))),
                            error_energy_change=float(egrowth[i]),
                            error_growth_projection={
                                k: float(v[i]) for k, v in proj.items()
                            },
                        )
                    )
                for start, end in CFG["phase_hours"]:
                    sl = slice(start // 6, end // 6)
                    phases.append(
                        dict(
                            source=source,
                            level=lev,
                            region=region,
                            start_hour=start,
                            end_hour=end,
                            error_rmse_start=float(np.sqrt(avg(err[start // 6] ** 2))),
                            error_rmse_end=float(np.sqrt(avg(err[end // 6] ** 2))),
                            error_energy_change=float(egrowth[sl].sum()),
                            error_growth_projection={
                                k: float(v[sl].sum()) for k, v in proj.items()
                            },
                            mean_tendency_rms=float(
                                np.sqrt(avg(actual[sl] ** 2).mean())
                            ),
                            mean_residual_rms=float(
                                np.sqrt(avg(residual[sl] ** 2).mean())
                            ),
                        )
                    )
            if lev == "low_mean" and source != "era5t":
                for start, end in CFG["phase_hours"]:
                    for key in [
                        "relative_advection",
                        "planetary_advection",
                        "stretching",
                        "residual",
                        "actual",
                    ]:
                        phase_maps[f"{source}_{start}_{end}_{key}"] = (
                            ("lat", "lon"),
                            (err_mid * delta[key])[start // 6 : end // 6]
                            .sum(axis=0)
                            .astype("float32"),
                        )
        if source.startswith("optimal"):
            ctrl = source_fields(
                source.replace("optimal", "control"), lat, lon, valid, den
            )[0]
            dd = data[0] - ctrl
            for (region, lev), w in masks.items():
                arr = (
                    dd.mean(axis=0)
                    if lev == "low_mean"
                    else dd[CFG["levels_hpa"].index(int(lev))]
                )
                initial.append(
                    dict(
                        source=source,
                        region=region,
                        level=lev,
                        delta_vorticity_rms=float(np.sqrt(np.sum(arr[0] ** 2 * w))),
                        delta_divergence_rms=float(np.sqrt(np.sum(arr[1] ** 2 * w))),
                    )
                )
        print("SOURCE_DIAGNOSED", source, flush=True)
    assert closure_max < 1e-10, closure_max
    payload = dict(
        config=CFG,
        units={
            "zeta": "1e-5 s-1",
            "tendency": "1e-5 s-1 per 6 h",
            "projection": "(1e-5 s-1)^2",
        },
        rows=rows,
        intervals=intervals,
        phases=phases,
        initial=initial,
        checks={
            "sources": len(SOURCES),
            "times_per_source": 29,
            "intervals_per_source": 28,
            "max_energy_identity_closure": closure_max,
            "config_sha256": hashlib.sha256(CFG_PATH.read_bytes()).hexdigest(),
        },
    )
    (OUT / "vorticity_budget.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ds = xr.Dataset(phase_maps, coords=dict(lat=lat, lon=lon))
    ds.to_netcdf(
        OUT / "error_growth_maps.nc",
        encoding={v: {"zlib": True, "complevel": 1} for v in ds.data_vars},
    )
    print("PROCESS_COMPLETE", json.dumps(payload["checks"]), flush=True)


def configure():
    import argparse

    global INPUT, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    INPUT = args.fields
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    configure()
    main()
