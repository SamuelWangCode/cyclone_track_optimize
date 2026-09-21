"""First-day optimal-minus-control circulation and transport diagnostics."""

from pathlib import Path
import sys, json
import numpy as np
import xarray as xr
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typhoon.diagnostics import smooth, raw_kinematics

R = 6371000.0
DT = 21600.0
S = 1e-5
G = 9.80665
CFG = {
    "init": "2026-08-27T00:00:00",
    "early_diagnostic": {
        "leads_hours": [0, 6, 12, 18, 24],
        "display_leads_hours": [0, 6, 12, 24],
        "levels_hpa": [850, 700],
        "main_box_NWSE": [25, 127, 12, 142],
        "display_box_NWSE": [30, 120, 10, 145],
        "smoothing_km": 100,
        "description": "OPT minus CTRL evolution, with relative-vorticity advection split against CTRL, stretching and explicit remainder. Compute products before smoothing; retain both pressure levels. Inspect temperature, specific humidity, relative humidity and convergence with the same times and terrain mask.",
    },
}


def rh(t, q, p):
    ew = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
    ei = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))
    alpha = np.clip((t - 250.16) / 23, 0, 1) ** 2
    vap = q * p / (0.621981 + (1 - 0.621981) * q)
    return 100 * vap / (alpha * ew + (1 - alpha) * ei)


def boxmask(lat, lon, box):
    n, w, s, e = box
    return (
        (lat[:, None] >= s)
        & (lat[:, None] <= n)
        & (lon[None, :] >= w)
        & (lon[None, :] <= e)
    )


def stats(a, lat, lon, mask):
    ok = mask & np.isfinite(a)
    w = np.cos(np.deg2rad(lat))[:, None] * ok
    w = w / w.sum()
    masked = np.where(ok, a, np.nan)
    ij = np.unravel_index(np.nanargmax(masked), a.shape)
    ik = np.unravel_index(np.nanargmin(masked), a.shape)
    return dict(
        mean=float(np.nansum(w * a)),
        rms=float(np.sqrt(np.nansum(w * a * a))),
        positive_area_fraction=float(w[a > 0].sum()),
        maximum={
            "value": float(a[ij]),
            "lat": float(lat[ij[0]]),
            "lon": float(lon[ij[1]]),
        },
        minimum={
            "value": float(a[ik]),
            "lat": float(lat[ik[0]]),
            "lon": float(lon[ik[1]]),
        },
    )


def early():
    folder = INPUT
    with xr.open_dataset(folder / "control_tianxing/000.nc") as d:
        lat = d.lat.values
        lon = d.lon.values
        sp = d.sp.values
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    co = np.cos(phi)[:, None]
    dx = lambda a: np.gradient(a, lam, axis=-1, edge_order=2) / (R * co)
    dy = lambda a: np.gradient(a, phi, axis=-2, edge_order=2) / R
    levels = [850, 700]
    hours = CFG["early_diagnostic"]["leads_hours"]
    main = boxmask(lat, lon, CFG["early_diagnostic"]["main_box_NWSE"])
    names = [
        "delta_zeta",
        "control_zeta",
        "optimal_zeta",
        "delta_divergence",
        "delta_relative_advection",
        "wind_on_control_gradient",
        "control_on_delta_gradient",
        "nonlinear_advection",
        "delta_planetary_advection",
        "delta_stretching",
        "delta_u",
        "delta_v",
        "delta_q",
        "delta_t",
        "delta_rh",
    ]
    records = []
    allstats = []
    budget = []
    checks = {"advection_decomposition_max_abs_s2": 0.0}
    for model in ["tianxing", "pangu_6h"]:
        times = []
        for h in hours:
            with xr.open_dataset(folder / f"control_{model}" / f"{h:03d}.nc") as f:
                b = f.load()
            with xr.open_dataset(folder / f"optimal_{model}" / f"{h:03d}.nc") as f:
                o = f.load()
            levs = []
            for p in levels:
                u, v = [b[x].sel(level=p).values.astype(float) for x in ["u", "v"]]
                ou, ov = [o[x].sel(level=p).values.astype(float) for x in ["u", "v"]]
                a = raw_kinematics(u, v, lat, lon)
                c = raw_kinematics(ou, ov, lat, lon)
                du = ou - u
                dv = ov - v
                dz = c["zeta"] - a["zeta"]
                wind = -du * dx(a["zeta"]) - dv * dy(a["zeta"])
                structure = -u * dx(dz) - v * dy(dz)
                nonlinear = -du * dx(dz) - dv * dy(dz)
                checks["advection_decomposition_max_abs_s2"] = max(
                    checks["advection_decomposition_max_abs_s2"],
                    float(
                        np.max(
                            np.abs(
                                c["relative_advection"]
                                - a["relative_advection"]
                                - wind
                                - structure
                                - nonlinear
                            )
                        )
                    ),
                )
                valid = binary_erosion(sp > p * 100, iterations=2)
                den = smooth(valid.astype(float), lat, lon, 100)
                sm = lambda x: smooth(
                    np.where(valid, x, 0), lat, lon, 100
                ) / np.maximum(den, 1e-12)
                get = lambda d, v: d[v].sel(level=p).values.astype(float)
                arrays = [
                    dz / S,
                    a["zeta"] / S,
                    c["zeta"] / S,
                    (c["divergence"] - a["divergence"]) / S,
                    (c["relative_advection"] - a["relative_advection"]) * DT / S,
                    wind * DT / S,
                    structure * DT / S,
                    nonlinear * DT / S,
                    (c["planetary_advection"] - a["planetary_advection"]) * DT / S,
                    (c["stretching"] - a["stretching"]) * DT / S,
                    du,
                    dv,
                    1000 * (get(o, "q") - get(b, "q")),
                    get(o, "t") - get(b, "t"),
                    rh(get(o, "t"), get(o, "q"), p * 100)
                    - rh(get(b, "t"), get(b, "q"), p * 100),
                ]
                arrays = np.stack(
                    [np.where(den >= 0.95, sm(x), np.nan) for x in arrays]
                )
                levs.append(arrays)
                for j, name in enumerate(names):
                    if name in [
                        "delta_zeta",
                        "delta_divergence",
                        "delta_q",
                        "delta_t",
                        "delta_rh",
                        "delta_relative_advection",
                    ]:
                        allstats.append(
                            dict(
                                model=model,
                                lead_hours=h,
                                level_hpa=p,
                                variable=name,
                                **stats(arrays[j], lat, lon, main),
                            )
                        )
            times.append(np.stack(levs))
        data = np.stack(times)
        records.append(data)
        for li, level in [(0, "850"), (1, "700"), (None, "850_700_mean")]:
            a = data.mean(axis=1) if li is None else data[:, li]
            actual = np.diff(a[:, 0], axis=0)
            rhs = {
                name: 0.5 * (a[:-1, j] + a[1:, j])
                for j, name in enumerate(names)
                if name
                in [
                    "delta_relative_advection",
                    "wind_on_control_gradient",
                    "control_on_delta_gradient",
                    "nonlinear_advection",
                    "delta_planetary_advection",
                    "delta_stretching",
                ]
            }
            rhs["remainder"] = (
                actual
                - rhs["delta_relative_advection"]
                - rhs["delta_planetary_advection"]
                - rhs["delta_stretching"]
            )
            for start, end in [(0, 6), (6, 12), (12, 18), (18, 24), (0, 24)]:
                sl = slice(start // 6, end // 6)
                act = actual[sl].sum(axis=0)
                row = dict(
                    model=model,
                    level=level,
                    start_hour=start,
                    end_hour=end,
                    actual=stats(act, lat, lon, main),
                    terms={},
                )
                for key, arr in rhs.items():
                    term = arr[sl].sum(axis=0)
                    st = stats(term, lat, lon, main)
                    valid = main & np.isfinite(act) & np.isfinite(term)
                    w = co * valid
                    w = w / w.sum()
                    st["projection_on_actual_change"] = float(
                        np.nansum(w * act * term) / np.nansum(w * act * act)
                    )
                    row["terms"][key] = st
                budget.append(row)
        print("EARLY_DIAGNOSED", model, flush=True)
    allarr = np.stack(records)
    ds = xr.Dataset(
        {
            name: (("model", "lead", "level", "lat", "lon"), allarr[:, :, :, j])
            for j, name in enumerate(names)
        },
        coords={
            "model": ["tianxing", "pangu_6h"],
            "lead": hours,
            "level": levels,
            "lat": lat,
            "lon": lon,
        },
    )
    ds.attrs.update(
        definition="OPT-minus-CTRL. zeta/divergence:1e-5s-1; advection/stretching:1e-5s-1 per6h; u/v:m/s;q:g/kg;t:K;RH:percentage points. Raw products before100km smoothing, same fixedIFS terrain mask. Remainder includes unresolved processes and temporal sampling.",
        initial_time=CFG["init"],
    )
    ds.to_netcdf(
        OUT / "early_response_fields.nc",
        encoding={k: {"zlib": True, "complevel": 1} for k in ds},
    )
    assert checks["advection_decomposition_max_abs_s2"] < 1e-15, checks
    (OUT / "early_response_statistics.json").write_text(
        json.dumps(
            {
                "config": CFG,
                "statistics": allstats,
                "budgets": budget,
                "checks": checks,
                "projection_definition": "Weighted inner product of cumulative term with actual OPT-minus-CTRL vorticity change, divided by squared actual change; may exceed100% or be negative due to cancellation. Diagnostic association, not a causal fraction.",
            },
            indent=2,
        )
    )
    print("EARLY_CHECKS", checks, flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    INPUT = args.fields
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    early()
