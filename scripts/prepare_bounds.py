"""Freeze numeric wet-energy and 3-EDA-spread bounds before optimization."""

from pathlib import Path
from datetime import datetime
import json, sys, hashlib
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typhoon.fields import CHANNELS, LEVELS, SURFACE, canonical
from typhoon.energy import total_energy_coefficients, pressure_channel_index
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--initial", type=Path, required=True)
parser.add_argument("--spread", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
stamp = datetime(2026, 8, 27)
source = args.spread
out = args.output
out.mkdir(parents=True, exist_ok=True)
with xr.open_dataset(args.initial / "sfc.nc") as s:
    initial = canonical(s, stamp).load()
lat = initial.lat.values
lon = initial.lon.values
with xr.open_dataset(source / "sfc.nc") as s, xr.open_dataset(source / "pl.nc") as p:
    s = canonical(s, stamp, global_grid=False).load()
    p = canonical(p, stamp, global_grid=False).load()


def refine(da):
    if not np.isfinite(da.values).all() or np.min(da.values) < 0:
        raise ValueError("Invalid EDA spread")
    variance = da.astype("float64") ** 2
    # Bilinear variance interpolation, periodic at the Greenwich seam. No new
    # small-scale uncertainty is inferred from the 0.5-degree spread product.
    variance = xr.concat(
        [variance, variance.isel(lon=[0]).assign_coords(lon=[360.0])], dim="lon"
    )
    refined = np.sqrt(
        variance.interp(lat=lat, lon=lon).transpose("lat", "lon").values
    ).astype("float32")
    if not np.isfinite(refined).all():
        raise ValueError("Spread interpolation left missing cells")
    return refined


sigma = np.stack(
    [refine(s[v]) for v in SURFACE]
    + [refine(p[v].sel(level=lev)) for v in ["u", "v", "z", "t", "r"] for lev in LEVELS]
)
q_sigma = np.stack([refine(p.q.sel(level=lev)) for lev in LEVELS])
mask = np.ones_like(sigma, dtype=bool)
for lev in LEVELS:
    above = initial.sp.values >= lev * 100
    for var in ["u", "v", "z", "t", "r"]:
        mask[pressure_channel_index(CHANNELS, f"{var}{lev}")] = above
caps = 3 * sigma * mask
dry = total_energy_coefficients(
    CHANNELS, LEVELS, lat, lon, initial.sp.values, moist=False
)[0]
dry_budget = float(np.sum(dry * (sigma * mask) ** 2, dtype=np.float64))
humidity_coeff = (
    dry[[pressure_channel_index(CHANNELS, f"t{lev}") for lev in LEVELS]]
    * ((2.5e6) / 1004.0) ** 2
)
humidity_mask = np.stack([initial.sp.values >= lev * 100 for lev in LEVELS])
wet_budget = float(
    np.sum(humidity_coeff * (q_sigma * humidity_mask) ** 2, dtype=np.float64)
)
if not dry_budget > 0 or not wet_budget > 0:
    raise ValueError("Invalid wet-energy budget")
# Unique artifact channel labels; physical tensor order remains TianXing's.
names = CHANNELS.copy()
names[2] = "u100m"
names[3] = "v100m"
ds = xr.Dataset(
    {
        "sigma": (("channel", "lat", "lon"), sigma),
        "max_increment": (("channel", "lat", "lon"), caps),
        "control_mask": (("channel", "lat", "lon"), mask.astype("int8")),
        "q_sigma": (("level", "lat", "lon"), q_sigma),
    },
    coords={"channel": names, "level": LEVELS, "lat": lat, "lon": lon},
    attrs={
        "init_UTC": stamp.isoformat(),
        "initial_source": "IFS operational analysis",
        "spread_source": "ERA5 EDA ensemble_spread; uncertainty proxy, not operational IFS analysis-error covariance",
        "energy_limit_J_kg": dry_budget + wet_budget,
        "energy_budget_definition": "Expected one-spread wet perturbation energy with fixed global area and above-ground layer weights; humidity weight 1",
        "pointwise_cap": "3 sigma at each variable / level / grid cell; below-ground pressure-level controls frozen",
        "interpolation": "0.5 to 0.25 degree: bilinear variance interpolation with periodic longitude, then square root",
        "physical_tensor_order": "73-channel TianXing order. u100m/v100m denote 100 m; u100/v100 denote 100 hPa.",
    },
)
target = out / "eda_bounds.nc"
temp = target.with_suffix(".partial.nc")
ds.to_netcdf(temp, encoding={v: {"zlib": True, "complevel": 1} for v in ds.data_vars})
temp.replace(target)
print(target, flush=True)
