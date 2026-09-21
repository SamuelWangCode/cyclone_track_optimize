"""Prepare common physical initial conditions for TianXing and Pangu-Weather."""

from pathlib import Path
from datetime import datetime
import json, sys, hashlib
import numpy as np
import xarray as xr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
from typhoon.fields import read_pair, CHANNELS, LEVELS, SURFACE
from typhoon.energy import pressure_channel_index, total_energy_coefficients
from typhoon.moist_energy import specific_humidity
import argparse

parser = argparse.ArgumentParser(
    description="Apply the frozen physical correction to original IFS fields."
)
parser.add_argument("--initial", type=Path, required=True)
parser.add_argument("--delta", type=Path, required=True)
parser.add_argument("--bounds", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--region", choices=["global", "inside", "outside"], default="global"
)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
p, s = read_pair(args.initial, datetime(2026, 8, 27))
delta = np.load(args.delta)[0].copy()
if args.region != "global":
    mask = (
        (s.lat.values[:, None] >= 0)
        & (s.lat.values[:, None] <= 55)
        & (s.lon.values[None, :] >= 90)
        & (s.lon.values[None, :] <= 170)
    )
    delta *= mask if args.region == "inside" else ~mask
delta[[2, 3, 5, 7]] = 0
with xr.open_dataset(args.bounds) as b:
    caps = b.max_increment.values
    qcaps = 3 * b.q_sigma.values
    budget = float(b.attrs["energy_limit_J_kg"])
ti = [CHANNELS.index(f"t{v}") for v in LEVELS]
ri = [CHANNELS.index(f"r{v}") for v in LEVELS]
t = torch.tensor(p.t.sel(level=LEVELS).values)
r = torch.tensor(p.r.sel(level=LEVELS).values)
pressure = torch.tensor(np.asarray(LEVELS)[:, None, None] * 100)
q0 = p.q.sel(level=LEVELS).values.copy()
diagnosed0 = specific_humidity(t, r, pressure)
corrected = np.zeros_like(q0, dtype=bool)
for _ in range(32):
    dq = (
        specific_humidity(
            t + torch.tensor(delta[ti]), r + torch.tensor(delta[ri]), pressure
        )
        - diagnosed0
    ).numpy()
    qnew = (q0 + dq).astype(np.float32)
    actual_dq = qnew.astype(np.float64) - q0
    bad = (qnew < 0) | (np.abs(actual_dq) > qcaps)
    if not bad.any():
        break
    corrected |= bad
    for indices in [ti, ri]:
        values = delta[indices]
        values[bad] *= 0.5
        delta[indices] = values
else:
    raise ValueError("Physical humidity transfer did not satisfy bounds")
# Check the actual shared physical increment, including q serialization.
if not np.isfinite(delta).all() or np.any(
    np.abs(delta) > caps + 1e-4 * np.maximum(caps, 1e-6)
):
    raise ValueError("Native pointwise bound failed")
coef = total_energy_coefficients(
    CHANNELS, LEVELS, s.lat.values, s.lon.values, s.sp.values, moist=False
)[0]
energy = float(
    np.sum(coef * delta**2, dtype=np.float64)
    + np.sum(coef[ti] * ((2.5e6) / 1004.0) ** 2 * actual_dq**2, dtype=np.float64)
)
if energy > budget or np.any(qnew < 0) or np.any(np.abs(actual_dq) > qcaps):
    raise ValueError("Shared physical constraints failed")
for i, var in enumerate(SURFACE):
    s[var].values[:] += delta[i]
for var in ["u", "v", "z", "t", "r"]:
    for lev in LEVELS:
        p[var].loc[dict(level=lev)] = (
            p[var].sel(level=lev)
            + delta[pressure_channel_index(CHANNELS, f"{var}{lev}")]
        )
for i, lev in enumerate(LEVELS):
    p.q.loc[dict(level=lev)] = qnew[i]
meta = {
    "source": "Original IFS plus the constrained physical initial perturbation",
    "region": args.region,
    "humidity_serialization_contracted_cells": int(corrected.sum()),
    "energy_J_kg": energy,
    "energy_limit_J_kg": budget,
    "delta_sha256": hashlib.sha256(delta.tobytes()).hexdigest(),
}
for kind, ds in [("sfc", s), ("pl", p)]:
    ds.attrs.update(initial_source=meta["source"])
    ds.to_netcdf(
        out / f"{kind}.nc",
        encoding={k: {"zlib": True, "complevel": 1} for k in ds.data_vars},
    )
np.save(out / "delta.npy", delta[None])
(out / "initial.json").write_text(json.dumps(meta, indent=2))
