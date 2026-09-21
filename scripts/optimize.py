"""Seven-day circulation objective under moist-energy and pointwise bounds."""

from pathlib import Path
from datetime import datetime, timedelta
import json, sys, time, traceback, argparse
import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typhoon.fields import read_pair, tianxing_state, CHANNELS, LEVELS
from typhoon.tianxing import TianXing
from typhoon.moist_energy import TianXingMoistEnergyConstraint
from typhoon.objectives import (
    CirculationOperator,
    circulation_pattern_loss,
    reference_window_weights,
)
from typhoon.optimize import discrete_adjoint, rollout_loss, fit_initial_state


def clean(ds):
    return ds.rename(
        {
            k: v
            for k, v in {
                "latitude": "lat",
                "longitude": "lon",
                "valid_time": "time",
                "pressure_level": "level",
            }.items()
            if k in ds.coords
        }
    ).sortby("lat", ascending=False)


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(
        description="Solve for optimal initial conditions using TianXing."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/optimization.json"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--initial",
        type=Path,
        required=True,
        help="Folder with initial sfc.nc and pl.nc",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Folder with all 28 future target times",
    )
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument(
        "--warm-start",
        type=Path,
        required=True,
        help="Normalized preceding feasible initial state",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf8"))
    init = datetime.fromisoformat(cfg["init_UTC"])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    reference = args.reference
    bounds_path = args.bounds
    (out / "experiment.json").write_text(
        json.dumps(
            {**cfg, "state": "preparing", "started": datetime.now().isoformat()},
            indent=2,
        )
    )
    model = TianXing(model_root=args.model_root)
    pl, sfc = read_pair(args.initial, init)
    lat = sfc.lat.values
    lon = sfc.lon.values
    ps = sfc.sp.values.copy()
    archived_q = pl.q.sel(level=LEVELS).values[None].copy()
    x0 = model.normalize(tianxing_state(pl, sfc)).detach()
    del pl, sfc
    with xr.open_dataset(bounds_path) as b:
        sigma = torch.as_tensor(b.sigma.values[None], device=model.device) / model.std
        caps = (
            torch.as_tensor(b.max_increment.values[None], device=model.device)
            / model.std
        )
        controls = torch.as_tensor(
            b.control_mask.values[None].astype(bool), device=model.device
        )
        budget = float(b.attrs["energy_limit_J_kg"])
        q_caps = (
            3 * b.q_sigma.values[None] if cfg.get("common_variables_only") else None
        )
    if cfg.get("common_variables_only"):
        controls[:, [2, 3, 5, 7]] = False
    constraint = TianXingMoistEnergyConstraint(
        x0,
        sigma,
        caps,
        budget,
        channels=CHANNELS,
        levels_hpa=LEVELS,
        latitude=lat,
        longitude=lon,
        surface_pressure_pa=ps,
        model_mean=model.mean,
        model_std=model.std,
        control_mask=controls,
        q_max_increment=q_caps,
        archived_background_q=archived_q if cfg.get("common_variables_only") else None,
    )
    # RH may contain pre-existing supersaturation. Retain that background,
    # enforce nonnegative humidity for new increments without imposing RH<=100.
    lower = torch.full_like(x0, -torch.inf)
    for lev in LEVELS:
        i = CHANNELS.index(f"r{lev}")
        lower[:, i] = torch.minimum(x0[:, i], -model.mean[:, i] / model.std[:, i])
    constraint.lower = lower
    del sigma, caps, controls
    refs = {
        r["valid_time"]: r
        for r in json.loads((ROOT / cfg["reference_track"]).read_text(encoding="utf8"))[
            "points"
        ]
    }
    targets = []
    with xr.open_dataset(reference / "sfc.nc") as sf, xr.open_dataset(
        reference / "pl.nc"
    ) as pf:
        sf = clean(sf)
        pf = clean(pf)
        rlat = sf.lat.values
        rlon = sf.lon.values
        sy = slice(
            int(round((90 - rlat[0]) / 0.25)), int(round((90 - rlat[-1]) / 0.25)) + 1
        )
        sx = slice(int(round(rlon[0] / 0.25)), int(round(rlon[-1] / 0.25)) + 1)
        if not np.array_equal(lat[sy], rlat) or not np.array_equal(lon[sx], rlon):
            raise ValueError("Target/model grid mismatch")
        for lead in range(6, 169, 6):
            valid = init + timedelta(hours=lead)
            ref = refs[valid.isoformat()]
            s = sf.sel(time=np.datetime64(valid))
            p = pf.sel(time=np.datetime64(valid), level=[850, 700])
            op = CirculationOperator(
                rlat,
                rlon,
                np.stack([s.sp.values >= lev * 100 for lev in [850, 700]]),
                model.device,
            )
            zeta = op(
                torch.as_tensor(p.u.values, device=model.device),
                torch.as_tensor(p.v.values, device=model.device),
            ).detach()
            weights = (
                torch.as_tensor(
                    reference_window_weights(rlat, rlon, ref["lat"], ref["lon"]),
                    device=model.device,
                )
                * op.valid
            )
            if not torch.isfinite(zeta).all() or float(weights.sum()) <= 0:
                raise ValueError("Invalid circulation target")
            targets.append((op, zeta, weights))
    ui = [CHANNELS.index("u850"), CHANNELS.index("u700")]
    vi = [CHANNELS.index("v850"), CHANNELS.index("v700")]

    def step(x, t):
        return model(x, init + timedelta(hours=6 * (t + 1)))

    def loss(x, target):
        op, truth, weights = target
        u = (x[:, ui, sy, sx] * model.std[:, ui] + model.mean[:, ui])[0]
        v = (x[:, vi, sy, sx] * model.std[:, vi] + model.mean[:, vi])[0]
        return circulation_pattern_loss(op(u, v), truth, weights)

    started = time.time()
    value, gradient = discrete_adjoint(step, x0, targets, loss)
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
        raise ValueError("Invalid full-chain gradient")
    direction = gradient / gradient.abs().max()
    analytic = float((gradient * direction).sum(dtype=torch.float64))
    del gradient
    eps = 0.0003
    with torch.no_grad():
        plus = float(rollout_loss(step, x0 + eps * direction, targets, loss))
        minus = float(rollout_loss(step, x0 - eps * direction, targets, loss))
    numeric = (plus - minus) / (2 * eps)
    check = {
        "hours": 168,
        "target_n": 28,
        "epsilon_normalized": eps,
        "adjoint": analytic,
        "finite_difference": numeric,
        "relative_error": abs(numeric - analytic) / abs(analytic),
    }
    check["passed"] = check["relative_error"] < 0.03
    (out / "gradient_check.json").write_text(json.dumps(check, indent=2))
    print("GRADIENT_CHECK", json.dumps(check), flush=True)
    del direction
    if not check["passed"]:
        raise ValueError("New circulation objective gradient check failed")
    history = []

    def callback(entry, best):
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        np.save(out / "best_normalized.npy", best.detach().cpu().numpy())
        print("ITERATION", json.dumps(entry), flush=True)

    warm_start = torch.as_tensor(np.load(args.warm_start), device=model.device)
    best, history = fit_initial_state(
        step,
        constraint,
        targets,
        loss,
        iterations=cfg["iterations"],
        lr=cfg["initial_step"],
        max_backtracks=5,
        callback=callback,
        warm_start=warm_start,
    )
    energy = float(constraint.energy(best))
    delta = best - x0
    # Report rather than hide floating-point subtraction tolerance in boxes.
    numerical_tolerance = 8 * torch.finfo(best.dtype).eps * (1 + x0.abs())
    box_excess = float(
        (delta.abs() - constraint.max_increment - numerical_tolerance)
        .clamp_min(0)
        .max()
    )
    if energy > budget * (1 + 1e-6) or box_excess > 0:
        raise ValueError("Saved perturbation violates constraints")
    if cfg.get("common_variables_only"):
        dq = constraint.humidity(best) - constraint.background_q
        if (dq.abs() > constraint.q_max_increment).any() or (
            dq + constraint.archived_background_q < 0
        ).any():
            raise ValueError("Shared specific-humidity constraints violated")
    physical_delta = (delta * model.std).detach().cpu().numpy()
    np.save(out / "delta_physical.npy", physical_delta)
    np.save(out / "optimized_physical.npy", model.physical(best).detach().cpu().numpy())
    with torch.no_grad():
        final_loss = float(rollout_loss(step, best, targets, loss))
    result = {
        "state": "complete_first_finite_optimization",
        "baseline_loss": value,
        "optimized_loss": final_loss,
        "loss_reduction_fraction": 1 - final_loss / value,
        "energy_J_kg": energy,
        "energy_limit_J_kg": budget,
        "box_excess_after_roundoff_tolerance": box_excess,
        "iterations": len(history),
        "accepted_iterations": sum(r["accepted"] for r in history),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_GB": torch.cuda.max_memory_allocated() / 1e9,
        "model": model.provenance,
        "result_scope": "Best feasible state in this finite run, not a proven global optimum. Track verification and cross-model transfer are separate and not yet implied by objective reduction.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE_OPTIMIZATION", json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
