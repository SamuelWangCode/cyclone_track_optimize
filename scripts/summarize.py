"""Recalculate track scores, steering decomposition, and budget closure."""

from pathlib import Path
import sys, json, csv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typhoon.tracks import great_circle_km
from typhoon.diagnostics import decompose_steering

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "tables")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main = read("tracks.json")
    groups = [
        (
            g["model"],
            "control" if g["experiment"] == "baseline" else "optimal",
            g["rows"],
        )
        for g in main["results"]
    ]
    groups += [
        (g["model"], "era5t_start", g["rows"])
        for g in read("reanalysis_tracks.json")["results"]
    ]
    groups += [
        (g["model"], g["variant"], g["rows"])
        for g in read("regional_tracks.json")["results"]
    ]
    groups += [
        ("era5t", "reference_verification", g["rows"])
        for g in main["reference_results"]
    ]
    scores = []
    positions = []
    for model, state, rows in groups:
        future = [r for r in rows if r["lead_hours"] > 0]
        assert len(future) == 28 and all(r["center"] and r["reference"] for r in future)
        errors = []
        for r in future:
            c, o = r["center"], r["reference"]
            error = float(great_circle_km(c["lat"], c["lon"], o["lat"], o["lon"]))
            np.testing.assert_allclose(error, r["distance_km"], rtol=1e-10, atol=1e-8)
            errors.append(error)
            positions.append(
                {
                    "model": model,
                    "experiment": state,
                    "lead_hours": r["lead_hours"],
                    "forecast_lat": c["lat"],
                    "forecast_lon": c["lon"],
                    "reference_lat": o["lat"],
                    "reference_lon": o["lon"],
                    "error_km": error,
                }
            )
        scores.append(
            {
                "model": model,
                "experiment": state,
                "n": len(errors),
                "mean_error_km": float(np.mean(errors)),
                "max_error_km": float(np.max(errors)),
            }
        )
    diag = read("diagnostics.json")
    steering = []
    for r in diag["steering"]:
        computed = decompose_steering(
            *[
                r[k]
                for k in [
                    "control_control",
                    "control_optimal",
                    "optimal_control",
                    "optimal_optimal",
                ]
            ]
        )
        for key, value in computed.items():
            np.testing.assert_allclose(value, r[key], atol=1e-10)
        steering.append(r)
    budget = []
    for r in read("vorticity_budget.json")["phases"]:
        if (
            r["level"] != "low_mean"
            or r["region"] != "eastern_core"
            or r["source"] == "era5t"
        ):
            continue
        growth = 0.5 * (r["error_rmse_end"] ** 2 - r["error_rmse_start"] ** 2)
        contributions = r["error_growth_projection"]
        np.testing.assert_allclose(
            growth,
            sum(v for k, v in contributions.items() if k != "actual"),
            atol=1e-10,
        )
        budget.append(
            {
                "forecast": r["source"],
                "start_hour": r["start_hour"],
                "end_hour": r["end_hour"],
                "K_change": growth,
                **{k: v for k, v in contributions.items() if k != "actual"},
                "mean_remainder_rms": r["mean_residual_rms"],
            }
        )
    write_csv(args.output / "track_scores.csv", scores)
    write_csv(args.output / "tracks.csv", positions)
    write_csv(args.output / "steering.csv", steering)
    write_csv(args.output / "vorticity_budget.csv", budget)
    write_csv(args.output / "early_budget.csv", diag["early_norms"])
    print(json.dumps(scores, indent=2))
    print("All archived distance, steering, and budget identities verified.")


if __name__ == "__main__":
    main()
