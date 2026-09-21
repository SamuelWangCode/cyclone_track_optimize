"""Run native six-hour TianXing or Pangu-Weather forecasts from supplied fields."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typhoon.fields import read_pair, tianxing_state, pangu_state, LEVELS, CHANNELS
from typhoon.energy import pressure_channel_index
from typhoon.circulation_tracker import CirculationTracker
from typhoon.tracking import wind_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tianxing", "pangu"], required=True)
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="TianXing implementation folder or native six-hour Pangu ONNX file",
    )
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--init", default="2026-08-27T00:00:00")
    p.add_argument(
        "--seed", nargs=2, type=float, default=[28.6, 124.9], metavar=("LAT", "LON")
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Choose an empty output folder")
    init = datetime.fromisoformat(args.init)
    pl, sfc = read_pair(args.initial, init)
    lat, lon = sfc.lat.values, sfc.lon.values
    iy = np.where((lat >= -5) & (lat <= 55))[0]
    ix = np.where((lon >= 85) & (lon <= 175))[0]
    sy = slice(iy[0], iy[-1] + 1)
    sx = slice(ix[0], ix[-1] + 1)
    ps = sfc.sp.values[sy, sx].copy()
    levels = [925, 850, 700, 500, 300]
    if args.model == "tianxing":
        import torch
        from typhoon.tianxing import TianXing

        model = TianXing(model_root=args.model_path, device=args.device)
        state = model.normalize(tianxing_state(pl, sfc))
    else:
        import onnxruntime as ort

        providers = (
            ["CPUExecutionProvider"]
            if args.device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model = ort.InferenceSession(str(args.model_path), providers=providers)
        state, surface = pangu_state(pl, sfc)
    tracker = CirculationTracker(args.seed)
    rows = []
    for h in range(0, 169, 6):
        valid = init + timedelta(hours=h)
        if args.model == "tianxing":
            if h:
                with torch.no_grad():
                    state = model(state, valid).detach()
            physical = model.physical(state)[0, :, sy, sx].detach().cpu().numpy()
            data = {
                v: np.stack(
                    [
                        physical[pressure_channel_index(CHANNELS, f"{v}{p}")]
                        for p in levels
                    ]
                )
                for v in ["u", "v", "z", "t", "r"]
            }
            from typhoon.moist_energy import specific_humidity

            data["q"] = specific_humidity(
                torch.as_tensor(data["t"]),
                torch.as_tensor(data["r"]),
                torch.as_tensor(np.array(levels)[:, None, None] * 100),
            ).numpy()
            msl = physical[6]
        else:
            if h:
                values = model.run(None, {"input": state, "input_surface": surface})
                state, surface = values[0], values[1]
            data = {
                v: np.stack(
                    [state[j, list(reversed(LEVELS)).index(p), sy, sx] for p in levels]
                )
                for j, v in enumerate(["z", "q", "t", "u", "v"])
            }
            msl = surface[0, sy, sx]
        ds = xr.Dataset(
            {v: (("level", "lat", "lon"), a) for v, a in data.items()},
            coords={"level": levels, "lat": lat[sy], "lon": lon[sx]},
        )
        ds["sp"] = (("lat", "lon"), ps)
        ds["msl"] = (("lat", "lon"), msl)
        ds.attrs.update(
            valid_time=valid.isoformat(),
            lead_hours=h,
            model=args.model,
            surface_pressure_mask="Fixed initial field",
            step_hours=6,
        )
        tracking_fields = ds.sel(lat=slice(45, 5), lon=slice(95, 140))
        candidates = wind_candidates(
            tracking_fields,
            tracking_fields.sp.values,
            tracking_fields.lat.values,
            tracking_fields.lon.values,
            maxima_window=3,
        )
        center, status = tracker.locate(candidates, h)
        rows.append(
            {
                "lead_hours": h,
                "valid_time": valid.isoformat(),
                "center": center,
                "status": status,
            }
        )
        ds.to_netcdf(
            args.output / f"{h:03d}.nc",
            encoding={v: {"zlib": True, "complevel": 3} for v in ds.data_vars},
        )
        (args.output / "track.json").write_text(json.dumps(rows, indent=2))
        print(h, center, flush=True)


if __name__ == "__main__":
    main()
