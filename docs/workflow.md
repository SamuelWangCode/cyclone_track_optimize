# Workflow and input contract

## Archived results

Run `python scripts/summarize.py` and `python scripts/plot_figures.py` from the repository root. Neither needs access to the research computer. The plotting functions use the packaged NetCDF and JSON files. Font metrics can differ from the manuscript exports; scientific arrays, units, limits, and vector scaling are preserved. DejaVu Sans avoids distributing proprietary font files.

The verification script recalculates every future track distance rather than averaging stored rounded scores. Missing centers would cause it to stop. It also checks the two steering-decomposition orders, their symmetric average, and the squared-vorticity-error budget identity.

## External initial and verifying fields

Supply `sfc.nc` and `pl.nc` in each initial-state folder. Coordinates are `lat`, `lon`, and `level`; latitude descends, longitude runs from 0 to 359.75°, and the global grid is 721 × 1440. Pressure levels are 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa.

Surface variables: `u10`, `v10`, `u100`, `v100`, `t2m`, `sp`, `msl`, `tcwv`. Pressure-level variables: `u`, `v`, `z`, `t`, `r`, `q`. Wind is in m s−1, temperature in K, pressure in Pa, geopotential in m² s−2, relative humidity in percent, and specific humidity in kg kg−1. The surface `u100` and `v100` names denote 100 m; `pressure_channel_index` explicitly distinguishes them from the 100 hPa tensor channels.

The background is IFS operational analysis, not an IFS forecast valid at a later lead. Verifying ERA5T is a separate six-hour analysis sequence from +6 through +168 h. The optimization target fields cover 0–55°N, 85–155°E, on the same 0.25° grid. Preserve `time` in target files and include both 850 and 700 hPa winds and surface pressure.

## Bounds and optimization

```bash
python scripts/prepare_bounds.py --initial inputs/ifs --spread inputs/eda --output inputs/bounds
python scripts/optimize.py --initial inputs/ifs --reference inputs/reference --bounds inputs/bounds/eda_bounds.nc --warm-start inputs/warm_start.npy --model-root inputs/tianxing --output runs/optimization
```

The EDA inputs contain standard deviations, not variances. The preparation code interpolates variance from 0.5° to 0.25°, periodically in longitude, and takes its square root. Pointwise bounds equal three standard deviations. Fixed global area and above-ground pressure thicknesses determine moist-energy weights. The energy ceiling is calculated from the one-spread uncertainty fields. The original IFS background remains fixed during all updates. The energy ceiling actually used is preserved in the configuration; it is not fitted to the final perturbation.

The target is equally weighted over 28 future six-hour times. At each time the objective compares smoothed mean 850/700 hPa relative vorticity in a fixed observed-center Gaussian window (500 km standard deviation, 1500 km cutoff). The tracked forecast center does not enter the gradient. `optimize.py` recomputes the entire adjoint chain and checks a directional derivative before updating the initial state.

The reported final run starts from an earlier feasible state, allows five iterations, and accepts two updates before stopping. Its exact starting array is `inputs/warm_start.npy` in the companion archive. The archive also contains `optimal_normalized.npy`, `optimizer_delta.npy`, the exported `delta.npy`, and the gradient check and update history. A new run from IFS is a different numerical experiment and should be reported as such. See `data/optimization.json` for the archived objective and feasibility values.

The TianXing directory must provide `weaformer2/backbone.py` and the compatible `assets/model_extra.pth`, `global_means.npy`, `global_stds.npy`, and `land_constant.nc`. They are external dependencies. The adapter uses the native geographic and time features and frozen model parameters. The archived optimization uses PyTorch 2.0.1 with CUDA 11.8; small changes can depend on software and hardware.

## Common physical state and regional retention

```bash
python scripts/prepare_initial.py --initial inputs/ifs --delta runs/optimization/delta_physical.npy --bounds inputs/bounds/eda_bounds.nc --output inputs/optimal
```

The physical correction is applied to the original IFS fields. Specific humidity for Pangu-Weather is the archived IFS specific humidity plus the humidity change consistently diagnosed from the temperature and relative-humidity increments. The export checks pointwise bounds and nonnegative specific humidity after float32 conversion.

Use `--region inside` or `--region outside` for the complementary retention forecasts. The fixed inclusive rectangle is 0–55°N, 90–170°E. Retained increments keep their original amplitude; the other increments are zero. The forecast model still runs globally.

## Six-hour forward forecasts

```bash
python scripts/forecast.py --model tianxing --model-path inputs/tianxing --initial inputs/optimal --output runs/fields/optimal_tianxing
python scripts/forecast.py --model pangu --model-path inputs/pangu_weather_6.onnx --initial inputs/optimal --output runs/fields/optimal_pangu_6h
```

Pangu requires the native **six-hour** model. Install an ONNX Runtime build matching the intended CPU or CUDA provider. The driver writes regional fields at all 29 times, including initialization, and applies the same causal circulation tracker. The default regional output is 5°S–55°N, 85–175°E. Tracking uses the archived 5–45°N, 95–140°E subdomain and its 25°N Gaussian zonal metric. The final tracked centers are also supplied for direct verification.

## Full-field diagnostics

`diagnose_budget.py`, `diagnose_advection.py`, and `diagnose_early.py` accept `--fields` and `--output`. Input folders are `control_tianxing`, `optimal_tianxing`, `era5init_tianxing`, `control_pangu_6h`, `optimal_pangu_6h`, `era5init_pangu_6h`, and `era5t`. Files are named `000.nc`, `006.nc`, …, `168.nc`. They contain regional `u`, `v`, `t`, `q`, `z`, and fixed initial `sp`; the early diagnostic also derives relative humidity. All fields use identical grids. Model normalization is removed before these calculations.

```bash
python scripts/diagnose_budget.py --fields runs/fields --output runs/budget
python scripts/diagnose_advection.py --fields runs/fields --output runs/budget
python scripts/diagnose_early.py --fields runs/fields --output runs/early
```

Raw spherical tendencies are calculated before 100 km smoothing. The budget uses six-hour midpoint errors and trapezoidal tendency integrals, with a diagnosed remainder. A negative regional advection contribution describes a reduction in the regional squared-error budget, including boundary transport. It is not an isolated latent-heating or network-internal tendency.

The small regional figure files under `data/` are subsets. Complete diagnostic input sequences are under `inputs/fields/` in the companion archive; use that path for `--fields` to recalculate the archived budgets. Those sequences retain the exact native model output values and grid used in the study. No model weights are needed to rerun these circulation diagnostics.

The release is checked by rerunning these three diagnostic scripts on the companion fields and comparing their numerical outputs with the original study archive. `data/verification.json` records the number of values compared and the largest difference. `data/archive_manifest.json` supplies file sizes and SHA-256 hashes for every large data file. `SHA256SUMS` covers the small repository snapshot.
