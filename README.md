# Cyclone track optimization

Code and derived result data for **Optimal initial conditions reveal the circulation errors behind Typhoon Saudel's missed turn**.

The case starts at 00 UTC on 27 August 2026 and covers 168 hours. TianXing is used to solve for optimal initial conditions. Pangu-Weather uses those physical initial conditions for six-hour forward forecasts without further optimization. The analysis connects the initial environmental structure, early vorticity transport and ridge evolution, accumulated storm displacement, and the later return flow.

## Reproduce the reported results

Use Python 3.10 or later in a separate environment:

```bash
python -m pip install -r requirements.txt
python scripts/summarize.py
python scripts/plot_figures.py
python -m pytest -q
```

The first command script recomputes the track scores from the archived coordinates and verifies steering and vorticity-budget identities. The second generates all four main figures and eight supporting figures from the supplied data. Cartopy downloads Natural Earth coastlines on first use. Output folders are `tables/` and `figures/`.

| Mean track error over 28 future times (km) | TianXing | Pangu-Weather |
|---|---:|---:|
| IFS-initialized control | 367.3 | 408.5 |
| Optimal initial conditions | 74.1 | 122.4 |
| ERA5T-initialized forecast | 393.1 | 415.4 |
| Perturbation retained within East Asia–western North Pacific | 124.7 | 127.4 |
| Perturbation retained outside that region | 277.3 | 358.9 |

## Contents

- `typhoon/`: differentiable objective, moist-energy and pointwise constraints, adjoint updates, model interface, tracking, and circulation diagnostics.
- `scripts/`: input preparation, optimization, six-hour forecasts, diagnostics, tables, and figures.
- `config/`: numerical settings for the reported case.
- `data/`: archived tracks, numerical diagnostics, and losslessly compressed figure-source fields.
- `tables/`: plain CSV exports of the principal results.
- `docs/`: input requirements, source attribution, and figure-to-data mapping.

## Forecast integrations and archive coverage

The supplied data reproduce the displayed figures and principal numerical summaries without model weights. Full forecast integrations additionally require external model implementations, weights, original IFS fields, verifying ERA5T fields, and uncertainty fields. See [the workflow](docs/workflow.md).

Large simulation inputs are distributed separately in the companion archive, `initial_conditions_and_forecasts.zip`: the original IFS state, full optimal physical initial conditions, the preceding optimization starting state, actual and optimizer increments, gridded bounds, verifying ERA5T targets, and the complete regional six-hour field sequences used for the circulation diagnostics. Extract its `inputs/` directory at the repository root. These files are included in the prepared Zenodo deposit package and are not stored in Git. A persistent download link will be added after the authors deposit that package.

The numerical core is tested with small synthetic cases; the cleaned forecast launchers require the external assets and are not a new rerun of the global model integrations. The recorded run settings and model fingerprints are in `data/optimization.json` and `config/optimization.json`.

## Rights and citation

The authors retain copyright. No open-source license is granted; viewing and local scholarly reproduction of the reported results are permitted under [RIGHTS.md](RIGHTS.md). Third-party materials retain their existing terms, listed in [sources.md](docs/sources.md). Use `CITATION.cff` to cite this version and identify its Git commit. A Zenodo DOI will be added after deposit.
