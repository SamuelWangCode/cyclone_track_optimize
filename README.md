# Cyclone track optimization

Code and results for **Optimal initial conditions reveal the circulation errors behind Typhoon Saudel's missed turn**.

## Run

Use Python 3.10 or later. Extract `data/` and `inputs/` from the accompanying Zenodo package into this repository (the DOI will be added after deposit), then run:

```bash
python -m pip install -r requirements.txt
python scripts/summarize.py
python scripts/plot_figures.py
```

Results are saved in `tables/` and `figures/`. Optimization and forecast commands are described by `python scripts/optimize.py --help` and `python scripts/forecast.py --help`; model weights are obtained separately.
