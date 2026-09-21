# Data and model sources

## Atmospheric fields

- **IFS operational analysis:** ECMWF, distributed through NSF NCAR GDEX dataset d113001, *ECMWF IFS High-Resolution Operational Forecasts*, [doi:10.5065/D68050ZV](https://doi.org/10.5065/D68050ZV). The study uses the `ec.oper.an` analysis products. The [dataset page](https://gdex.ucar.edu/datasets/d113001/) specifies CC BY 4.0. The initial analysis time is 2026-08-27 00 UTC. Initial fields are regridded to the models' common 0.25° grid.
- **ERA5T verification and initialization comparison:** Copernicus Climate Change Service (C3S), implemented by ECMWF, ERA5 hourly pressure-level and single-level data through the Climate Data Store: [pressure levels](https://doi.org/10.24381/cds.bd0915c6), [single levels](https://doi.org/10.24381/cds.adbb2d47). The study uses the preliminary ERA5T release available during the event. A later CDS request may return revised ERA5 values. The supplied derived diagnostics preserve the values used in the paper.
- **Uncertainty proxy:** corresponding ERA5 ensemble data assimilation spread products, interpolated in variance. These describe a constraint scale and are not an operational IFS analysis-error covariance.

Contains modified Copernicus Climate Change Service information and ECMWF information (2026). Regridding, forecast generation, increments, smoothing, and diagnostic calculations are performed by the study authors. Neither ECMWF nor the European Commission is responsible for use of these derived products. Source data retain their applicable [ECMWF/Copernicus terms](https://apps.ecmwf.int/datasets/licences/general/).

## Track reference

The continuous reference is the archived ATCF-format Saudel record `bwp172026.dat` retrieved from the third-party NaTyphoon archive via the Internet Archive. Its source URL and original checksum are retained in `data/observed_track.json`. It is a provisional archived track, not a final IBTrACS best track. Forecast skill is evaluated at the 28 six-hour valid times using the coordinates provided in `data/tracks.json`. ERA5T circulation centers are independently checked against the same reference.

## Models

- **TianXing:** Yuan, S., Wang, G., Mu, B., and Zhou, F. (2025), *TianXing: A Linear Complexity Transformer Model with Explicit Attention Decay for Global Weather Forecasting*, [doi:10.1007/s00376-024-3313-9](https://doi.org/10.1007/s00376-024-3313-9). The model implementation, trained weights, and normalization/constant assets must be obtained separately from the model authors. This repository supplies the study's adapter and numerical workflow. Compatible implementation and weight hashes are recorded in `data/optimization.json`; their inclusion does not redistribute the model.
- **Pangu-Weather:** Bi et al. (2023), *Accurate medium-range global weather forecasting with 3D neural networks*, [doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3). Obtain the native six-hour ONNX model from the [official repository](https://github.com/198808xc/Pangu-Weather) under its terms. No Pangu weights are included here.

## Mapping and software

Maps use Cartopy and Natural Earth coastlines. Colormaps use Matplotlib and `cmaps.sunshine_9lev`. Dependencies retain their own licenses; their source code, fonts, and model assets are not bundled. Requirements are listed in `requirements.txt`.
