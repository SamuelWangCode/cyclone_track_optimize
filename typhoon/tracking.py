"""Cyclone candidates from smoothed lower-tropospheric vorticity."""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def wind_candidates(ds, sp, lat, lon, maxima_window=3):
    if maxima_window < 3 or maxima_window % 2 != 1:
        raise ValueError("maxima_window must be an odd integer >= 3")
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[:, None]
    # Fixed 100 km Gaussian standard deviation; no tuning against these tracks.
    sigma = (
        100 / (111.195 * abs(lat[1] - lat[0])),
        100 / (111.195 * abs(lon[1] - lon[0]) * np.cos(np.deg2rad(np.median(lat)))),
    )
    smoothed = []
    coverage = []
    for level in [850, 700]:
        u = ds.u.sel(level=level).values
        v = ds.v.sel(level=level).values
        zeta = (np.gradient(v, lam, axis=1) - np.gradient(u * cosphi, phi, axis=0)) / (
            6371000 * cosphi
        )
        above = sp > level * 100
        den = gaussian_filter(above.astype(float), sigma, mode="nearest")
        num = gaussian_filter(np.where(above, zeta, 0), sigma, mode="nearest")
        smoothed.append(num / np.maximum(den, 1e-8))
        coverage.append(den)
    zeta = np.mean(smoothed, axis=0)
    valid = np.min(coverage, axis=0) >= 0.8
    maxima = (
        (zeta == maximum_filter(zeta, size=maxima_window, mode="nearest"))
        & valid
        & (zeta >= 1e-5)
    )
    maxima[:4] = False
    maxima[-4:] = False
    maxima[:, :4] = False
    maxima[:, -4:] = False
    result = []
    for i, j in np.argwhere(maxima):
        result.append(
            {
                "lat": float(lat[i]),
                "lon": float(lon[j]),
                "smoothed_mean_850_700_vorticity_s1": float(zeta[i, j]),
            }
        )
    return result
