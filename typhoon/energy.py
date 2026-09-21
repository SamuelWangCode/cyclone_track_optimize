"""Pressure-level perturbation total-energy weights in physical units (J/kg)."""

import numpy as np


def pressure_channel_index(channels, name):
    """Resolve TianXing's legacy 100 m / 100 hPa label collision explicitly."""
    matches = [i for i, value in enumerate(channels) if value == name]
    if len(matches) == 1:
        return matches[0]
    if (
        name in ["u100", "v100"]
        and list(channels[:8])
        == ["u10", "v10", "u100", "v100", "t2m", "sp", "msl", "tcwv"]
        and len(matches) == 2
        and matches[1] >= 8
    ):
        return matches[1]
    raise ValueError(f"Missing or ambiguous pressure-level channel: {name}")


def total_energy_coefficients(
    channels,
    levels_hpa,
    latitude,
    longitude,
    surface_pressure_pa,
    *,
    moist=True,
    reference_temperature=270.0,
    reference_pressure=100000.0,
):
    """Fixed background layer masses; truncated 50--1000 hPa column for TianXing.

    Moist TE requires explicit q channels in kg/kg. Relative humidity percent
    cannot substitute for q. Other channels have zero TE weight and must receive
    their own pointwise bounds. Reference scales and domain stay fixed across runs.
    """
    levels = np.asarray(levels_hpa, dtype=float) * 100
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    ps = np.asarray(surface_pressure_pa, dtype=float)
    if (
        not np.all(np.diff(levels) > 0)
        or not np.all(np.diff(lat) < 0)
        or ps.shape != (len(lat), len(lon))
        or not np.isfinite(ps).all()
        or np.any(ps <= 0)
    ):
        raise ValueError("Invalid pressure/latitude geometry")
    if not np.allclose(np.diff(lon), np.diff(lon)[0]):
        raise ValueError("Equal longitude spacing required")
    lat_edges = np.r_[
        min(90.0, lat[0] + (lat[0] - lat[1]) / 2),
        (lat[:-1] + lat[1:]) / 2,
        max(-90.0, lat[-1] - (lat[-2] - lat[-1]) / 2),
    ]
    area = np.abs(np.diff(np.sin(np.deg2rad(lat_edges))))[:, None] * np.ones(
        (1, len(lon))
    )
    area /= area.sum()
    edges = np.r_[levels[0], (levels[:-1] + levels[1:]) / 2, levels[-1]]
    dp = np.maximum(np.minimum(edges[1:, None, None], ps) - edges[:-1, None, None], 0.0)
    base = 0.5 * area[None] * dp / reference_pressure
    coefficients = np.zeros((len(channels), len(lat), len(lon)), dtype=np.float32)
    factors = {"u": 1.0, "v": 1.0, "t": 1004.0 / reference_temperature}
    if moist:
        factors["q"] = (2.5e6) ** 2 / (1004.0 * reference_temperature)
    for variable, factor in factors.items():
        for j, level in enumerate(levels_hpa):
            name = f"{variable}{level}"
            if name not in channels:
                raise ValueError(f"Total-energy channel missing: {name}")
            coefficients[pressure_channel_index(channels, name)] = base[j] * factor
    if "sp" not in channels:
        raise ValueError("TE surface pressure is sp, not mean sea-level pressure")
    coefficients[channels.index("sp")] = (
        0.5 * area * 287.04 * reference_temperature / reference_pressure**2
    )
    return coefficients[None]
