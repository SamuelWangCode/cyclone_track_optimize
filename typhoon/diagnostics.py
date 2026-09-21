"""Spherical circulation diagnostics in physical units."""

import numpy as np
from scipy.ndimage import gaussian_filter
from typhoon.tracks import great_circle_km

R = 6371000.0
OMEGA = 7.292115e-5


def raw_kinematics(u, v, lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    co = np.cos(phi)[:, None]
    dx = lambda a: np.gradient(a, lam, axis=-1, edge_order=2) / (R * co)
    dy = lambda a: np.gradient(a, phi, axis=-2, edge_order=2) / R
    zeta = dx(v) - dy(u * co) / co
    div = dx(u) + dy(v * co) / co
    f = 2 * OMEGA * np.sin(phi)[:, None]
    beta = 2 * OMEGA * co / R
    return dict(
        zeta=zeta,
        divergence=div,
        relative_advection=-u * dx(zeta) - v * dy(zeta),
        planetary_advection=-v * beta,
        stretching=-(zeta + f) * div,
    )


def smooth(a, lat, lon, km):
    sigma = (
        km / (111.195 * abs(lat[1] - lat[0])),
        km / (111.195 * abs(lon[1] - lon[0]) * np.cos(np.deg2rad(25))),
    )
    return gaussian_filter(a, sigma, mode="nearest")


def mean(a, lat, mask):
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    valid = mask & np.isfinite(a)
    return (
        float(np.sum(np.where(valid, a, 0) * w) / np.sum(w * valid))
        if valid.any()
        else None
    )


def steering(ds, center, levels=(850, 700, 500)):
    """Area-mean each above-ground pressure level, then pressure average."""
    dist = great_circle_km(
        center["lat"], center["lon"], ds.lat.values[:, None], ds.lon.values[None, :]
    )
    ring = (dist >= 300) & (dist <= 800)
    weights = {
        (850, 700, 500): np.array([75, 175, 100]) / 350,
        (850, 700): np.array([0.5, 0.5]),
    }[tuple(levels)]
    return {
        v: float(
            np.dot(
                weights,
                [
                    mean(
                        ds[v].sel(level=p).values,
                        ds.lat.values,
                        ring & (ds.sp.values > p * 100),
                    )
                    for p in levels
                ],
            )
        )
        for v in ["u", "v"]
    }


def decompose_steering(cc, co, oc, oo):
    """First letter: field; second: center (c=control, o=optimal)."""
    return {
        "field_symmetric": 0.5 * ((oc - cc) + (oo - co)),
        "position_symmetric": 0.5 * ((co - cc) + (oo - oc)),
        "total": oo - cc,
        "field_at_control_center": oc - cc,
        "field_at_optimal_center": oo - co,
        "position_in_control_field": co - cc,
        "position_in_optimal_field": oo - oc,
        "interaction": oo - oc - co + cc,
    }


def error_budget(error, tendencies, reference_tendencies, weights, dt=21600.0):
    """Exact discrete squared-error identity; tendencies are in s^-2."""
    weights = np.asarray(weights) / np.sum(weights)
    error = np.asarray(error)
    avg = lambda a: np.sum(a * weights, axis=(-2, -1))
    midpoint = (error[1:] + error[:-1]) / 2
    delta = {
        k: dt * (np.asarray(tendencies[k]) - np.asarray(reference_tendencies[k]))
        for k in tendencies
    }
    integrated = {k: (a[1:] + a[:-1]) / 2 for k, a in delta.items()}
    remainder = np.diff(error, axis=0) - sum(integrated.values())
    integrated["remainder"] = remainder
    return {
        "K": 0.5 * avg(error**2),
        "contributions": {k: avg(midpoint * a) for k, a in integrated.items()},
        "remainder_rms": np.sqrt(avg(remainder**2)) / dt,
    }
