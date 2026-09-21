"""Differentiable circulation objective with fixed verification geometry."""

import torch


class CirculationOperator:
    """Differentiable spherical vorticity and fixed 100 km Gaussian smoothing.

    Inputs have shape [2, latitude, longitude] ordered 850, 700 hPa. Terrain
    masks and Gaussian kernels are fixed target geometry, never hard centers.
    """

    def __init__(
        self,
        latitude,
        longitude,
        above_ground,
        device="cpu",
        sigma_km=100.0,
        metric_latitude=25.0,
    ):
        import numpy as np

        self.phi = torch.as_tensor(
            np.deg2rad(np.asarray(latitude).copy()), device=device, dtype=torch.float64
        )
        self.lam = torch.as_tensor(
            np.deg2rad(np.asarray(longitude).copy()), device=device, dtype=torch.float64
        )
        self.cos = self.phi.cos()[None, :, None]
        self.mask = torch.as_tensor(above_ground, device=device, dtype=torch.float64)
        if self.mask.shape != (2, len(latitude), len(longitude)):
            raise ValueError("Wrong circulation-mask shape")
        scales = [
            sigma_km / (111.195 * abs(latitude[1] - latitude[0])),
            sigma_km
            / (
                111.195
                * abs(longitude[1] - longitude[0])
                * np.cos(np.deg2rad(metric_latitude))
            ),
        ]
        self.kernels = []
        for sigma in scales:
            radius = int(4 * sigma + 0.5)
            x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            self.kernels.append(kernel / kernel.sum())
        self.coverage = self.smooth(self.mask)
        self.valid = self.coverage.min(dim=0).values >= 0.8

    def smooth(self, field):
        import torch.nn.functional as F

        x = field[:, None]
        for axis, k in enumerate(self.kernels):
            r = len(k) // 2
            padding = (0, 0, r, r) if axis == 0 else (r, r, 0, 0)
            kernel = k[None, None, :, None] if axis == 0 else k[None, None, None, :]
            x = F.conv2d(F.pad(x, padding, mode="replicate"), kernel)
        return x[:, 0]

    def __call__(self, u, v):
        u = u.double()
        v = v.double()
        dvdlam = torch.gradient(v, spacing=(self.lam,), dim=(2,))[0]
        ducosdphi = torch.gradient(u * self.cos, spacing=(self.phi,), dim=(1,))[0]
        zeta = (dvdlam - ducosdphi) / (6371000 * self.cos)
        return (self.smooth(zeta * self.mask) / self.coverage.clamp_min(1e-8)).mean(
            dim=0
        )


def circulation_pattern_loss(predicted, reference, weights, vorticity_scale_s1=1e-5):
    weights = weights.double()
    weights = weights / weights.sum()
    return (
        weights
        * ((predicted.double() - reference.double()) / vorticity_scale_s1).square()
    ).sum()


def reference_window_weights(
    latitude, longitude, center_lat, center_lon, length_scale_km=500.0, cutoff_km=1500.0
):
    """Fixed reference-centered window; L is a distance, not the EDA sigma."""
    import numpy as np
    from typhoon.tracks import great_circle_km

    if not 0 < length_scale_km <= cutoff_km:
        raise ValueError("Invalid target-window radii")
    yy, xx = np.meshgrid(latitude, longitude, indexing="ij")
    distance = great_circle_km(center_lat, center_lon, yy, xx)
    weights = (
        np.exp(-0.5 * (distance / length_scale_km) ** 2)
        * (distance <= cutoff_km)
        * np.cos(np.deg2rad(yy))
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Empty reference target window")
    return weights
