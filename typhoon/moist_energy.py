"""Moist total-energy constraint for TianXing's native temperature/RH state."""

import numpy as np
import torch
from typhoon.energy import total_energy_coefficients, pressure_channel_index
from typhoon.optimize import EnergyBoxConstraint


def specific_humidity(temperature_k, relative_humidity_percent, pressure_pa):
    """ECMWF mixed-phase saturation convention, RH in %, q in kg/kg.

    https://metview.readthedocs.io/en/5.25.0/api/functions/specific_humidity_from_relative_humidity.html
    No clipping: physical bounds belong to the optimization constraints.
    """
    t = temperature_k.double()
    r = relative_humidity_percent.double()
    water = 611.21 * torch.exp(17.502 * (t - 273.16) / (t - 32.19))
    ice = 611.21 * torch.exp(22.587 * (t - 273.16) / (t + 0.7))
    liquid_fraction = ((t - 250.16) / (273.16 - 250.16)).clamp(0, 1).square()
    vapour = (liquid_fraction * water + (1 - liquid_fraction) * ice) * r / 100
    epsilon = 0.621981
    return epsilon * vapour / (pressure_pa - (1 - epsilon) * vapour)


class TianXingMoistEnergyConstraint(EnergyBoxConstraint):
    """Dry terms plus Lv^2/(cp Tr) times the actual diagnosed q increment.

    Baseline and candidate q use the same T/RH conversion, so the unmodified
    IFS input has exactly zero perturbation energy. Archived IFS q is not mixed
    with reconstructed candidate q. The model continues to receive native RH.
    """

    def __init__(
        self,
        background,
        scale,
        max_increment,
        energy_limit,
        *,
        channels,
        levels_hpa,
        latitude,
        longitude,
        surface_pressure_pa,
        model_mean=0.0,
        model_std=1.0,
        q_max_increment=None,
        archived_background_q=None,
        **bounds,
    ):
        self.temperature_indices = [channels.index(f"t{p}") for p in levels_hpa]
        self.humidity_indices = [channels.index(f"r{p}") for p in levels_hpa]

        def tensor(x):
            return torch.as_tensor(x, device=background.device, dtype=background.dtype)

        self.model_mean = tensor(model_mean)
        self.model_std = tensor(model_std)
        if (
            not torch.isfinite(self.model_mean).all()
            or not torch.isfinite(self.model_std).all()
            or (self.model_std <= 0).any()
        ):
            raise ValueError("Invalid model normalization")
        dry = tensor(
            total_energy_coefficients(
                channels,
                levels_hpa,
                latitude,
                longitude,
                surface_pressure_pa,
                moist=False,
            )
        )
        self.humidity_coefficients = (
            dry[:, self.temperature_indices].double() * ((2.5e6) / 1004.0) ** 2
        )
        self.pressure_pa = tensor(
            np.asarray(levels_hpa)[None, :, None, None] * 100
        ).double()
        control = torch.ones_like(background, dtype=torch.bool)
        for level in levels_hpa:
            above = tensor(np.asarray(surface_pressure_pa) >= level * 100).bool()
            for variable in ["u", "v", "t", "z", "r"]:
                if f"{variable}{level}" in channels:
                    control[
                        :, pressure_channel_index(channels, f"{variable}{level}")
                    ] = above
        control &= torch.as_tensor(
            bounds.pop("control_mask", True), device=background.device, dtype=torch.bool
        )
        super().__init__(
            background,
            scale,
            dry * self.model_std.square(),
            energy_limit,
            max_increment,
            control_mask=control,
            **bounds,
        )
        self.background_q = self.humidity(self.background).detach()
        if not torch.isfinite(self.background_q).all():
            raise ValueError("Non-finite background humidity")
        self.q_max_increment = (
            None if q_max_increment is None else tensor(q_max_increment).double()
        )
        self.archived_background_q = (
            None
            if archived_background_q is None
            else tensor(archived_background_q).double()
        )
        if self.q_max_increment is not None and (
            (self.q_max_increment < 0).any()
            or not torch.isfinite(self.q_max_increment).all()
        ):
            raise ValueError("Invalid specific-humidity bounds")
        if self.archived_background_q is not None and (
            (self.archived_background_q < 0).any()
            or not torch.isfinite(self.archived_background_q).all()
        ):
            raise ValueError("Invalid archived baseline specific humidity")

    def humidity(self, x):
        # Convert selected channels only; keep the normalization graph intact.
        mean = torch.broadcast_to(self.model_mean, x.shape)
        std = torch.broadcast_to(self.model_std, x.shape)
        ti = self.temperature_indices
        ri = self.humidity_indices
        t = x[:, ti].double() * std[:, ti].double() + mean[:, ti].double()
        r = x[:, ri].double() * std[:, ri].double() + mean[:, ri].double()
        return specific_humidity(t, r, self.pressure_pa)

    def energy(self, x):
        dq = self.humidity(x) - self.background_q
        return super().energy(x) + (self.humidity_coefficients * dq.square()).sum()

    @torch.no_grad()
    def project(self, x):
        candidate = super().project(x)
        # q(T,RH) is nonlinear: the quadratic contraction must be checked again.
        # Backtracking toward the same background retains every pointwise bound.
        for _ in range(64):
            dq = self.humidity(candidate) - self.background_q
            bad = torch.zeros_like(dq, dtype=torch.bool)
            if self.q_max_increment is not None:
                bad |= dq.abs() > self.q_max_increment
            if self.archived_background_q is not None:
                bad |= self.archived_background_q + dq < 0
            if bad.any():
                # Contract T and RH together only in cells that violate the
                # shared q bound. Keep the original background and all boxes.
                for indices in [self.temperature_indices, self.humidity_indices]:
                    values = candidate[:, indices]
                    base = self.background[:, indices]
                    candidate[:, indices] = torch.where(
                        bad, base + 0.5 * (values - base), values
                    )
                continue
            value = self.energy(candidate)
            if torch.isfinite(value) and value <= self.energy_limit:
                return candidate
            candidate = self.background + 0.5 * (candidate - self.background)
        raise RuntimeError("Wet-energy contraction did not find a feasible state")
