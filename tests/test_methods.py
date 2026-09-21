import numpy as np
import torch
from typhoon.optimize import discrete_adjoint, rollout_loss, EnergyBoxConstraint
from typhoon.energy import total_energy_coefficients, pressure_channel_index
from typhoon.fields import CHANNELS, LEVELS
from typhoon.moist_energy import specific_humidity, TianXingMoistEnergyConstraint
from typhoon.objectives import CirculationOperator
from typhoon.tracks import great_circle_km
from typhoon.diagnostics import error_budget, decompose_steering, raw_kinematics


def test_recomputed_adjoint_matches_full_graph():
    x = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64, requires_grad=True)
    step = lambda z, k: torch.sin(z) * (1.01 + 0.002 * k) + 0.01
    targets = [torch.zeros_like(x) for _ in range(28)]
    loss = lambda z, y: ((z - y) ** 2).mean()
    full = rollout_loss(step, x, targets, loss)
    gradient = torch.autograd.grad(full, x)[0]
    value, adjoint = discrete_adjoint(step, x, targets, loss)
    np.testing.assert_allclose(value, full.item(), rtol=1e-12)
    torch.testing.assert_close(adjoint, gradient, rtol=1e-12, atol=1e-12)


def test_vorticity_solid_body_rotation():
    lat = np.linspace(35, 15, 101)
    lon = np.linspace(120, 140, 101)
    rate = 1e-5
    u = rate * 6371000 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    expected = 2 * rate * np.sin(np.deg2rad(lat))[:, None] * np.ones_like(u)
    result = raw_kinematics(u, np.zeros_like(u), lat, lon)["zeta"]
    np.testing.assert_allclose(result[2:-2, 2:-2], expected[2:-2, 2:-2], rtol=1e-5)


def test_circulation_operator_gradient():
    lat = np.linspace(30, 20, 12)
    lon = np.linspace(120, 130, 13)
    op = CirculationOperator(lat, lon, np.ones((2, 12, 13)))
    torch.manual_seed(2)
    u = torch.randn(2, 12, 13, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(u)
    direction = torch.randn_like(u)
    loss = (op(u, v) / 1e-5).square().mean()
    analytic = (torch.autograd.grad(loss, u)[0] * direction).sum()
    eps = 1e-5
    numeric = (
        ((op(u + eps * direction, v) / 1e-5) ** 2).mean()
        - ((op(u - eps * direction, v) / 1e-5) ** 2).mean()
    ) / (2 * eps)
    torch.testing.assert_close(analytic, numeric, rtol=1e-7, atol=1e-8)


def test_moist_constraint_and_fixed_fields():
    lat = np.array([30.0, 20.0, 10.0])
    lon = np.array([120.0, 130.0, 140.0, 150.0])
    ps = np.full((3, 4), 95000.0)
    base = torch.zeros((1, 73, 3, 4), dtype=torch.float64)
    for p in LEVELS:
        base[:, CHANNELS.index(f"t{p}")] = 270
        base[:, CHANNELS.index(f"r{p}")] = 50
    base[:, 5] = 95000
    control = torch.ones_like(base, dtype=torch.bool)
    control[:, [2, 3, 5, 7]] = False
    constraint = TianXingMoistEnergyConstraint(
        base,
        1.0,
        0.2,
        0.001,
        channels=CHANNELS,
        levels_hpa=LEVELS,
        latitude=lat,
        longitude=lon,
        surface_pressure_pa=ps,
        control_mask=control,
    )
    result = constraint.project(base + 0.4)
    assert float(constraint.energy(base)) == 0
    assert float(constraint.energy(result)) <= 0.001
    assert torch.all((result - base).abs() <= 0.2 + 1e-9)
    assert torch.equal(result[:, [2, 3, 5, 7]], base[:, [2, 3, 5, 7]])
    assert pressure_channel_index(CHANNELS, "u100") != 2


def test_budget_and_symmetric_steering():
    rng = np.random.default_rng(3)
    error = rng.normal(size=(29, 4, 5))
    tend = {
        k: rng.normal(size=(29, 4, 5)) * 1e-6
        for k in ["relative", "planetary", "stretching"]
    }
    result = error_budget(
        error, tend, {k: np.zeros_like(a) for k, a in tend.items()}, np.ones((4, 5))
    )
    np.testing.assert_allclose(
        np.diff(result["K"]), sum(result["contributions"].values()), atol=1e-14
    )
    d = decompose_steering(-0.63, 1.30, -2.84, 1.60)
    np.testing.assert_allclose(
        d["position_symmetric"] + d["field_symmetric"], d["total"], atol=1e-14
    )
    np.testing.assert_allclose(d["position_symmetric"], 3.185)


def test_distance_and_humidity_units():
    np.testing.assert_allclose(great_circle_km(0, 0, 0, 1), 111.1950802335, rtol=1e-10)
    q = specific_humidity(
        torch.tensor(273.16), torch.tensor(100.0), torch.tensor(100000.0)
    )
    assert 0.0037 < float(q) < 0.0039
