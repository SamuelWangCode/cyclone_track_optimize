"""Constrained descent with a recomputed full-chain discrete adjoint."""

from dataclasses import dataclass
import torch


@dataclass
class EnergyBoxConstraint:
    """Intersection of a physical quadratic energy budget and spatial box bounds.

    energy_coefficients include physical units, area/layer weights and, when
    optimizing normalized states, the square of the model's channel scales.
    Zero energy coefficients do not freeze auxiliary variables: their pointwise
    bounds still apply. Projection is a feasible retraction, not an exact nearest
    Euclidean projection onto the intersection.
    """

    background: torch.Tensor
    scale: torch.Tensor
    energy_coefficients: torch.Tensor
    energy_limit: float
    max_increment: torch.Tensor
    control_mask: torch.Tensor | None = None
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None

    def __post_init__(self):
        self.background = self.background.detach().clone()

        def expand(value):
            return torch.broadcast_to(
                torch.as_tensor(
                    value, device=self.background.device, dtype=self.background.dtype
                ),
                self.background.shape,
            ).detach()

        self.scale = expand(self.scale)
        self.energy_coefficients = expand(self.energy_coefficients)
        self.max_increment = expand(self.max_increment)
        if not all(
            torch.isfinite(x).all()
            for x in [
                self.background,
                self.scale,
                self.energy_coefficients,
                self.max_increment,
            ]
        ):
            raise ValueError("Non-finite energy or box specification")
        if (
            (self.scale <= 0).any()
            or (self.energy_coefficients < 0).any()
            or not (self.energy_coefficients > 0).any()
            or (self.max_increment < 0).any()
            or not 0 < self.energy_limit < float("inf")
        ):
            raise ValueError("Invalid energy or box specification")
        self.control_mask = (
            torch.ones_like(self.background, dtype=torch.bool)
            if self.control_mask is None
            else expand(self.control_mask).bool()
        )
        self.control_mask = self.control_mask & (self.max_increment > 0)
        self.weights = self.control_mask.to(self.background.dtype)
        for name in ["lower", "upper"]:
            if getattr(self, name) is not None:
                setattr(self, name, expand(getattr(self, name)))
        if self.lower is not None and (self.background < self.lower).any():
            raise ValueError("Background violates lower bound")
        if self.upper is not None and (self.background > self.upper).any():
            raise ValueError("Background violates upper bound")

    def energy(self, x):
        return (self.energy_coefficients * (x - self.background).square()).sum(
            dtype=torch.float64
        )

    def norm(self, x):
        return (self.energy(x) / self.energy_limit).sqrt()

    @torch.no_grad()
    def project(self, x):
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite candidate")
        delta = torch.maximum(
            torch.minimum(x - self.background, self.max_increment), -self.max_increment
        )
        delta = torch.where(self.control_mask, delta, torch.zeros_like(delta))
        candidate = self.background + delta
        if self.lower is not None:
            candidate = torch.maximum(candidate, self.lower)
        if self.upper is not None:
            candidate = torch.minimum(candidate, self.upper)
        ratio = self.norm(candidate)
        factor = torch.clamp(
            1 / ratio.clamp_min(torch.finfo(ratio.dtype).tiny), max=1.0
        ).to(candidate.dtype)
        return self.background + factor * (candidate - self.background)


def rollout_loss(step, x0, targets, loss_fn, time_weights=None):
    weights = time_weights or [1.0] * len(targets)
    denom = sum(w for w, t in zip(weights, targets) if t is not None)
    if denom <= 0:
        raise ValueError("No positive-weight verification targets")
    x = x0
    loss = x0.new_zeros(())
    for t, target in enumerate(targets):
        x = step(x, t)
        if target is not None:
            loss = loss + weights[t] * loss_fn(x, target) / denom
    return loss


def discrete_adjoint(step, x0, targets, loss_fn, time_weights=None):
    """One-step graph recomputation; every adjoint term propagated, no truncation.

    CPU checkpoints trade host RAM for constant GPU graph memory. The step must
    be deterministic and model weights/forcings identical in both sweeps.
    """
    weights = time_weights or [1.0] * len(targets)
    if len(weights) != len(targets) or any(w < 0 for w in weights):
        raise ValueError("Invalid time weights")
    denom = sum(w for w, t in zip(weights, targets) if t is not None)
    if denom <= 0:
        raise ValueError("No positive-weight verification targets")
    states = []
    x = x0.detach()
    with torch.no_grad():
        for t in range(len(targets)):
            states.append(x.cpu())
            x = step(x, t).detach()
    adjoint = None
    value = 0.0
    for t in reversed(range(len(targets))):
        x = states[t].to(x0.device).detach().requires_grad_(True)
        y = step(x, t)
        objective = y.new_zeros(())
        if targets[t] is not None:
            term = weights[t] * loss_fn(y, targets[t]) / denom
            value += term.detach().item()
            objective = objective + term
        if adjoint is not None:
            objective = objective + (y * adjoint).sum()
        if objective.requires_grad:
            adjoint = torch.autograd.grad(objective, x)[0].detach()
        else:
            adjoint = torch.zeros_like(x)
        states[t] = None
    return value, adjoint


def fit_initial_state(
    step,
    constraint,
    targets,
    loss_fn,
    iterations=25,
    lr=0.5,
    time_weights=None,
    warm_start=None,
    max_backtracks=8,
    callback=None,
):
    """Feasible descent with post-projection loss evaluation and best-state return.

    Step in dimensionless uncertainty coordinates. Neither continuation nor a
    warm start can move the original background or enlarge its feasible set.
    """
    x = constraint.project(constraint.background if warm_start is None else warm_start)
    with torch.no_grad():
        best_loss = float(rollout_loss(step, x, targets, loss_fn, time_weights))
    best = x.detach().clone()
    history = []
    for it in range(iterations):
        loss, grad = discrete_adjoint(step, x, targets, loss_fn, time_weights)
        if not torch.isfinite(grad).all():
            raise ValueError("Non-finite adjoint")
        direction = grad * constraint.scale
        direction = torch.where(
            constraint.weights > 0, direction, torch.zeros_like(direction)
        )
        denom = direction.square().mean().sqrt()
        if float(denom) == 0:
            break
        direction = direction / denom
        accepted = False
        trial_lr = lr
        for _ in range(max_backtracks):
            candidate = constraint.project(x - trial_lr * constraint.scale * direction)
            with torch.no_grad():
                trial_loss = float(
                    rollout_loss(step, candidate, targets, loss_fn, time_weights)
                )
            if trial_loss < loss:
                x = candidate.detach()
                accepted = True
                if trial_loss < best_loss:
                    best_loss, best = trial_loss, x.clone()
                break
            trial_lr *= 0.5
        entry = {
            "iteration": it,
            "loss_before": loss,
            "loss_after": trial_loss,
            "accepted": accepted,
            "step_size": trial_lr,
            "constraint_norm": float(constraint.norm(x)),
            "best_loss": best_loss,
        }
        history.append(entry)
        if callback:
            callback(entry, best)
        if not accepted:
            break
    return best, history
