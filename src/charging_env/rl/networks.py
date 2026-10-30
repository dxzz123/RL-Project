"""Masked pricing/dispatch policies and separate centralized value networks.

The dispatch policy consumes only ``ChargingSimulator.local_observation``.
Each policy/value instance owns its encoder; no optimizer can accidentally
update another policy's representation. Request lists are padded per batch,
never capped, and pooled with a padding mask and request-order features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from ..config import Scenario
from ..simulator import Mode


def _linear(input_size: int, output_size: int, gain: float = math.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(input_size, output_size)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class _ObservationEncoder(nn.Module):
    """Normalize physical units and encode every request, including its order.

    Global observations include fleet-wide action masks. Local observations
    include the acting vehicle's mask and explicitly supplied peer state. An
    agent id aligns eligibility rows with own/peer state. By default it is
    omitted from the features, preserving the legacy shared policy. The
    optional local-only identity feature distinguishes roles in the simulator's
    fixed vehicle allocation order without adding any observations.
    """

    def __init__(self, scenario: Scenario, hidden_size: int, *, local: bool,
                 include_agent_id: bool = False, layer_norm: bool = False):
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not isinstance(include_agent_id, bool):
            raise ValueError("include_agent_id must be a bool")
        if include_agent_id and not local:
            raise ValueError("include_agent_id is only supported for local dispatch observations")
        self.scenario = scenario
        self.local = local
        self.include_agent_id = include_agent_id
        c = scenario
        z, m, f = c.n_zones, c.n_mcv, c.n_fcs
        count_scale = max(1.0, float(np.max(c.resolved_arrival_rates)))
        energy_scale = count_scale * c.request_max_kwh
        price_scale = max(c.price_max, float(np.max(c.resolved_fcs_charge_price_profile)),
                          float(np.max(c.resolved_fcs_discharge_price_profile)),
                          max(c.grid_buy_per_kwh), 1e-6)
        self.count_scale = count_scale
        self.specs = (
            ("arrival_rates", z, count_scale),
            ("requests_by_zone", z, count_scale),
            ("potential_energy_kwh", z, energy_scale),
            ("mobile_prices", z, price_scale),
            ("mobile_demand", z, count_scale),
            ("fixed_demand", f, count_scale),
            ("mobile_requested_kwh", z, energy_scale),
            ("fixed_requested_kwh", f, energy_scale),
            ("outside_requests", 1, count_scale * z),
            ("mobile_wait_hours", z, c.period_hours),
            ("mobile_available", z, 1.0),
            ("fcs_wait_hours", f, c.period_hours),
            ("fcs_retail_prices", f, price_scale),
            ("fcs_discharge_prices", f, price_scale),
            ("grid_buy_price", 1, price_scale),
            ("pricing_mask", z * c.n_price_levels, 1.0),
        )
        # Period, stage, fleet positions/energy, eligibility, action masks.
        dense_size = (1 + 3 + m * (z + 1) + m * z
                      + (1 if local else m) * len(Mode) * z
                      + sum(size for _, size, _ in self.specs)
                      + (m if include_agent_id else 0))
        # Origin one-hot, normalized energy, choice one-hot (unresolved=-1,
        # mobile=0, stations=1..F, outside=F+1), access flags, row position.
        self.request_size = z + 1 + (f + 3) + f + 1
        self.request_encoder = nn.Sequential(
            _linear(self.request_size, hidden_size), nn.Tanh(),
            _linear(hidden_size, hidden_size), nn.Tanh(),
        )
        if not isinstance(layer_norm, bool):
            raise ValueError('layer_norm must be boolean')
        body = [_linear(dense_size + (z + 1) * hidden_size, hidden_size)]
        if layer_norm:
            body.append(nn.LayerNorm(hidden_size))
        body.extend([nn.Tanh(), _linear(hidden_size, hidden_size)])
        if layer_norm:
            body.append(nn.LayerNorm(hidden_size))
        body.append(nn.Tanh())
        self.body = nn.Sequential(*body)

    def _dense(self, obs: dict) -> np.ndarray:
        c = self.scenario
        locations = np.asarray(obs["locations"], dtype=np.int64).reshape(-1)
        energy = np.asarray(obs["energy_kwh"], dtype=np.float32).reshape(-1)
        eligibility = np.asarray(obs["eligible_mcvs"], dtype=np.float32)
        if self.local:
            if locations.size != 1 or energy.size != 1:
                raise ValueError("DispatchActor requires local_observation(m), not a global observation")
            locations = np.concatenate((locations, np.asarray(obs["other_locations"], dtype=np.int64)))
            energy = np.concatenate((energy, np.asarray(obs["other_energy_kwh"], dtype=np.float32)))
            agent_id = int(np.asarray(obs["agent_id"]).reshape(-1)[0])
            order = [agent_id] + [i for i in range(c.n_mcv) if i != agent_id]
            eligibility = eligibility[order]
        if locations.size != c.n_mcv or energy.size != c.n_mcv:
            raise ValueError("Observation fleet size does not match the scenario")
        result = [
            np.asarray(obs["period"], dtype=np.float32).reshape(-1) / c.horizon,
            np.eye(3, dtype=np.float32)[int(obs["stage"])],
            np.eye(c.n_zones, dtype=np.float32)[locations].reshape(-1),
            energy / c.battery_capacity_kwh,
            eligibility.reshape(-1),
            np.asarray(obs["action_mask"], dtype=np.float32).reshape(-1),
        ]
        for key, size, scale in self.specs:
            value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if value.size != size:
                raise ValueError(f"Observation {key!r} has {value.size} elements; expected {size}")
            result.append(value / scale)
        if self.include_agent_id:
            result.append(np.eye(c.n_mcv, dtype=np.float32)[agent_id])
        return np.concatenate(result)

    def forward(self, observations: Sequence[dict]) -> Tensor:
        if len(observations) == 0:
            raise ValueError("An observation batch cannot be empty")
        reference = next(self.parameters())
        dtype, device = reference.dtype, reference.device
        dense = torch.as_tensor(np.stack([self._dense(obs) for obs in observations]),
                                dtype=dtype, device=device)
        c = self.scenario
        details = [np.asarray(obs["request_details"], dtype=np.float32).reshape(-1, 3 + c.n_fcs)
                   for obs in observations]
        # At least one padding row makes an entirely empty batch well-defined.
        width = max(1, max(len(rows) for rows in details))
        requests = np.zeros((len(details), width, self.request_size), dtype=np.float32)
        valid = np.zeros((len(details), width), dtype=np.float32)
        origins = np.zeros((len(details), width, c.n_zones), dtype=np.float32)
        for batch_index, rows in enumerate(details):
            n = len(rows)
            if n == 0:
                continue
            origin = np.eye(c.n_zones, dtype=np.float32)[rows[:, 0].astype(np.int64)]
            choice = np.eye(c.n_fcs + 3, dtype=np.float32)[rows[:, 2].astype(np.int64) + 1]
            position = (np.arange(n, dtype=np.float32) / max(1, n - 1))[:, None]
            requests[batch_index, :n] = np.concatenate(
                (origin, rows[:, 1:2] / c.request_max_kwh, choice, rows[:, 3:], position), axis=1)
            origins[batch_index, :n] = origin
            valid[batch_index, :n] = 1.0
        valid_tensor = torch.as_tensor(valid, dtype=dtype, device=device)
        request_tensor = torch.as_tensor(requests, dtype=dtype, device=device)
        encoded = self.request_encoder(request_tensor) * valid_tensor.unsqueeze(-1)
        mean = encoded.sum(dim=1) / valid_tensor.sum(dim=1, keepdim=True).clamp_min(1)
        origin_tensor = torch.as_tensor(origins, dtype=dtype, device=device)
        # Node sums retain demand magnitude; no request count is ever clipped.
        by_zone = torch.bmm(origin_tensor.transpose(1, 2), encoded) / self.count_scale
        return self.body(torch.cat((dense, mean, by_zone.flatten(start_dim=1)), dim=-1))


class PricingActor(nn.Module):
    """Single centralized actor with one independent categorical per node."""

    def __init__(self, scenario: Scenario, hidden_size: int = 64, *, layer_norm: bool = False):
        super().__init__()
        self.scenario = scenario
        self.encoder = _ObservationEncoder(scenario, hidden_size, local=False, layer_norm=layer_norm)
        self.head = _linear(hidden_size, scenario.n_zones * scenario.n_price_levels, gain=0.01)

    def _distribution(self, observations: Sequence[dict]) -> tuple[Categorical, Tensor]:
        logits = self.head(self.encoder(observations)).reshape(
            len(observations), self.scenario.n_zones, self.scenario.n_price_levels)
        mask = torch.as_tensor(np.stack([obs["pricing_mask"] for obs in observations]),
                               dtype=torch.bool, device=logits.device)
        active = mask.any(dim=-1)
        # An unavailable node has one constant dummy outcome. Its probability,
        # log probability, entropy and policy gradient are exactly 1, 0, 0, 0.
        safe_mask = mask.clone()
        safe_mask[..., 0] |= ~active
        logits = torch.where(active.unsqueeze(-1), logits, torch.zeros_like(logits))
        logits = logits.masked_fill(~safe_mask, -torch.inf)
        return Categorical(logits=logits), active

    def forward(self, observations: Sequence[dict]) -> Tensor:
        """Return masked, normalized logit factors with shape [batch, zone, level]."""
        return self._distribution(observations)[0].logits

    def sample(self, observations: Sequence[dict], deterministic: bool = False) -> dict[str, Tensor]:
        distribution, active = self._distribution(observations)
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return {"actions": actions, "log_prob": distribution.log_prob(actions).sum(dim=-1),
                "entropy": distribution.entropy().sum(dim=-1), "active": active.any(dim=-1)}

    def evaluate_actions(self, observations: Sequence[dict], actions: Tensor) -> dict[str, Tensor]:
        distribution, active = self._distribution(observations)
        actions = torch.as_tensor(actions, dtype=torch.long, device=distribution.logits.device)
        if actions.shape != active.shape:
            raise ValueError(f"Pricing actions must have shape {tuple(active.shape)}")
        safe_actions = torch.where(active, actions, torch.zeros_like(actions))
        return {"log_prob": distribution.log_prob(safe_actions).sum(dim=-1),
                "entropy": distribution.entropy().sum(dim=-1), "active": active.any(dim=-1)}


class DispatchActor(nn.Module):
    """Shared local actor over masked (mode, target) and conditional energy.

    Flat discrete actions are ``mode * n_zones + target``. Energy fractions
    follow a Beta distribution conditional on the chosen discrete action for
    RECHARGE/DISCHARGE, and are zero otherwise. PPO must retain this latent
    fraction; the environment's clipped/executed energy is not its likelihood.
    """

    def __init__(self, scenario: Scenario, hidden_size: int = 64, *,
                 final_activation: str = 'tanh', include_agent_id: bool = False,
                 layer_norm: bool = False):
        if not isinstance(final_activation, str) or final_activation not in ('tanh', 'silu'):
            raise ValueError("final_activation must be 'tanh' or 'silu'")
        super().__init__()
        self.scenario = scenario
        self.final_activation = final_activation
        self.include_agent_id = include_agent_id
        self.encoder = _ObservationEncoder(scenario, hidden_size, local=True,
                                           include_agent_id=include_agent_id, layer_norm=layer_norm)
        # Target only this actor's final hidden activation. Pricing, both
        # critics, and the earlier/request encoder activations stay unchanged.
        if final_activation == 'silu':
            self.encoder.body[-1] = nn.SiLU()
        self.n_actions = len(Mode) * scenario.n_zones
        self.action_head = _linear(hidden_size, self.n_actions, gain=0.01)
        self.energy_head = _linear(hidden_size, self.n_actions * 2, gain=0.01)
        energy_actions = torch.arange(self.n_actions) // scenario.n_zones
        self.register_buffer("energy_actions", (energy_actions == Mode.RECHARGE)
                             | (energy_actions == Mode.DISCHARGE))

    def forward(self, observations: Sequence[dict]) -> tuple[Tensor, Tensor]:
        features = self.encoder(observations)
        logits = self.action_head(features)
        masks = torch.as_tensor(np.stack([obs["action_mask"] for obs in observations]),
                                dtype=torch.bool, device=logits.device).reshape(len(observations), -1)
        if masks.shape != logits.shape:
            raise ValueError("Dispatch action masks must contain one vehicle's (mode, target) grid")
        if not bool(masks.any(dim=-1).all()):
            raise ValueError("Every dispatch observation must have a feasible action")
        logits = logits.masked_fill(~masks, -torch.inf)
        concentrations = F.softplus(self.energy_head(features).reshape(-1, self.n_actions, 2)) + 1.0
        return logits, concentrations

    def _distributions(self, observations: Sequence[dict]) -> tuple[Categorical, Beta]:
        logits, concentrations = self(observations)
        return Categorical(logits=logits), Beta(concentrations[..., 0], concentrations[..., 1])

    def _statistics(self, categorical: Categorical, energy: Beta,
                    actions: Tensor, fractions: Tensor) -> dict[str, Tensor]:
        is_energy = self.energy_actions[actions]
        selected_energy = Beta(energy.concentration1.gather(1, actions[:, None]).squeeze(1),
                               energy.concentration0.gather(1, actions[:, None]).squeeze(1))
        # Nonenergy actions use a harmless interior value to avoid 0 * -inf.
        safe_fractions = torch.where(is_energy, fractions, torch.full_like(fractions, 0.5))
        energy_log_prob = torch.where(is_energy, selected_energy.log_prob(safe_fractions),
                                      torch.zeros_like(fractions))
        # Exact entropy of the categorical + conditional continuous policy.
        # It includes every energy branch weighted by its categorical mass.
        entropy = categorical.entropy() + (
            categorical.probs * energy.entropy() * self.energy_actions.to(categorical.probs.dtype)
        ).sum(dim=-1)
        return {"log_prob": categorical.log_prob(actions) + energy_log_prob, "entropy": entropy}

    def sample(self, observations: Sequence[dict], deterministic: bool = False) -> dict[str, Tensor]:
        categorical, energy = self._distributions(observations)
        actions = categorical.logits.argmax(dim=-1) if deterministic else categorical.sample()
        selected_energy = Beta(energy.concentration1.gather(1, actions[:, None]).squeeze(1),
                               energy.concentration0.gather(1, actions[:, None]).squeeze(1))
        fractions = selected_energy.mean if deterministic else selected_energy.sample()
        epsilon = torch.finfo(fractions.dtype).eps
        fractions = fractions.clamp(epsilon, 1.0 - epsilon)
        fractions = torch.where(self.energy_actions[actions], fractions, torch.zeros_like(fractions))
        return {"actions": actions, "energy_fraction": fractions,
                **self._statistics(categorical, energy, actions, fractions)}

    def evaluate_actions(self, observations: Sequence[dict], actions: Tensor,
                         energy_fraction: Tensor) -> dict[str, Tensor]:
        categorical, energy = self._distributions(observations)
        actions = torch.as_tensor(actions, dtype=torch.long, device=categorical.logits.device)
        fractions = torch.as_tensor(energy_fraction, dtype=energy.concentration1.dtype,
                                    device=categorical.logits.device)
        expected = (len(observations),)
        if actions.shape != expected or fractions.shape != expected:
            raise ValueError(f"Dispatch actions and energy fractions must have shape {expected}")
        return self._statistics(categorical, energy, actions, fractions)


class ValueNetwork(nn.Module):
    """Centralized global-state value; use independent pre/post-choice instances."""

    def __init__(self, scenario: Scenario, hidden_size: int = 64, *, layer_norm: bool = False):
        super().__init__()
        self.encoder = _ObservationEncoder(scenario, hidden_size, local=False, layer_norm=layer_norm)
        self.head = _linear(hidden_size, 1, gain=1.0)

    def forward(self, observations: Sequence[dict]) -> Tensor:
        return self.head(self.encoder(observations)).squeeze(-1)
