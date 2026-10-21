"""Shared request-level customer choice for simulation and pricing benchmarks."""
from __future__ import annotations

import numpy as np

from .config import Scenario


def charging_choice_probabilities(
    config: Scenario, mobile_cost, fixed_cost, mobile_available, accessible_fcs,
) -> np.ndarray:
    """Return MNL probabilities ordered [mobile, FCSs, outside].

    Service utility is ``b - eta * cost``; outside utility stays zero. Costs
    already include payment, waiting, and applicable travel. The common
    intercept therefore changes participation, while preserving conditional
    provider preferences for a given request and fixed costs/availability.

    Inputs broadcast over request/price-candidate batch axes. ``fixed_cost``
    and ``accessible_fcs`` have a final axis of length ``config.n_fcs``.
    Unavailable alternatives retain probability zero, even for a large b.
    The caller supplies a validated Scenario. No random state is consumed.
    """
    mobile_cost = np.asarray(mobile_cost, dtype=float)
    fixed_cost = np.asarray(fixed_cost, dtype=float)
    mobile_available = np.asarray(mobile_available, dtype=bool)
    accessible_fcs = np.asarray(accessible_fcs, dtype=bool)
    if (fixed_cost.ndim < 1 or fixed_cost.shape[-1] != config.n_fcs
            or accessible_fcs.ndim < 1 or accessible_fcs.shape[-1] != config.n_fcs):
        raise ValueError('Fixed costs and accessibility must end in an n_fcs axis')
    if not np.isfinite(mobile_cost).all() or not np.isfinite(fixed_cost).all():
        raise ValueError('Charging costs must be finite')
    shape = np.broadcast_shapes(mobile_cost.shape, fixed_cost.shape[:-1],
                                mobile_available.shape, accessible_fcs.shape[:-1])
    mobile_cost = np.broadcast_to(mobile_cost, shape)
    fixed_cost = np.broadcast_to(fixed_cost, shape + (config.n_fcs,))
    mobile_available = np.broadcast_to(mobile_available, shape)
    accessible_fcs = np.broadcast_to(accessible_fcs, shape + (config.n_fcs,))
    utilities = np.full(shape + (config.n_fcs + 2,), -np.inf)
    b, eta = config.charging_utility_intercept, config.price_sensitivity_per_dollar
    utilities[..., 0] = np.where(mobile_available, b - eta * mobile_cost, -np.inf)
    utilities[..., 1:-1] = np.where(accessible_fcs, b - eta * fixed_cost, -np.inf)
    utilities[..., -1] = 0.0
    # Outside is always finite, so every row has a defined normalization.
    weights = np.exp(utilities - utilities.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)
