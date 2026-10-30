"""Simple controls and a coordinated, one-period dispatch benchmark.

Every policy uses only the supplied public observation and static physics. The
coordinated dispatcher exhaustively searches a finite exchange-energy grid; it
is an operational benchmark, not an upper bound on the multi-period optimum.
"""
from itertools import product
from numbers import Real

import numpy as np

from ..simulator import DispatchAction, Mode
from ..choice import charging_choice_probabilities


def _stage(obs, expected):
    if int(obs["stage"]) != expected:
        raise ValueError(f"Policy requires stage {expected}")


def fixed_prices(obs, config, index):
    """Use one admissible price level at every available node."""
    _stage(obs, 0)
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError("Price index must be an integer")
    if not 0 <= index < config.n_price_levels:
        raise ValueError("Price index is out of range")
    return np.where(np.asarray(obs["mobile_available"], dtype=bool), index, 0).astype(np.int64)


def random_prices(obs, config, rng):
    """Uniform price-level sampling at available nodes, with explicit RNG."""
    _stage(obs, 0)
    result = rng.integers(config.n_price_levels, size=config.n_zones)
    result[~np.asarray(obs["mobile_available"], dtype=bool)] = 0
    return result.astype(np.int64)


def _travel_arrays(obs, config):
    distances = config.distances[np.asarray(obs["locations"], dtype=int)]
    hours = config.period_hours - distances / config.speed_kmph
    travel_energy = distances * config.travel_kwh_per_km
    capacities = np.maximum(0., np.minimum(
        (np.asarray(obs["energy_kwh"])[:, None] - travel_energy - config.reserve_kwh)
        * config.mobile_service_efficiency,
        hours * config.mobile_service_kw,
    ))
    return distances, hours, travel_energy, capacities


def adaptive_prices(obs, config):
    """Approximate expected-margin pricing using the model's current MNL.

Each vehicle's service capacity is apportioned across eligible nodes by their
observed potential energy. The score subtracts replacement-energy cost and an
overflow penalty and includes expected FCS margin. Expected energy and capacity
are fluid proxies; no future arrivals or sampled customer choices are used.
    """
    _stage(obs, 0)
    result = np.zeros(config.n_zones, dtype=np.int64)
    details = np.asarray(obs["request_details"], dtype=float)
    if len(details) == 0:
        return result
    origins, energy = details[:, 0].astype(int), details[:, 1]
    access = details[:, 3:].astype(bool)
    _, _, _, capacity = _travel_arrays(obs, config)
    eligible = np.asarray(obs["eligible_mcvs"], dtype=bool)
    weights = eligible * np.asarray(obs["potential_energy_kwh"])[None, :]
    weights = np.divide(weights, weights.sum(axis=1, keepdims=True),
                        out=np.zeros_like(weights, dtype=float), where=weights.sum(axis=1, keepdims=True) > 0)
    mobile_capacity = (capacity * weights).sum(axis=0)
    # Allocate FCS capacity by the pre-choice accessibility-weighted load. This
    # avoids counting the same station capacity independently at every node.
    accessible_load = access * (energy / np.maximum(1, access.sum(axis=1)))[:, None]
    station_load = accessible_load.sum(axis=0)
    buy = float(np.asarray(obs["grid_buy_price"])[0])
    replacement_cost = buy / (config.charge_efficiency * config.mobile_service_efficiency)
    menu = np.asarray(obs["mobile_price_menu"], dtype=float)
    for z in np.flatnonzero(obs["mobile_available"]):
        at = origins == z
        if not at.any():
            continue
        q = energy[at]
        levels = menu[z]
        mobile_cost = (levels[:, None] * q
                       + config.mobile_wait_cost_per_hour * obs["mobile_wait_hours"][z])
        fixed_cost = (q[:, None] * obs["fcs_retail_prices"]
                      + config.fixed_travel_cost_per_km * config.distances[z, config.fcs_zones]
                      + config.fixed_wait_cost_per_hour * obs["fcs_wait_hours"])
        probabilities = charging_choice_probabilities(
            config, mobile_cost, fixed_cost,
            bool(obs["mobile_available"][z]), access[at])
        mobile_demand = (probabilities[..., 0] * q).sum(axis=1)
        fixed_demand = (probabilities[..., 1:1+config.n_fcs] * q[None, :, None]).sum(axis=1)
        node_station_share = np.divide(accessible_load[at].sum(axis=0), station_load,
                                       out=np.zeros(config.n_fcs), where=station_load > 0)
        fixed_capacity = np.asarray(config.fcs_ev_capacity_kwh) * node_station_share
        mobile_served = np.minimum(mobile_demand, mobile_capacity[z])
        fixed_served = np.minimum(fixed_demand, fixed_capacity)
        scores = ((levels - replacement_cost) * mobile_served
                  - config.unmet_penalty_per_kwh * (mobile_demand - mobile_served)
                  + (fixed_served * (obs["fcs_retail_prices"] - buy / config.fixed_service_efficiency)).sum(axis=1)
                  - config.unmet_penalty_per_kwh * (fixed_demand - fixed_served).sum(axis=1))
        result[z] = int(np.argmax(scores))
    return result


def _maximum_exchange(obs, config, m, mode, zone):
    distance = config.distances[int(obs["locations"][m]), zone]
    hours = max(0., config.period_hours - distance / config.speed_kmph)
    after_travel = obs["energy_kwh"][m] - distance * config.travel_kwh_per_km
    if mode == Mode.RECHARGE:
        return max(0., min(hours * config.mobile_charge_kw,
                          (config.battery_capacity_kwh - after_travel) / config.charge_efficiency))
    if mode == Mode.DISCHARGE:
        return max(0., min(hours * config.mobile_discharge_kw,
                          (after_travel - config.reserve_kwh) * config.discharge_efficiency))
    return 0.


def random_dispatch(obs, config, rng):
    """Uniform feasible mode/target choices and uniform feasible exchange."""
    _stage(obs, 1)
    actions = []
    for m in range(config.n_mcv):
        choices = np.argwhere(np.asarray(obs["action_mask"])[m])
        mode, zone = choices[int(rng.integers(len(choices)))]
        mode, zone = Mode(int(mode)), int(zone)
        maximum = _maximum_exchange(obs, config, m, mode, zone)
        actions.append(DispatchAction(mode, zone, float(rng.random() * maximum) if maximum else 0.))
    return actions


def service_first_dispatch(obs, config, recharge_threshold=.3):
    """Coordinate request service, then refill low batteries when useful.

Vehicles act in the simulator's allocation order. Candidate requests are packed
first-fit in their observed order and removed once assigned. Among service
destinations choose the largest immediate margin including avoided unmet cost
and replacement energy. Recharge uses spare station capacity and is skipped in
the terminal period; the rule never draws on hidden state.
    """
    _stage(obs, 1)
    if not np.isfinite(recharge_threshold) or not 0 <= recharge_threshold <= 1:
        raise ValueError("recharge_threshold must be in [0, 1]")
    details = np.asarray(obs["request_details"])
    origins, q, choices = details[:, 0].astype(int), details[:, 1], details[:, 2].astype(int)
    served = np.zeros(len(q), dtype=bool)
    masks = np.asarray(obs["action_mask"], dtype=bool)
    distances, _, _, capacities = _travel_arrays(obs, config)
    buy = float(obs["grid_buy_price"][0])
    replacement = buy / config.charge_efficiency
    station_used = np.zeros(config.n_fcs)
    terminal = int(obs["period"][0]) + 1 >= config.horizon
    actions = []
    for m in range(config.n_mcv):
        best_score, best_zone, best_requests = 0., None, []
        for z in np.flatnonzero(masks[m, Mode.SERVE]):
            remaining, selected = capacities[m, z], []
            for r in np.flatnonzero((origins == z) & (choices == 0) & ~served):
                if q[r] <= remaining + 1e-9:
                    selected.append(r)
                    remaining -= q[r]
            amount = float(q[selected].sum())
            # Existing stock has no replacement charge in the final period.
            energy_value = (config.terminal_value_per_stored_kwh if terminal else replacement)
            score = ((obs["mobile_prices"][z] + config.unmet_penalty_per_kwh) * amount
                     - config.travel_cost_per_km * distances[m, z]
                     - energy_value * (amount / config.mobile_service_efficiency
                                       + distances[m, z] * config.travel_kwh_per_km))
            if selected and score > best_score:
                best_score, best_zone, best_requests = score, int(z), selected
        if best_zone is not None:
            actions.append(DispatchAction(Mode.SERVE, best_zone))
            served[best_requests] = True
            continue
        usable = config.battery_capacity_kwh - config.reserve_kwh
        fraction = ((obs["energy_kwh"][m] - config.reserve_kwh) / usable) if usable else 1.
        recharge_choices = []
        if not terminal and fraction < recharge_threshold:
            for z in np.flatnonzero(masks[m, Mode.RECHARGE]):
                f = config.fcs_zones.index(int(z))
                amount = min(_maximum_exchange(obs, config, m, Mode.RECHARGE, int(z)),
                             max(0., config.fcs_fleet_exchange_capacity_kwh[f] - station_used[f]))
                if amount > 1e-9:
                    # Prefer low travel cost per replenished kWh, then shorter travel.
                    cost = (config.travel_cost_per_km * distances[m, z]
                            + replacement * distances[m, z] * config.travel_kwh_per_km)
                    recharge_choices.append((cost / amount, distances[m, z], int(z), f, amount))
        if recharge_choices:
            _, _, z, f, amount = min(recharge_choices)
            actions.append(DispatchAction(Mode.RECHARGE, z, float(amount)))
            station_used[f] += amount
        else:
            actions.append(DispatchAction(Mode.WAIT, int(obs["locations"][m])))
    return actions


def _batch_dispatch_outcomes(obs, c, action_modes, action_targets, requested_energy):
    """Independent analytical dispatch dynamics, vectorized over joint actions."""
    batch = len(action_modes)
    mcv_index = np.arange(c.n_mcv)[None, :]
    valid = np.asarray(obs["action_mask"], dtype=bool)[mcv_index, action_modes, action_targets]
    modes = np.where(valid, action_modes, int(Mode.WAIT))
    moves = modes != Mode.WAIT
    targets = np.where(moves, action_targets, np.asarray(obs["locations"])[None, :])
    distances = c.distances[np.asarray(obs["locations"])[None, :], targets]
    hours = c.period_hours - distances / c.speed_kmph
    travel_energy = distances * c.travel_kwh_per_km
    energy = np.broadcast_to(obs["energy_kwh"], (batch, c.n_mcv)).copy() - travel_energy
    details = np.asarray(obs["request_details"])
    origins, q, choices = details[:, 0].astype(int), details[:, 1], details[:, 2].astype(int)
    served = np.zeros((batch, len(q)), dtype=bool)
    assigned = np.full((batch, len(q)), -1, dtype=np.int64)
    delivered = np.zeros((batch, c.n_mcv, c.n_zones))
    # Iterate vehicle and request order exactly; only candidate joint actions are
    # vectorized, so heterogeneous whole-request first-fit behavior is preserved.
    for m in range(c.n_mcv):
        remaining = np.maximum(0., np.minimum((energy[:, m] - c.reserve_kwh) * c.mobile_service_efficiency,
                                             hours[:, m] * c.mobile_service_kw))
        for r in np.flatnonzero(choices == 0):
            accept = ((modes[:, m] == Mode.SERVE) & (targets[:, m] == origins[r])
                      & ~served[:, r] & (q[r] <= remaining + 1e-9))
            served[:, r] |= accept
            assigned[accept, r] = m
            remaining -= accept * q[r]
            delivered[:, m, origins[r]] += accept * q[r]
    charge_request = np.where(modes == Mode.RECHARGE, np.minimum(requested_energy,
        np.minimum(hours * c.mobile_charge_kw, np.maximum(0., c.battery_capacity_kwh - energy) / c.charge_efficiency)), 0.)
    export_request = np.where(modes == Mode.DISCHARGE, np.minimum(requested_energy,
        np.minimum(hours * c.mobile_discharge_kw, np.maximum(0., energy - c.reserve_kwh) * c.discharge_efficiency)), 0.)
    charge, export = np.zeros_like(energy), np.zeros_like(energy)
    station_charge, station_export = np.zeros((batch, c.n_fcs)), np.zeros((batch, c.n_fcs))
    for f, z in enumerate(c.fcs_zones):
        members = targets == z
        total = ((charge_request + export_request) * members).sum(axis=1)
        factor = np.minimum(1., np.divide(c.fcs_fleet_exchange_capacity_kwh[f], total,
                                         out=np.ones(batch), where=total > 0))
        allocated_charge = charge_request * members * factor[:, None]
        allocated_export = export_request * members * factor[:, None]
        charge += allocated_charge
        export += allocated_export
        station_charge[:, f] = allocated_charge.sum(axis=1)
        station_export[:, f] = allocated_export.sum(axis=1)
    energy += c.charge_efficiency * charge - delivered.sum(axis=2) / c.mobile_service_efficiency - export / c.discharge_efficiency
    fixed_delivered = np.zeros(c.n_fcs)
    for f in range(c.n_fcs):
        remaining = c.fcs_ev_capacity_kwh[f]
        for r in np.flatnonzero(choices == f + 1):
            if q[r] <= remaining + 1e-9:
                served[:, r] = True
                remaining -= q[r]
                fixed_delivered[f] += q[r]
    unmet_mobile = ((~served) * (choices == 0)[None, :] * q).sum(axis=1)
    unmet_fixed = ((~served) * ((choices >= 1) & (choices <= c.n_fcs))[None, :] * q).sum(axis=1)
    terminal = int(obs["period"][0]) + 1 == c.horizon
    terms = {
        "mobile_service_revenue": (delivered * obs["mobile_prices"][None, None, :]).sum(axis=(1, 2)),
        "fixed_service_revenue": np.full(batch, float(np.dot(obs["fcs_retail_prices"], fixed_delivered))),
        "discharge_revenue": (station_export * obs["fcs_discharge_prices"]).sum(axis=1),
        "procurement_cost": float(obs["grid_buy_price"][0]) * (fixed_delivered.sum() / c.fixed_service_efficiency + charge.sum(axis=1)),
        "travel_cost": c.travel_cost_per_km * distances.sum(axis=1),
        "operating_cost": np.full(batch, c.operating_cost_per_mcv_period * c.n_mcv),
        "unmet_demand_cost": c.unmet_penalty_per_kwh * (unmet_mobile + unmet_fixed),
        "terminal_inventory_value": c.terminal_value_per_stored_kwh * energy.sum(axis=1) if terminal else np.zeros(batch),
    }
    reward = (terms["mobile_service_revenue"] + terms["fixed_service_revenue"] + terms["discharge_revenue"]
              + terms["terminal_inventory_value"] - terms["procurement_cost"] - terms["travel_cost"]
              - terms["operating_cost"] - terms["unmet_demand_cost"])
    return {"reward": reward, "reward_terms": terms, "next_energy_kwh": energy,
            "next_locations": targets, "served_mask": served, "assigned_mcv": assigned,
            "delivered_mobile_kwh": delivered, "charge_grid_kwh": charge,
            "export_grid_kwh": export, "invalid_mask": ~valid,
            "unmet_mobile_kwh": unmet_mobile, "unmet_fixed_kwh": unmet_fixed,
            "executed_modes": modes}


def dispatch_outcome(obs, config, actions):
    """Compute exact immediate outcome from public state without advancing it.

Individually infeasible actions execute WAIT, matching the environment. This
function does not create a simulator, read future profiles, or consume RNG.
    """
    _stage(obs, 1)
    if len(actions) != config.n_mcv:
        raise ValueError("Provide one DispatchAction per MCV")
    for action in actions:
        if not isinstance(action, DispatchAction):
            raise ValueError("Actions must be DispatchAction objects")
        if (isinstance(action.mode, (float, np.floating, bool, np.bool_))
                or not isinstance(action.mode, (int, np.integer, Mode))
                or int(action.mode) not in [int(mode) for mode in Mode]):
            raise ValueError("Invalid dispatch mode")
        if (isinstance(action.target_zone, (bool, np.bool_))
                or not isinstance(action.target_zone, (int, np.integer))
                or not 0 <= action.target_zone < config.n_zones):
            raise ValueError("Invalid target zone")
        if (isinstance(action.energy_kwh, (bool, np.bool_)) or not isinstance(action.energy_kwh, Real)
                or not np.isfinite(action.energy_kwh) or action.energy_kwh < 0):
            raise ValueError("energy_kwh must be finite and nonnegative")
    modes = np.array([[int(a.mode) for a in actions]], dtype=np.int64)
    targets = np.array([[a.target_zone for a in actions]], dtype=np.int64)
    amounts = np.array([[a.energy_kwh for a in actions]], dtype=float)
    batch = _batch_dispatch_outcomes(obs, config, modes, targets, amounts)
    result = {key: value[0].copy() if isinstance(value[0], np.ndarray) else value[0].item()
              for key, value in batch.items() if key not in ("reward_terms", "invalid_mask")}
    result["reward_terms"] = {key: float(value[0]) for key, value in batch["reward_terms"].items()}
    result["served_request_mask"] = result["served_mask"].copy()
    result["invalid_actions"] = [int(m) for m in np.flatnonzero(batch["invalid_mask"][0])]
    return result


def joint_dispatch_outcomes(obs, config, energy_fractions=(.5, 1.), max_joint_actions=200000):
    """Enumerate the shared feasible action grid and its exact current outcomes.

    Returns ``(modes, targets, requested_energy, outcomes)``. The first three
    arrays have shape [joint actions, vehicles]; ``outcomes`` includes original
    business reward, next locations, and next stored energy for every action.
    This is the same grid used by the constant-inventory benchmark, so learned
    continuation values can replace its shadow price without changing physics,
    action resolution, candidate ordering, or tie breaking. No RNG is consumed.
    """
    _stage(obs, 1)
    fractions = np.asarray(energy_fractions, dtype=float)
    if (fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all()
            or (fractions < 0).any() or (fractions > 1).any()):
        raise ValueError("energy_fractions must be a nonempty sequence in [0, 1]")
    if isinstance(max_joint_actions, bool) or not isinstance(max_joint_actions, (int, np.integer)) or max_joint_actions < 1:
        raise ValueError("max_joint_actions must be a positive integer")
    candidates = []
    for m in range(config.n_mcv):
        alternatives = []
        for mode, z in np.argwhere(obs["action_mask"][m]):
            mode, z = Mode(int(mode)), int(z)
            if mode == Mode.REPOSITION and z == obs["locations"][m]:
                continue  # Identical outcome to WAIT, which wins exact ties.
            if mode in (Mode.RECHARGE, Mode.DISCHARGE):
                maximum = _maximum_exchange(obs, config, m, mode, z)
                for fraction in np.unique(fractions):
                    alternatives.append(DispatchAction(mode, z, float(maximum * fraction)))
            else:
                alternatives.append(DispatchAction(mode, z))
        candidates.append(alternatives)
    count = 1
    for alternatives in candidates:
        count *= len(alternatives)
        if count > max_joint_actions:
            raise ValueError(f"Joint action enumeration exceeds max_joint_actions={max_joint_actions}; "
                             "use a smaller scenario or explicitly raise the limit")
    joint_indices = np.asarray(list(product(*(range(len(a)) for a in candidates))), dtype=np.int64)
    modes = np.empty((count, config.n_mcv), dtype=np.int64)
    targets = np.empty_like(modes)
    amounts = np.empty_like(modes, dtype=float)
    for m, alternatives in enumerate(candidates):
        modes[:, m] = np.asarray([int(a.mode) for a in alternatives])[joint_indices[:, m]]
        targets[:, m] = np.asarray([a.target_zone for a in alternatives])[joint_indices[:, m]]
        amounts[:, m] = np.asarray([a.energy_kwh for a in alternatives])[joint_indices[:, m]]
    outcomes = _batch_dispatch_outcomes(obs, config, modes, targets, amounts)
    return modes, targets, amounts, outcomes


def coordinated_dispatch(obs, config, shadow_price=0., energy_fractions=(.5, 1.), max_joint_actions=200000):
    """Maximize immediate reward plus a fixed value for remaining stored energy.

All individually feasible mode/target combinations are enumerated. Exchange
amounts are fractions of each vehicle's physically feasible maximum, before
shared-station allocation. The stock shadow price is zero in the terminal
period, where the scenario's actual terminal inventory value is retained.
The enumeration limit raises an error instead of silently changing algorithms.
    """
    _stage(obs, 1)
    if not np.isfinite(shadow_price) or shadow_price < 0:
        raise ValueError("shadow_price must be finite and nonnegative")
    modes, targets, amounts, outcomes = joint_dispatch_outcomes(
        obs, config, energy_fractions, max_joint_actions)
    terminal = int(obs["period"][0]) + 1 >= config.horizon
    effective_shadow = 0. if terminal else shadow_price
    scores = outcomes["reward"] + effective_shadow * (outcomes["next_energy_kwh"].sum(axis=1) - np.sum(obs["energy_kwh"]))
    winner = int(np.argmax(scores))
    return [DispatchAction(Mode(int(modes[winner, m])), int(targets[winner, m]), float(amounts[winner, m]))
            for m in range(config.n_mcv)]
