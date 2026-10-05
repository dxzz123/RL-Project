"""Two-stage finite-horizon MDP dynamics. There is no policy or learner here."""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .config import Scenario


class Mode(IntEnum):
    WAIT = 0
    SERVE = 1
    REPOSITION = 2
    RECHARGE = 3
    DISCHARGE = 4


@dataclass(frozen=True)
class DispatchAction:
    mode: Mode
    target_zone: int
    energy_kwh: float = 0.0  # Grid input for RECHARGE, export through FCS for DISCHARGE.


class ChargingSimulator:
    """reset -> quote_prices -> dispatch -> quote_prices -> ...

    Prices are per MCV, not per zone. Users select a named MCV and a zone;
    each MCV can serve only its selected users at one zone in a period.
    """
    def __init__(self, scenario=None):
        self.config = (scenario or Scenario()).validate()
        self.rng = np.random.default_rng()
        self._ready = False

    def reset(self, *, seed=None, options=None):
        c = self.config
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if set(options) - {"requests"}:
            raise ValueError("Only the first-period 'requests' override is supported")
        self.period = 0
        self.stage = 0  # 0 = pricing, 1 = dispatch, 2 = terminal.
        self.locations = np.array(c.initial_locations, dtype=np.int64)
        self.energy = np.array(c.initial_energy_kwh, dtype=float)
        self.mobile_prices = np.full(c.n_mcv, c.price_min, dtype=float)
        self.mobile_demand = np.zeros((c.n_mcv, c.n_zones), dtype=np.int64)
        self.fixed_demand = np.zeros(c.n_fcs, dtype=np.int64)
        self.outside = 0
        self.cumulative_profit = 0.0
        self.cumulative_discounted_profit = 0.0
        self.requests = self._arrivals()
        if "requests" in options:
            requested = np.asarray(options["requests"])
            if requested.shape != (c.n_zones,) or not np.issubdtype(requested.dtype, np.integer):
                raise ValueError("requests must be an integer vector of length n_zones")
            if (requested < 0).any() or (requested > c.max_requests_per_zone).any():
                raise ValueError("requests exceed the configured bounds")
            self.requests = requested.astype(np.int64).copy()
        self._ready = True
        return self.observe(), {"stage": "pricing", "discount": 1.0}

    def _arrivals(self):
        c = self.config
        means = np.asarray(c.mean_requests_by_zone) * c.arrival_profile[self.period]
        return self.rng.binomial(c.max_requests_per_zone, means / c.max_requests_per_zone)

    def _require(self, stage):
        if not self._ready:
            raise RuntimeError("Call reset() first")
        if self.stage != stage:
            raise RuntimeError(f"Expected stage {stage}; current stage is {self.stage}")

    def _travel(self, m, zone):
        c = self.config
        distance = float(c.distances[self.locations[m], zone])
        return distance, c.period_hours - distance / c.speed_kmph, distance * c.travel_kwh_per_km

    def service_limit(self, m, zone):
        c = self.config
        _, hours, travel = self._travel(m, zone)
        if hours < 0:
            return 0
        deliverable = min(max(0., self.energy[m] - travel - c.reserve_kwh)
                          * c.mobile_service_efficiency, hours * c.mobile_service_kw)
        return max(0, int(np.floor((deliverable + 1e-9) / c.request_kwh)))

    def choice_probabilities(self, prices):
        """Columns: MCV ids, FCS ids, outside. One categorical distribution per zone."""
        self._require(0)
        c = self.config
        prices = self._validate_prices(prices)
        utilities = np.full((c.n_zones, c.n_mcv + c.n_fcs + 1), -np.inf)
        for z in range(c.n_zones):
            for m in range(c.n_mcv):
                if self.service_limit(m, z) >= 1:
                    utilities[z, m] = (c.mobile_utility_intercept
                        - c.price_sensitivity_per_dollar * c.request_kwh * prices[m]
                        - c.mobile_distance_disutility_per_km * c.distances[self.locations[m], z])
            for f, fzone in enumerate(c.fcs_zones):
                distance = c.distances[z, fzone]
                if distance <= c.max_fcs_access_km and c.fcs_ev_capacity_kwh[f] >= c.request_kwh:
                    utilities[z, c.n_mcv + f] = (c.fixed_utility_intercept
                        - c.price_sensitivity_per_dollar * c.request_kwh * c.fcs_retail_per_kwh[f]
                        - c.fixed_distance_disutility_per_km * distance)
            utilities[z, -1] = c.outside_utility
        shifted = utilities - utilities.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)

    def _validate_prices(self, prices):
        c = self.config
        prices = np.asarray(prices, dtype=float)
        if prices.shape != (c.n_mcv,) or not np.isfinite(prices).all():
            raise ValueError("prices must be a finite vector of length n_mcv")
        if (prices < c.price_min).any() or (prices > c.price_max).any():
            raise ValueError("Prices are outside the configured bounds")
        return prices.copy()

    def quote_prices(self, prices):
        self._require(0)
        c = self.config
        prices = self._validate_prices(prices)
        probabilities = self.choice_probabilities(prices)
        choices = np.stack([self.rng.multinomial(int(n), p)
                            for n, p in zip(self.requests, probabilities)])
        self.mobile_prices = prices
        self.mobile_demand = choices[:, :c.n_mcv].T.copy()
        self.fixed_demand = choices[:, c.n_mcv:c.n_mcv+c.n_fcs].sum(axis=0)
        self.outside = int(choices[:, -1].sum())
        self.stage = 1
        return self.observe(), 0.0, False, False, {
            "stage": "dispatch", "choice_probabilities": probabilities.copy(),
            "choice_counts_by_zone": choices.copy(), "discount": 1.0,
        }

    def feasible_actions(self):
        """Individual feasibility [MCV, mode, zone]; joint station limits are resolved later."""
        c = self.config
        mask = np.zeros((c.n_mcv, len(Mode), c.n_zones), dtype=bool)
        if not self._ready or self.stage != 1:
            return mask
        for m in range(c.n_mcv):
            mask[m, Mode.WAIT, self.locations[m]] = True
            for z in range(c.n_zones):
                _, hours, travel = self._travel(m, z)
                if hours < 0 or self.energy[m] - travel < c.reserve_kwh - 1e-9:
                    continue
                mask[m, Mode.REPOSITION, z] = True
                mask[m, Mode.SERVE, z] = self.mobile_demand[m, z] > 0 and self.service_limit(m, z) > 0
                if z in c.fcs_zones and hours > 0:
                    f = c.fcs_zones.index(z)
                    has_exchange = c.fcs_fleet_exchange_capacity_kwh[f] > 0
                    mask[m, Mode.RECHARGE, z] = has_exchange and self.energy[m] - travel < c.battery_capacity_kwh
                    mask[m, Mode.DISCHARGE, z] = has_exchange and self.energy[m] - travel > c.reserve_kwh
        return mask

    def _validate_actions(self, actions):
        c = self.config
        if len(actions) != c.n_mcv:
            raise ValueError("Provide one DispatchAction per MCV")
        clean = []
        for action in actions:
            if not isinstance(action, DispatchAction):
                raise ValueError("Actions must be DispatchAction objects")
            try:
                mode = Mode(action.mode)
            except (ValueError, TypeError) as exc:
                raise ValueError("Unknown dispatch mode") from exc
            if isinstance(action.mode, (float, np.floating, bool, np.bool_)):
                raise ValueError("mode must be an integer or Mode")
            z = action.target_zone
            if isinstance(z, bool) or not isinstance(z, (int, np.integer)) or not 0 <= z < c.n_zones:
                raise ValueError("target_zone must be a valid integer zone index")
            if not np.isfinite(action.energy_kwh) or action.energy_kwh < 0:
                raise ValueError("energy_kwh must be finite and nonnegative")
            clean.append(DispatchAction(mode, int(z), float(action.energy_kwh)))
        return clean

    def dispatch(self, actions):
        self._require(1)
        actions = self._validate_actions(actions)  # Fail before any mutation.
        c = self.config
        before = self.energy.copy()
        mask = self.feasible_actions()
        served_mobile = np.zeros_like(self.mobile_demand)
        travel_kwh = np.zeros(c.n_mcv)
        travel_km = np.zeros(c.n_mcv)
        charge_grid = np.zeros(c.n_mcv)
        export_grid = np.zeros(c.n_mcv)
        fcs_charge_grid = np.zeros(c.n_fcs)
        fcs_export_grid = np.zeros(c.n_fcs)
        requested_exchange = np.zeros(c.n_mcv)
        station = np.full(c.n_mcv, -1, dtype=int)
        modes = np.full(c.n_mcv, Mode.WAIT, dtype=int)
        invalid = []
        for m, action in enumerate(actions):
            mode, z = action.mode, action.target_zone
            if not mask[m, mode, z]:
                invalid.append({"mcv": m, "reason": "individually infeasible action; executed WAIT"})
                continue
            modes[m] = mode
            if mode == Mode.WAIT:
                continue
            distance, hours, travel = self._travel(m, z)
            if mode == Mode.SERVE:
                served_mobile[m, z] = min(self.service_limit(m, z), self.mobile_demand[m, z])
            travel_km[m], travel_kwh[m] = distance, travel
            self.energy[m] -= travel
            self.locations[m] = z
            if mode in (Mode.RECHARGE, Mode.DISCHARGE):
                station[m] = c.fcs_zones.index(z)
                if mode == Mode.RECHARGE:
                    requested_exchange[m] = min(action.energy_kwh, hours*c.mobile_charge_kw,
                        max(0., c.battery_capacity_kwh-self.energy[m])/c.charge_efficiency)
                else:
                    requested_exchange[m] = min(action.energy_kwh, hours*c.mobile_discharge_kw,
                        max(0., self.energy[m]-c.reserve_kwh)*c.discharge_efficiency)
        # Proportional sharing: charging plus export use one fleet exchange kWh budget.
        for f in range(c.n_fcs):
            members = np.flatnonzero(station == f)
            total = float(requested_exchange[members].sum())
            factor = min(1., c.fcs_fleet_exchange_capacity_kwh[f]/total) if total > 0 else 1.
            for m in members:
                allocated = requested_exchange[m]*factor
                if modes[m] == Mode.RECHARGE:
                    charge_grid[m] = allocated
                    fcs_charge_grid[f] += allocated
                else:
                    export_grid[m] = allocated
                    fcs_export_grid[f] += allocated
        mobile_delivered = served_mobile.sum(axis=1)*c.request_kwh
        self.energy += c.charge_efficiency*charge_grid
        self.energy -= mobile_delivered/c.mobile_service_efficiency + export_grid/c.discharge_efficiency
        fixed_limits = np.floor((np.asarray(c.fcs_ev_capacity_kwh)+1e-9)/c.request_kwh).astype(int)
        served_fixed = np.minimum(self.fixed_demand, fixed_limits)
        fixed_delivered = served_fixed*c.request_kwh
        fixed_grid = fixed_delivered/c.fixed_service_efficiency
        unmet_mobile = int(self.mobile_demand.sum()-served_mobile.sum())
        unmet_fixed = int(self.fixed_demand.sum()-served_fixed.sum())
        t = self.period
        terminal = t + 1 == c.horizon
        terms = {
            "mobile_service_revenue": float(np.dot(self.mobile_prices, mobile_delivered)),
            "fixed_service_revenue": float(np.dot(c.fcs_retail_per_kwh, fixed_delivered)),
            "discharge_revenue": float(c.grid_sell_per_kwh[t]*export_grid.sum()),
            "procurement_cost": float(c.grid_buy_per_kwh[t]*(fixed_grid.sum()+charge_grid.sum())),
            "travel_cost": float(c.travel_cost_per_km*travel_km.sum()),
            "operating_cost": float(c.operating_cost_per_mcv_period*c.n_mcv),
            "unmet_demand_cost": float(c.unmet_penalty_per_kwh*c.request_kwh*(unmet_mobile+unmet_fixed)),
            "terminal_inventory_value": float(c.terminal_value_per_stored_kwh*self.energy.sum()) if terminal else 0.,
        }
        reward = (terms["mobile_service_revenue"]+terms["fixed_service_revenue"]
            +terms["discharge_revenue"]+terms["terminal_inventory_value"]
            -terms["procurement_cost"]-terms["travel_cost"]-terms["operating_cost"]
            -terms["unmet_demand_cost"])
        self.cumulative_profit += reward
        self.cumulative_discounted_profit += (c.discount_per_period**t)*reward
        residual = (self.energy-before+travel_kwh-c.charge_efficiency*charge_grid
            +mobile_delivered/c.mobile_service_efficiency+export_grid/c.discharge_efficiency)
        if not np.allclose(residual, 0., atol=1e-8) or (self.energy < c.reserve_kwh-1e-8).any() or (self.energy > c.battery_capacity_kwh+1e-8).any():
            raise AssertionError("Energy conservation or battery bounds violated")
        info = {
            "completed_period": t, "stage": "terminal" if terminal else "pricing",
            "reward_terms": terms, "served_mobile": served_mobile.copy(),
            "served_fixed": served_fixed.copy(), "unmet_mobile_requests": unmet_mobile,
            "unmet_fixed_requests": unmet_fixed, "outside_requests": self.outside,
            "total_requests": int(self.requests.sum()), "invalid_actions": invalid,
            "executed_modes": modes.copy(), "cumulative_profit": self.cumulative_profit,
            "cumulative_discounted_profit": self.cumulative_discounted_profit,
            "discount": 0. if terminal else c.discount_per_period,
            "energy_ledger": {"before_kwh": before, "after_kwh": self.energy.copy(),
                "travel_kwh": travel_kwh, "mobile_delivered_kwh": mobile_delivered,
                "charge_grid_kwh": charge_grid, "export_grid_kwh": export_grid,
                "fcs_charge_grid_kwh": fcs_charge_grid,
                "fcs_export_grid_kwh": fcs_export_grid,
                "fixed_grid_kwh": fixed_grid, "balance_residual_kwh": residual},
        }
        self.period += 1
        self.stage = 2 if terminal else 0
        self.mobile_prices = np.full(c.n_mcv, c.price_min)
        self.mobile_demand.fill(0)
        self.fixed_demand.fill(0)
        self.outside = 0
        self.requests = np.zeros(c.n_zones, dtype=np.int64) if terminal else self._arrivals()
        return self.observe(), float(reward), terminal, False, info

    def observe(self):
        if not self._ready:
            raise RuntimeError("Call reset() first")
        c = self.config
        t = min(self.period, c.horizon-1)
        return {
            "stage": self.stage, "period": np.array([self.period], dtype=np.int64),
            "locations": self.locations.copy(), "energy_kwh": self.energy.copy(),
            "requests_by_zone": self.requests.copy(), "mobile_prices": self.mobile_prices.copy(),
            "mobile_demand": self.mobile_demand.copy(), "fixed_demand": self.fixed_demand.copy(),
            "outside_requests": np.array([self.outside], dtype=np.int64),
            "fcs_retail_prices": np.asarray(c.fcs_retail_per_kwh, dtype=float).copy(),
            "grid_buy_price": np.array([c.grid_buy_per_kwh[t]], dtype=float),
            "grid_sell_price": np.array([c.grid_sell_per_kwh[t]], dtype=float),
            "action_mask": self.feasible_actions().astype(np.int8),
        }

    def local_observation(self, m):
        """Proposal-style partial dispatch observation; other MCV states are excluded."""
        self._require(1)
        if not isinstance(m, int) or not 0 <= m < self.config.n_mcv:
            raise ValueError("Invalid MCV id")
        obs = self.observe()
        return {key: (value[m:m+1].copy() if key in ("locations", "energy_kwh", "action_mask") else value)
                for key, value in obs.items()}
