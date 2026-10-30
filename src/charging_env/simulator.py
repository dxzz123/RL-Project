"""Request-level two-stage MDP dynamics; policies and learners are separate."""
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real

import numpy as np

from .config import Scenario
from .choice import charging_choice_probabilities


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
    energy_kwh: float = 0.0  # Grid input for recharge, grid export for discharge.


class ChargingSimulator:
    """reset -> node pricing -> request choices -> cooperative dispatch -> ...

    Choice columns are mobile, FCS 0..F-1, outside. Requests select a pooled
    mobile service at their origin, never an individual vehicle.
    """

    def __init__(self, scenario=None):
        self.config = (scenario or Scenario()).validate()
        self.rng = np.random.default_rng()
        self._ready = False

    def _validate_options(self, options):
        c = self.config
        options = {} if options is None else dict(options)
        if set(options) - {"requests", "request_energy_kwh", "accessible_fcs"}:
            raise ValueError("Unknown reset options")
        if "requests" not in options:
            if options:
                raise ValueError("Request details require a 'requests' count vector")
            return options
        counts = np.asarray(options["requests"])
        if (counts.shape != (c.n_zones,) or not np.issubdtype(counts.dtype, np.integer)
                or (counts < 0).any() or (counts > np.iinfo(np.int64).max).any()):
            raise ValueError("requests must be nonnegative integer counts of length n_zones")
        counts = counts.astype(np.int64).copy()
        n = sum(int(x) for x in counts)
        q = np.asarray(options.get("request_energy_kwh", np.full(n, c.request_kwh)), dtype=float)
        if (q.shape != (n,) or not np.isfinite(q).all()
                or (q < c.request_min_kwh).any() or (q > c.request_max_kwh).any()):
            raise ValueError("request_energy_kwh must have one bounded energy per request")
        result = {"requests": counts, "request_energy_kwh": q.copy()}
        if "accessible_fcs" in options:
            access = np.asarray(options["accessible_fcs"])
            if access.shape != (n, c.n_fcs) or not np.isin(access, [0, 1]).all():
                raise ValueError("accessible_fcs must be a request-by-station binary matrix")
            result["accessible_fcs"] = access.astype(bool).copy()
        return result

    def reset(self, *, seed=None, options=None):
        options = self._validate_options(options)  # Reject malformed input before RNG/state mutation.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.period = 0
        self.stage = 0
        self.locations = np.array(c.initial_locations, dtype=np.int64)
        self.energy = np.array(c.initial_energy_kwh, dtype=float)
        self.cumulative_profit = 0.0
        self.cumulative_discounted_profit = 0.0
        self._prepare_period(options)
        self._ready = True
        return self.observe(), {"stage": "pricing", "discount": 1.0}

    def _arrivals(self):
        return self.rng.poisson(self.config.resolved_arrival_rates[self.period]).astype(np.int64)

    def _prepare_period(self, options=None, *, terminal=False):
        c = self.config
        options = options or {}
        if terminal:
            self.requests = np.zeros(c.n_zones, dtype=np.int64)
        else:
            self.requests = options["requests"].copy() if "requests" in options else self._arrivals()
        self.request_origins = np.repeat(np.arange(c.n_zones, dtype=np.int64), self.requests)
        n = len(self.request_origins)
        self.request_energy_kwh = (options["request_energy_kwh"].copy()
            if "request_energy_kwh" in options else self.rng.uniform(c.request_min_kwh, c.request_max_kwh, n))
        self.request_choices = np.full(n, -1, dtype=np.int64)
        distances = c.distances[self.request_origins][:, c.fcs_zones]
        self.accessible_fcs = (distances <= c.max_fcs_access_km) & (
            np.asarray(c.fcs_ev_capacity_kwh) >= c.request_min_kwh)
        if "accessible_fcs" in options:
            # Explicit J_r overrides the geometric default; zero-capacity stations remain unavailable.
            self.accessible_fcs = options["accessible_fcs"].copy() & (
                np.asarray(c.fcs_ev_capacity_kwh) >= c.request_min_kwh)
        self._refresh_demand()
        self.potential_energy_kwh = np.bincount(self.request_origins,
            weights=self.request_energy_kwh, minlength=c.n_zones).astype(float)
        self.eligible_mcvs = np.array([[self.service_capacity_kwh(m, z) + 1e-9 >= c.request_min_kwh
            for z in range(c.n_zones)] for m in range(c.n_mcv)], dtype=bool)
        self.mobile_available = self.eligible_mcvs.any(axis=0)
        self.mobile_wait_hours = np.zeros(c.n_zones)
        for z in np.flatnonzero(self.mobile_available):
            self.mobile_wait_hours[z] = np.mean(c.distances[self.locations[self.eligible_mcvs[:, z]], z]) / c.speed_kmph
        # Pre-choice congestion estimate, independent of fleet exchange and realized choices.
        access_count = self.accessible_fcs.sum(axis=1)
        load = (self.accessible_fcs * (self.request_energy_kwh / np.maximum(1, access_count))[:, None]).sum(axis=0)
        capacity = np.asarray(c.fcs_ev_capacity_kwh, dtype=float)
        ratio = np.divide(load, capacity, out=np.zeros(c.n_fcs), where=capacity > 0)
        self.fcs_wait_hours = np.maximum(0., ratio - 1.) * c.period_hours / 2.
        t = min(self.period, c.horizon - 1)
        if not terminal:
            if c.fcs_pricing_mode == "base_price":
                # One independent base-price draw per station, from this period's
                # type band. The draw itself carries the price randomness, so the
                # legacy multiplicative shock is not applied on top of it.
                low, high = c.resolved_base_price_bounds[t]
                self.fcs_base_prices = self.rng.uniform(low, high, c.n_fcs)
                self.fcs_retail_prices = self.fcs_base_prices * (1. + c.fcs_service_fee_rate)
                self.fcs_discharge_prices = self.fcs_base_prices.copy()
            else:
                if c.price_noise_distribution == "uniform":
                    noise = self.rng.uniform(-c.price_noise_scale, c.price_noise_scale, c.n_fcs)
                elif c.price_noise_distribution == "normal":
                    noise = self.rng.normal(0., c.price_noise_scale, c.n_fcs)
                else:
                    noise = np.zeros(c.n_fcs)
                factor = np.maximum(0., 1. + noise)
                self.fcs_retail_prices = c.resolved_fcs_charge_price_profile[t] * factor
                self.fcs_discharge_prices = c.resolved_fcs_discharge_price_profile[t] * factor
                # Legacy mode has no single exchange price; recharge uses grid_buy.
                self.fcs_base_prices = self.fcs_discharge_prices.copy()
            self.mobile_price_menu = self._price_menu()
        else:
            self.eligible_mcvs.fill(False)
            self.mobile_available.fill(False)
            self.mobile_wait_hours.fill(0.)
        self.mobile_prices = self.mobile_price_menu[:, 0].copy()

    def _price_menu(self):
        """Admissible mobile prices [zone, level] for the current station prices."""
        return self.config.mobile_price_menu(self.fcs_retail_prices)

    def _refresh_demand(self):
        """Aggregate retained request choices without losing identities or energy needs."""
        c = self.config
        mobile = self.request_choices == 0
        self.mobile_demand = np.bincount(self.request_origins[mobile], minlength=c.n_zones).astype(np.int64)
        self.mobile_requested_kwh = np.bincount(self.request_origins[mobile],
            weights=self.request_energy_kwh[mobile], minlength=c.n_zones).astype(float)
        self.fixed_demand = np.array([np.count_nonzero(self.request_choices == f + 1)
            for f in range(c.n_fcs)], dtype=np.int64)
        self.fixed_requested_kwh = np.array([self.request_energy_kwh[self.request_choices == f + 1].sum()
            for f in range(c.n_fcs)], dtype=float)
        self.outside = int(np.count_nonzero(self.request_choices == c.n_fcs + 1))

    def _require(self, stage):
        if not self._ready:
            raise RuntimeError("Call reset() first")
        if self.stage != stage:
            raise RuntimeError(f"Expected stage {stage}; current stage is {self.stage}")

    def _travel(self, m, zone):
        c = self.config
        distance = float(c.distances[self.locations[m], zone])
        return distance, c.period_hours - distance / c.speed_kmph, distance * c.travel_kwh_per_km

    def service_capacity_kwh(self, m, zone):
        c = self.config
        _, hours, travel = self._travel(m, zone)
        return max(0., min((self.energy[m] - travel - c.reserve_kwh) * c.mobile_service_efficiency,
                           hours * c.mobile_service_kw))

    def service_limit(self, m, zone):
        """Legacy nominal-q capacity; heterogeneous service uses service_capacity_kwh."""
        return max(0, int(np.floor((self.service_capacity_kwh(m, zone) + 1e-9) / self.config.request_kwh)))

    def _validate_prices(self, prices):
        c = self.config
        prices = np.asarray(prices, dtype=float)
        if prices.shape != (c.n_zones,) or not np.isfinite(prices).all():
            raise ValueError("prices must be a finite vector of length n_zones")
        menu = self.mobile_price_menu
        matches = np.isclose(prices[:, None], menu, rtol=0., atol=1e-8)
        if not matches.any(axis=1).all():
            raise ValueError("Every node price must belong to that node's current price menu")
        return np.take_along_axis(menu, matches.argmax(axis=1)[:, None], axis=1).ravel().copy()

    def choice_probabilities(self, prices):
        """One MNL row per request: [mobile, station 0, ..., station F-1, outside]."""
        self._require(0)
        c = self.config
        prices = self._validate_prices(prices)
        z, q = self.request_origins, self.request_energy_kwh
        mobile_cost = prices[z] * q + c.mobile_wait_cost_per_hour * self.mobile_wait_hours[z]
        fixed_cost = (q[:, None] * self.fcs_retail_prices
            + c.fixed_travel_cost_per_km * c.distances[z][:, c.fcs_zones]
            + c.fixed_wait_cost_per_hour * self.fcs_wait_hours)
        return charging_choice_probabilities(
            c, mobile_cost, fixed_cost, self.mobile_available[z], self.accessible_fcs)

    def quote_price_levels(self, levels):
        self._require(0)
        c = self.config
        levels = np.asarray(levels)
        if (levels.shape != (c.n_zones,) or not np.issubdtype(levels.dtype, np.integer)
                or (levels < 0).any() or (levels >= c.n_price_levels).any()):
            raise ValueError("Provide one valid integer price level per node")
        return self.quote_prices(self.mobile_price_menu[np.arange(c.n_zones), levels])

    def quote_prices(self, prices):
        self._require(0)
        prices = self._validate_prices(prices)
        probabilities = self.choice_probabilities(prices)
        draws = self.rng.random(len(self.request_origins))
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.
        self.request_choices = (draws[:, None] >= cumulative).sum(axis=1).astype(np.int64)
        self.mobile_prices = prices
        self._refresh_demand()
        counts = np.zeros((self.config.n_zones, self.config.n_fcs + 2), dtype=np.int64)
        np.add.at(counts, (self.request_origins, self.request_choices), 1)
        self.stage = 1
        return self.observe(), 0.0, False, False, {
            "stage": "dispatch", "choice_probabilities": probabilities.copy(),
            "choice_counts_by_zone": counts, "request_choices": self.request_choices.copy(), "discount": 1.0,
        }

    def feasible_actions(self):
        """Individual [MCV, mode, zone] masks; shared capacity is resolved jointly."""
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
                q = self.request_energy_kwh[(self.request_choices == 0) & (self.request_origins == z)]
                mask[m, Mode.SERVE, z] = bool(np.any(q <= self.service_capacity_kwh(m, z) + 1e-9))
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
            if isinstance(z, (bool, np.bool_)) or not isinstance(z, (int, np.integer)) or not 0 <= z < c.n_zones:
                raise ValueError("target_zone must be a valid integer zone index")
            if (isinstance(action.energy_kwh, (bool, np.bool_)) or not isinstance(action.energy_kwh, Real)
                    or not np.isfinite(action.energy_kwh) or action.energy_kwh < 0):
                raise ValueError("energy_kwh must be finite and nonnegative")
            clean.append(DispatchAction(mode, int(z), float(action.energy_kwh)))
        return clean

    def dispatch(self, actions):
        self._require(1)
        actions = self._validate_actions(actions)
        c = self.config
        before = self.energy.copy()
        mask = self.feasible_actions()
        served_mobile = np.zeros((c.n_mcv, c.n_zones), dtype=np.int64)
        delivered_mobile = np.zeros((c.n_mcv, c.n_zones))
        served = np.zeros(len(self.request_origins), dtype=bool)
        assigned_mcv = np.full(len(served), -1, dtype=np.int64)
        travel_kwh, travel_km, charge_grid, export_grid = [np.zeros(c.n_mcv) for _ in range(4)]
        fcs_charge_grid, fcs_export_grid = np.zeros(c.n_fcs), np.zeros(c.n_fcs)
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
                remaining = self.service_capacity_kwh(m, z)
                candidates = np.flatnonzero((self.request_choices == 0) & (self.request_origins == z) & ~served)
                for r in candidates:
                    q = self.request_energy_kwh[r]
                    if q <= remaining + 1e-9:
                        served[r], assigned_mcv[r] = True, m
                        remaining -= q
                        served_mobile[m, z] += 1
                        delivered_mobile[m, z] += q
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
        mobile_delivered = delivered_mobile.sum(axis=1)
        self.energy += c.charge_efficiency*charge_grid
        self.energy -= mobile_delivered/c.mobile_service_efficiency + export_grid/c.discharge_efficiency
        served_fixed = np.zeros(c.n_fcs, dtype=np.int64)
        fixed_delivered = np.zeros(c.n_fcs)
        for f in range(c.n_fcs):
            remaining = c.fcs_ev_capacity_kwh[f]
            for r in np.flatnonzero(self.request_choices == f + 1):
                q = self.request_energy_kwh[r]
                if q <= remaining + 1e-9:
                    served[r] = True
                    remaining -= q
                    served_fixed[f] += 1
                    fixed_delivered[f] += q
        fixed_grid = fixed_delivered/c.fixed_service_efficiency
        unmet_mobile_mask = (self.request_choices == 0) & ~served
        unmet_fixed_mask = (self.request_choices >= 1) & (self.request_choices <= c.n_fcs) & ~served
        unmet_mobile, unmet_fixed = int(unmet_mobile_mask.sum()), int(unmet_fixed_mask.sum())
        unmet_mobile_kwh = float(self.request_energy_kwh[unmet_mobile_mask].sum())
        unmet_fixed_kwh = float(self.request_energy_kwh[unmet_fixed_mask].sum())
        t = self.period
        terminal = t + 1 == c.horizon
        terms = {
            "mobile_service_revenue": float(np.sum(delivered_mobile*self.mobile_prices)),
            "fixed_service_revenue": float(np.dot(self.fcs_retail_prices, fixed_delivered)),
            "discharge_revenue": float(np.dot(self.fcs_discharge_prices, fcs_export_grid)),
            "procurement_cost": float(c.grid_buy_per_kwh[t]*fixed_grid.sum()
                + (np.dot(self.fcs_base_prices, fcs_charge_grid) if c.fcs_pricing_mode == "base_price"
                   else c.grid_buy_per_kwh[t]*charge_grid.sum())),
            "travel_cost": float(c.travel_cost_per_km*travel_km.sum()),
            "operating_cost": float(c.operating_cost_per_mcv_period*c.n_mcv),
            "unmet_demand_cost": float(c.unmet_penalty_per_kwh*(unmet_mobile_kwh+unmet_fixed_kwh)),
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
        if (not np.allclose(residual, 0., atol=1e-8) or (self.energy < c.reserve_kwh-1e-8).any()
                or (self.energy > c.battery_capacity_kwh+1e-8).any()):
            raise AssertionError("Energy conservation or battery bounds violated")
        total_requests = len(served)
        fcs_capacity = np.asarray(c.fcs_ev_capacity_kwh, dtype=float)
        info = {
            "completed_period": t, "stage": "terminal" if terminal else "pricing",
            "reward_terms": terms, "served_mobile": served_mobile, "delivered_mobile_kwh": delivered_mobile,
            "served_fixed": served_fixed, "served_request_mask": served, "assigned_mcv": assigned_mcv,
            "request_origins": self.request_origins.copy(), "request_energy_kwh": self.request_energy_kwh.copy(),
            "request_choices": self.request_choices.copy(),
            "unmet_mobile_requests": unmet_mobile, "unmet_fixed_requests": unmet_fixed,
            "unmet_mobile_kwh": unmet_mobile_kwh, "unmet_fixed_kwh": unmet_fixed_kwh,
            "outside_requests": self.outside, "total_requests": total_requests, "invalid_actions": invalid,
            "executed_modes": modes, "cumulative_profit": self.cumulative_profit,
            "cumulative_discounted_profit": self.cumulative_discounted_profit,
            "discount": 0. if terminal else c.discount_per_period,
            "metrics": {"outside_option_share": self.outside / total_requests if total_requests else 0.,
                "served_request_share": float(served.sum()) / total_requests if total_requests else 0.,
                "mcv_active_fraction": float(np.mean(modes != Mode.WAIT)),
                "mcv_service_utilization": float(mobile_delivered.sum()/(c.n_mcv*c.mobile_service_kw*c.period_hours)),
                "fcs_service_utilization": np.divide(fixed_delivered, fcs_capacity,
                    out=np.zeros(c.n_fcs), where=fcs_capacity > 0)},
            "energy_ledger": {"before_kwh": before, "after_kwh": self.energy.copy(),
                "travel_kwh": travel_kwh, "mobile_delivered_kwh": mobile_delivered,
                "charge_grid_kwh": charge_grid, "export_grid_kwh": export_grid,
                "fcs_charge_grid_kwh": fcs_charge_grid, "fcs_export_grid_kwh": fcs_export_grid,
                "fixed_delivered_kwh": fixed_delivered, "fixed_grid_kwh": fixed_grid,
                "balance_residual_kwh": residual},
        }
        self.period += 1
        self.stage = 2 if terminal else 0
        self._prepare_period(terminal=terminal)
        return self.observe(), float(reward), terminal, False, info

    def observe(self):
        if not self._ready:
            raise RuntimeError("Call reset() first")
        c = self.config
        t = min(self.period, c.horizon-1)
        details = np.column_stack((self.request_origins, self.request_energy_kwh,
                                   self.request_choices, self.accessible_fcs)).astype(np.float64)
        pricing_mask = np.repeat(self.mobile_available[:, None], c.n_price_levels, axis=1)
        if self.stage != 0:
            pricing_mask.fill(False)
        return {
            "stage": self.stage, "period": np.array([self.period], dtype=np.int64),
            "locations": self.locations.copy(), "energy_kwh": self.energy.copy(),
            "arrival_rates": c.resolved_arrival_rates[t].copy() if self.stage != 2 else np.zeros(c.n_zones),
            "requests_by_zone": self.requests.copy(), "potential_energy_kwh": self.potential_energy_kwh.copy(),
            "request_details": details, "mobile_prices": self.mobile_prices.copy(),
            "mobile_demand": self.mobile_demand.copy(), "fixed_demand": self.fixed_demand.copy(),
            "mobile_requested_kwh": self.mobile_requested_kwh.copy(), "fixed_requested_kwh": self.fixed_requested_kwh.copy(),
            "outside_requests": np.array([self.outside], dtype=np.int64),
            "mobile_wait_hours": self.mobile_wait_hours.copy(), "mobile_available": self.mobile_available.astype(np.int8),
            "eligible_mcvs": self.eligible_mcvs.astype(np.int8), "fcs_wait_hours": self.fcs_wait_hours.copy(),
            "fcs_retail_prices": self.fcs_retail_prices.copy(), "fcs_discharge_prices": self.fcs_discharge_prices.copy(),
            "fcs_base_prices": self.fcs_base_prices.copy(), "mobile_price_menu": self.mobile_price_menu.copy(),
            "grid_buy_price": np.array([c.grid_buy_per_kwh[t]], dtype=float),
            "pricing_mask": pricing_mask.astype(np.int8), "action_mask": self.feasible_actions().astype(np.int8),
        }

    def local_observation(self, m):
        """Shared-policy dispatch input including peer fleet state (zeta_mt)."""
        self._require(1)
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, (int, np.integer)) or not 0 <= m < self.config.n_mcv:
            raise ValueError("Invalid MCV id")
        obs = self.observe()
        peers = np.arange(self.config.n_mcv) != m
        obs["agent_id"] = np.array([m], dtype=np.int64)
        obs["other_locations"] = self.locations[peers].copy()
        obs["other_energy_kwh"] = self.energy[peers].copy()
        for key in ("locations", "energy_kwh", "action_mask"):
            obs[key] = obs[key][m:m+1].copy()
        return obs
