"""Physical and behavioral assumptions, independent of any learning algorithm."""
from dataclasses import asdict, dataclass, fields
from functools import cached_property
import json
from numbers import Real
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scenario:
    horizon: int = 12
    period_hours: float = 1.0
    zone_xy_km: tuple = ((0., 0.), (4., 0.), (0., 4.), (4., 4.))
    fcs_zones: tuple = (0, 3)
    initial_locations: tuple = (1, 2)
    battery_capacity_kwh: float = 80.0
    initial_energy_kwh: tuple = (55., 55.)
    reserve_kwh: float = 5.0
    request_min_kwh: float = 5.0
    request_max_kwh: float = 15.0
    # Used for deterministic count-only request overrides, not sampled arrivals.
    request_kwh: float = 10.0
    # Legacy metadata only: Poisson realizations and their means are unbounded.
    max_requests_per_zone: int = 12
    mean_requests_by_zone: tuple = (3., 4., 3., 4.)
    arrival_profile: tuple = (0.7, 0.7, 0.8, 1., 1.2, 1.4, 1.4, 1.2, 1., 0.9, 0.8, 0.7)
    arrival_rates: tuple | None = None
    fcs_retail_per_kwh: tuple = (0.42, 0.44)
    grid_buy_per_kwh: tuple = (0.15, 0.15, 0.16, 0.18, 0.22, 0.26, 0.26, 0.22, 0.18, 0.16, 0.15, 0.15)
    grid_sell_per_kwh: tuple = (0.10, 0.10, 0.11, 0.13, 0.16, 0.19, 0.19, 0.16, 0.13, 0.11, 0.10, 0.10)
    fcs_charge_price_profile: tuple | None = None
    fcs_discharge_price_profile: tuple | None = None
    # One multiplicative shock per station/period, shared by its two prices.
    # Normal shocks may be below -1; the simulator clips realized prices to 0.
    price_noise_distribution: str = "uniform"
    price_noise_scale: float = 0.05
    price_min: float = 0.20
    price_max: float = 1.00
    price_levels: tuple = (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
    # eta in exp(b - eta * monetary_cost); outside utility is exactly zero.
    price_sensitivity_per_dollar: float = 0.7
    mobile_wait_cost_per_hour: float = 1.0
    fixed_wait_cost_per_hour: float = 1.0
    fixed_travel_cost_per_km: float = 0.25
    max_fcs_access_km: float = 8.0
    speed_kmph: float = 20.0
    travel_kwh_per_km: float = 0.35
    travel_cost_per_km: float = 0.25
    mobile_service_kw: float = 30.0
    mobile_charge_kw: float = 30.0
    mobile_discharge_kw: float = 20.0
    charge_efficiency: float = 0.95
    mobile_service_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    fixed_service_efficiency: float = 0.95
    fcs_ev_capacity_kwh: tuple = (40., 40.)
    fcs_fleet_exchange_capacity_kwh: tuple = (40., 40.)
    operating_cost_per_mcv_period: float = 0.5
    unmet_penalty_per_kwh: float = 0.30
    terminal_value_per_stored_kwh: float = 0.0
    discount_per_period: float = 0.99
    # Common service utility relative to outside; zero preserves the original model.
    charging_utility_intercept: float = 0.0

    @property
    def n_zones(self):
        return len(self.zone_xy_km)

    @property
    def n_mcv(self):
        return len(self.initial_locations)

    @property
    def n_fcs(self):
        return len(self.fcs_zones)

    @cached_property
    def distances(self):
        """Pairwise Manhattan distances in km for an axis-aligned road grid."""
        xy = np.asarray(self.zone_xy_km, dtype=float)
        return np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2)

    @property
    def resolved_arrival_rates(self):
        """Prescribed Poisson means [period, zone], without count truncation."""
        if self.arrival_rates is not None:
            return np.asarray(self.arrival_rates, dtype=float)
        return np.outer(self.arrival_profile, self.mean_requests_by_zone)

    @property
    def resolved_fcs_charge_price_profile(self):
        """Baseline FCS customer charging prices [period, station]."""
        if self.fcs_charge_price_profile is not None:
            return np.asarray(self.fcs_charge_price_profile, dtype=float)
        return np.tile(np.asarray(self.fcs_retail_per_kwh, dtype=float), (self.horizon, 1))

    @property
    def resolved_fcs_discharge_price_profile(self):
        """Baseline fleet discharge prices [period, station]."""
        if self.fcs_discharge_price_profile is not None:
            return np.asarray(self.fcs_discharge_price_profile, dtype=float)
        return np.repeat(np.asarray(self.grid_sell_per_kwh, dtype=float)[:, None], self.n_fcs, axis=1)

    def validate(self):
        integer_fields = ("horizon", "max_requests_per_zone")
        for name in integer_fields:
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.n_zones, self.n_mcv, self.n_fcs) < 1:
            raise ValueError("At least one zone, MCV, and FCS is required")
        xy = np.asarray(self.zone_xy_km, dtype=float)
        if xy.shape != (self.n_zones, 2) or not np.isfinite(xy).all():
            raise ValueError("zone_xy_km must be finite [n_zones, 2] coordinates")
        expected = {
            "initial_energy_kwh": self.n_mcv, "grid_buy_per_kwh": self.horizon,
            "fcs_ev_capacity_kwh": self.n_fcs,
            "fcs_fleet_exchange_capacity_kwh": self.n_fcs,
        }
        if self.arrival_rates is None:
            expected.update(mean_requests_by_zone=self.n_zones, arrival_profile=self.horizon)
        if self.fcs_charge_price_profile is None:
            expected["fcs_retail_per_kwh"] = self.n_fcs
        if self.fcs_discharge_price_profile is None:
            expected["grid_sell_per_kwh"] = self.horizon
        for name, size in expected.items():
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (size,) or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"{name} must contain {size} finite nonnegative values")
        for name in ("initial_locations", "fcs_zones"):
            for index in getattr(self, name):
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.n_zones:
                    raise ValueError(f"{name} must contain valid integer zone indices")
        if len(set(self.fcs_zones)) != self.n_fcs:
            raise ValueError("Use one FCS per zone in this version")
        positive = ("period_hours", "battery_capacity_kwh", "request_kwh", "speed_kmph",
                    "request_min_kwh", "request_max_kwh", "price_sensitivity_per_dollar",
                    "mobile_service_kw", "mobile_charge_kw", "mobile_discharge_kw")
        nonnegative = ("reserve_kwh", "price_min", "price_max", "price_noise_scale",
                       "mobile_wait_cost_per_hour", "fixed_wait_cost_per_hour", "fixed_travel_cost_per_km",
                       "max_fcs_access_km", "travel_kwh_per_km", "travel_cost_per_km",
                       "operating_cost_per_mcv_period", "unmet_penalty_per_kwh",
                       "terminal_value_per_stored_kwh")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0 or (name in positive and value == 0)):
                raise ValueError(f"Invalid {name}")
        intercept = self.charging_utility_intercept
        if (isinstance(intercept, bool) or not isinstance(intercept, Real)
                or not np.isfinite(intercept)):
            raise ValueError("charging_utility_intercept must be a finite real number")
        for name in ("charge_efficiency", "mobile_service_efficiency", "discharge_efficiency",
                     "fixed_service_efficiency", "discount_per_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.reserve_kwh > self.battery_capacity_kwh or self.price_min > self.price_max:
            raise ValueError("Invalid battery reserve or price bounds")
        if self.request_min_kwh >= self.request_max_kwh:
            raise ValueError("request_min_kwh must be strictly less than request_max_kwh")
        if not self.request_min_kwh <= self.request_kwh <= self.request_max_kwh:
            raise ValueError("request_kwh must lie within the request energy bounds")
        levels = np.asarray(self.price_levels, dtype=float)
        if (levels.ndim != 1 or levels.size == 0 or not np.isfinite(levels).all()
                or (np.diff(levels) <= 0).any()
                or (levels < self.price_min).any() or (levels > self.price_max).any()):
            raise ValueError("price_levels must be finite, strictly increasing, and within price_min/price_max")
        if self.price_noise_distribution not in ("none", "uniform", "normal"):
            raise ValueError("price_noise_distribution must be 'none', 'uniform', or 'normal'")
        if self.price_noise_distribution == "uniform" and self.price_noise_scale >= 1:
            raise ValueError("Uniform price_noise_scale must be in [0, 1)")
        e = np.asarray(self.initial_energy_kwh)
        if (e < self.reserve_kwh).any() or (e > self.battery_capacity_kwh).any():
            raise ValueError("Initial energy must lie between reserve and capacity")
        matrices = (
            ("arrival_rates", self.resolved_arrival_rates, (self.horizon, self.n_zones)),
            ("fcs_charge_price_profile", self.resolved_fcs_charge_price_profile, (self.horizon, self.n_fcs)),
            ("fcs_discharge_price_profile", self.resolved_fcs_discharge_price_profile, (self.horizon, self.n_fcs)),
        )
        for name, matrix, shape in matrices:
            if matrix.shape != shape or not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(f"{name} must contain finite nonnegative values with shape {shape}")
        return self

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown scenario settings: {sorted(unknown)}")
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, list) else value
        return cls(**{key: tuples(value) for key, value in data.items()}).validate()

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
