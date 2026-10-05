"""Physical and behavioral assumptions, independent of any learning algorithm."""
from dataclasses import asdict, dataclass, fields
from functools import cached_property
import json
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
    request_kwh: float = 10.0
    max_requests_per_zone: int = 12
    mean_requests_by_zone: tuple = (3., 4., 3., 4.)
    arrival_profile: tuple = (0.7, 0.7, 0.8, 1., 1.2, 1.4, 1.4, 1.2, 1., 0.9, 0.8, 0.7)
    fcs_retail_per_kwh: tuple = (0.42, 0.44)
    grid_buy_per_kwh: tuple = (0.15, 0.15, 0.16, 0.18, 0.22, 0.26, 0.26, 0.22, 0.18, 0.16, 0.15, 0.15)
    grid_sell_per_kwh: tuple = (0.10, 0.10, 0.11, 0.13, 0.16, 0.19, 0.19, 0.16, 0.13, 0.11, 0.10, 0.10)
    price_min: float = 0.20
    price_max: float = 1.00
    mobile_utility_intercept: float = 3.8
    fixed_utility_intercept: float = 3.2
    outside_utility: float = 0.0
    price_sensitivity_per_dollar: float = 0.7
    mobile_distance_disutility_per_km: float = 0.12
    fixed_distance_disutility_per_km: float = 0.25
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
    terminal_value_per_stored_kwh: float = 0.12
    discount_per_period: float = 0.99

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
        xy = np.asarray(self.zone_xy_km, dtype=float)
        return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)

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
            "initial_energy_kwh": self.n_mcv, "mean_requests_by_zone": self.n_zones,
            "arrival_profile": self.horizon, "grid_buy_per_kwh": self.horizon,
            "grid_sell_per_kwh": self.horizon, "fcs_retail_per_kwh": self.n_fcs,
            "fcs_ev_capacity_kwh": self.n_fcs,
            "fcs_fleet_exchange_capacity_kwh": self.n_fcs,
        }
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
                    "mobile_service_kw", "mobile_charge_kw", "mobile_discharge_kw")
        nonnegative = ("reserve_kwh", "price_min", "price_max", "price_sensitivity_per_dollar",
                       "mobile_distance_disutility_per_km", "fixed_distance_disutility_per_km",
                       "max_fcs_access_km", "travel_kwh_per_km", "travel_cost_per_km",
                       "operating_cost_per_mcv_period", "unmet_penalty_per_kwh",
                       "terminal_value_per_stored_kwh")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"Invalid {name}")
        for name in ("mobile_utility_intercept", "fixed_utility_intercept", "outside_utility"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("charge_efficiency", "mobile_service_efficiency", "discharge_efficiency",
                     "fixed_service_efficiency", "discount_per_period"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.reserve_kwh > self.battery_capacity_kwh or self.price_min > self.price_max:
            raise ValueError("Invalid battery reserve or price bounds")
        e = np.asarray(self.initial_energy_kwh)
        if (e < self.reserve_kwh).any() or (e > self.battery_capacity_kwh).any():
            raise ValueError("Initial energy must lie between reserve and capacity")
        rates = np.outer(self.arrival_profile, self.mean_requests_by_zone)
        if (rates > self.max_requests_per_zone).any():
            raise ValueError("Arrival means exceed max_requests_per_zone")
        if (np.asarray(self.grid_sell_per_kwh) > np.asarray(self.grid_buy_per_kwh)).any():
            raise ValueError("Default model requires grid sell <= grid buy prices")
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
