"""Physical and behavioral assumptions, independent of any learning algorithm."""
from dataclasses import asdict, dataclass, fields
from functools import cached_property
import json
from numbers import Real
from pathlib import Path

import numpy as np

# Experience-based time-of-use classes, ordered so that index 0 < 1 < 2 in price.
PERIOD_TYPES = ("off_peak", "normal", "peak")

# LADWP Electric Vehicle Commercial Charging Service (Pilot) weekday windows.
# Peak 4pm-9pm, mid peak 7am-4pm and 9pm-11pm, off peak 11pm-7am. Source:
# acquia.ladwp.com .../electric-vehicle-commercial-charging-service-monthly-rates
_LADWP_PEAK_HOURS = frozenset(range(16, 21))
_LADWP_MID_PEAK_HOURS = frozenset(range(7, 16)) | {21, 22}


def price_level_indices(menu, prices):
    """Recover the level index each quoted price came from, one per zone."""
    menu = np.asarray(menu, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if menu.ndim != 2 or prices.shape != (menu.shape[0],):
        raise ValueError("prices must hold one value per row of the menu")
    return np.abs(menu - prices[:, None]).argmin(axis=1).astype(np.int64)


def ladwp_period_type(hour):
    """Index into PERIOD_TYPES for a clock hour, wrapping past midnight."""
    hour = int(hour) % 24
    if hour in _LADWP_PEAK_HOURS:
        return PERIOD_TYPES.index("peak")
    if hour in _LADWP_MID_PEAK_HOURS:
        return PERIOD_TYPES.index("normal")
    return PERIOD_TYPES.index("off_peak")


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
    # --- Station base energy price -------------------------------------------
    # "base_price": every period is typed off-peak/normal/peak, and each station
    # draws its base price p^FCS from Uniform(mean -+ halfwidth) for that type.
    # An EV pays base * (1 + fcs_service_fee_rate); the fleet exchanges energy
    # (recharge and discharge alike) at the base price itself.
    # "legacy": the original fcs_retail_per_kwh / grid_sell_per_kwh profiles with
    # a multiplicative price_noise shock, and fleet recharge at grid_buy_per_kwh.
    fcs_pricing_mode: str = "base_price"
    # Explicit per-period type names; derived from the LADWP windows when None.
    period_types: tuple | None = None
    first_period_hour_of_day: int = 14
    # Defaults are LADWP EVA2a energy charges, effective 10/1/2026 - 12/31/2026:
    # off peak $0.24368, mid peak $0.27739, peak $0.36667 per kWh.
    fcs_base_price_mean: tuple = (0.24368, 0.27739, 0.36667)
    # Half-widths keep the three supports disjoint and increasing in width, so
    # peak > normal > off-peak holds for every draw, not only in expectation.
    fcs_base_price_halfwidth: tuple = (0.01218, 0.01387, 0.02933)
    fcs_service_fee_rate: float = 0.10
    # --- Mobile retail price --------------------------------------------------
    # "premium": the menu at zone i is mcv_premium_levels times the reference
    # price of the station nearest to i, so it moves with the base price.
    # "absolute": the original fixed price_levels menu, shared by every zone.
    mcv_pricing_mode: str = "premium"
    mcv_premium_levels: tuple = (1.1, 1.4, 1.7)
    # "ev_price" anchors on base * (1 + fee); "base_price" anchors on base.
    mcv_premium_reference: str = "ev_price"

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
    def n_price_levels(self):
        """Number of admissible mobile pricing actions at a zone."""
        if self.mcv_pricing_mode == "premium":
            return len(self.mcv_premium_levels)
        return len(self.price_levels)

    @property
    def resolved_period_types(self):
        """Index into PERIOD_TYPES for every period [horizon]."""
        if self.period_types is not None:
            return np.array([PERIOD_TYPES.index(name) for name in self.period_types], dtype=np.int64)
        hours = self.first_period_hour_of_day + np.arange(self.horizon) * self.period_hours
        return np.array([ladwp_period_type(hour) for hour in hours], dtype=np.int64)

    @property
    def resolved_base_price_bounds(self):
        """Uniform support (low, high) of the base price in every period [horizon, 2]."""
        mean = np.asarray(self.fcs_base_price_mean, dtype=float)[self.resolved_period_types]
        halfwidth = np.asarray(self.fcs_base_price_halfwidth, dtype=float)[self.resolved_period_types]
        return np.stack([mean - halfwidth, mean + halfwidth], axis=1)

    @cached_property
    def nearest_fcs(self):
        """Station index anchoring each zone's mobile price menu; ties take the lower index."""
        return self.distances[:, list(self.fcs_zones)].argmin(axis=1).astype(np.int64)

    @property
    def resolved_fcs_charge_price_profile(self):
        """Baseline FCS customer charging prices [period, station]."""
        if self.fcs_charge_price_profile is not None:
            return np.asarray(self.fcs_charge_price_profile, dtype=float)
        if self.fcs_pricing_mode == "base_price":
            mean = np.asarray(self.fcs_base_price_mean, dtype=float)[self.resolved_period_types]
            return np.tile((mean * (1. + self.fcs_service_fee_rate))[:, None], (1, self.n_fcs))
        return np.tile(np.asarray(self.fcs_retail_per_kwh, dtype=float), (self.horizon, 1))

    @property
    def resolved_fcs_discharge_price_profile(self):
        """Baseline fleet discharge prices [period, station]."""
        if self.fcs_discharge_price_profile is not None:
            return np.asarray(self.fcs_discharge_price_profile, dtype=float)
        if self.fcs_pricing_mode == "base_price":
            mean = np.asarray(self.fcs_base_price_mean, dtype=float)[self.resolved_period_types]
            return np.tile(mean[:, None], (1, self.n_fcs))
        return np.repeat(np.asarray(self.grid_sell_per_kwh, dtype=float)[:, None], self.n_fcs, axis=1)

    def mobile_price_menu(self, fcs_retail_prices):
        """Admissible mobile prices [zone, level] implied by station customer prices.

        Premium mode anchors each zone on its nearest station, so the menu is a
        function of the station prices alone and carries no request information.
        This is the single definition; the simulator and the pricing controllers
        both go through it.
        """
        retail = np.asarray(fcs_retail_prices, dtype=float)
        if retail.shape != (self.n_fcs,) or not np.isfinite(retail).all():
            raise ValueError("fcs_retail_prices must hold one finite price per station")
        if self.mcv_pricing_mode != "premium":
            return np.tile(np.asarray(self.price_levels, dtype=float), (self.n_zones, 1))
        anchor = retail[self.nearest_fcs]
        if self.mcv_premium_reference == "base_price":
            anchor = anchor / (1. + self.fcs_service_fee_rate)
        return np.outer(anchor, np.asarray(self.mcv_premium_levels, dtype=float))

    def representative_price(self, index):
        """A single dollar price standing in for menu level `index`.

        Averages that level across zones and periods at the expected base price.
        In absolute mode this returns ``price_levels[index]`` exactly, so callers
        that need one constant price per level work unchanged in both modes.
        """
        profile = self.resolved_fcs_charge_price_profile
        menus = np.stack([self.mobile_price_menu(profile[t]) for t in range(self.horizon)])
        return float(menus[:, :, int(index)].mean())

    @property
    def mobile_price_bounds(self):
        """Lowest and highest mobile price this scenario can ever quote."""
        if self.mcv_pricing_mode != "premium":
            levels = np.asarray(self.price_levels, dtype=float)
            return float(levels.min()), float(levels.max())
        bounds = self.resolved_base_price_bounds * self.base_price_reference_scale()
        premiums = np.asarray(self.mcv_premium_levels, dtype=float)
        return float(bounds[:, 0].min() * premiums.min()), float(bounds[:, 1].max() * premiums.max())

    def price_setting(self, index):
        """The menu entry an action index selects: a premium factor, or a dollar price."""
        levels = self.mcv_premium_levels if self.mcv_pricing_mode == "premium" else self.price_levels
        return float(levels[int(index)])

    def base_price_reference_scale(self):
        """Factor turning a base price into a mobile menu anchor."""
        return 1. + self.fcs_service_fee_rate if self.mcv_premium_reference == "ev_price" else 1.

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
        if self.fcs_pricing_mode not in ("base_price", "legacy"):
            raise ValueError("fcs_pricing_mode must be 'base_price' or 'legacy'")
        if self.mcv_pricing_mode not in ("premium", "absolute"):
            raise ValueError("mcv_pricing_mode must be 'premium' or 'absolute'")
        if self.mcv_premium_reference not in ("ev_price", "base_price"):
            raise ValueError("mcv_premium_reference must be 'ev_price' or 'base_price'")
        hour = self.first_period_hour_of_day
        if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour < 24:
            raise ValueError("first_period_hour_of_day must be an integer clock hour in [0, 24)")
        if self.period_types is not None:
            if (len(self.period_types) != self.horizon
                    or any(name not in PERIOD_TYPES for name in self.period_types)):
                raise ValueError(f"period_types must give one of {PERIOD_TYPES} for each of {self.horizon} periods")
        fee = self.fcs_service_fee_rate
        if isinstance(fee, bool) or not isinstance(fee, Real) or not np.isfinite(fee) or fee < 0:
            raise ValueError("fcs_service_fee_rate must be a finite nonnegative number")
        mean = np.asarray(self.fcs_base_price_mean, dtype=float)
        halfwidth = np.asarray(self.fcs_base_price_halfwidth, dtype=float)
        for name, array in (("fcs_base_price_mean", mean), ("fcs_base_price_halfwidth", halfwidth)):
            if array.shape != (len(PERIOD_TYPES),) or not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"{name} must contain {len(PERIOD_TYPES)} finite nonnegative values")
        if (halfwidth >= mean).any():
            raise ValueError("fcs_base_price_halfwidth must stay below the matching mean, so base prices are positive")
        # Disjoint, increasing supports make peak > normal > off-peak hold per draw.
        if (mean[:-1] + halfwidth[:-1] >= mean[1:] - halfwidth[1:]).any():
            raise ValueError(f"Base price supports must be disjoint and increasing across {PERIOD_TYPES}")
        if (np.diff(halfwidth) <= 0).any():
            raise ValueError(f"Base price ranges must widen across {PERIOD_TYPES}")
        premiums = np.asarray(self.mcv_premium_levels, dtype=float)
        if (premiums.ndim != 1 or premiums.size == 0 or not np.isfinite(premiums).all()
                or (premiums <= 0).any() or (np.diff(premiums) <= 0).any()):
            raise ValueError("mcv_premium_levels must be finite, positive, and strictly increasing")
        if self.fcs_pricing_mode == "base_price":
            explicit = ("fcs_charge_price_profile", "fcs_discharge_price_profile")
            if any(getattr(self, name) is not None for name in explicit):
                raise ValueError(f"base_price mode samples station prices; clear {explicit} or use 'legacy' mode")
        if self.mcv_pricing_mode == "premium":
            if self.fcs_pricing_mode != "base_price":
                raise ValueError("premium mobile pricing requires the base_price station pricing mode")
            low, high = self.mobile_price_bounds
            if low < self.price_min - 1e-12 or high > self.price_max + 1e-12:
                raise ValueError(f"Premium mobile prices span [{low:.4f}, {high:.4f}], "
                                 f"outside price_min/price_max [{self.price_min}, {self.price_max}]")
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
