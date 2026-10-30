"""Gymnasium adapter. One step advances ONE decision stage, not one full period."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import Scenario
from .simulator import ChargingSimulator, DispatchAction, Mode


class ChargingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scenario=None, render_mode=None):
        super().__init__()
        self.simulator = ChargingSimulator(scenario or Scenario())
        c = self.simulator.config
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        maximum_exchange = max(c.mobile_charge_kw, c.mobile_discharge_kw)*c.period_hours
        self.action_space = spaces.Dict({
            "price_levels": spaces.MultiDiscrete(np.full(c.n_zones, c.n_price_levels)),
            "modes": spaces.MultiDiscrete(np.full(c.n_mcv, len(Mode))),
            "targets": spaces.MultiDiscrete(np.full(c.n_mcv, c.n_zones)),
            "energy_kwh": spaces.Box(0., maximum_exchange, (c.n_mcv,), dtype=np.float64),
        })
        # Poisson arrivals have no finite model cap.  int64 is the storage limit,
        # not a sampling/truncation threshold.
        def count(shape, maximum=np.iinfo(np.int64).max):
            return spaces.Box(0, maximum, shape, dtype=np.int64)
        def nonnegative(shape):
            return spaces.Box(0., np.inf, shape, dtype=np.float64)
        request_low = np.array([0., c.request_min_kwh, -1.] + [0.]*c.n_fcs)
        request_high = np.array([c.n_zones-1., c.request_max_kwh, c.n_fcs+1.] + [1.]*c.n_fcs)
        self.observation_space = spaces.Dict({
            "stage": spaces.Discrete(3), "period": count((1,), c.horizon),
            "locations": spaces.MultiDiscrete(np.full(c.n_mcv, c.n_zones)),
            "energy_kwh": spaces.Box(0., c.battery_capacity_kwh, (c.n_mcv,), dtype=np.float64),
            "requests_by_zone": count((c.n_zones,)),
            "arrival_rates": nonnegative((c.n_zones,)),
            "potential_energy_kwh": nonnegative((c.n_zones,)),
            "request_details": spaces.Sequence(spaces.Box(request_low, request_high, dtype=np.float64), stack=True),
            "mobile_prices": spaces.Box(c.price_min, c.price_max, (c.n_zones,), dtype=np.float64),
            "mobile_demand": count((c.n_zones,)),
            "mobile_requested_kwh": nonnegative((c.n_zones,)),
            "fixed_requested_kwh": nonnegative((c.n_fcs,)),
            "fixed_demand": count((c.n_fcs,)), "outside_requests": count((1,)),
            "mobile_wait_hours": nonnegative((c.n_zones,)),
            "mobile_available": spaces.MultiBinary(c.n_zones),
            "eligible_mcvs": spaces.MultiBinary((c.n_mcv, c.n_zones)),
            "fcs_wait_hours": nonnegative((c.n_fcs,)),
            # Normal multiplicative shocks are clipped below at zero, but have
            # no upper bound; observation bounds must allow their full support.
            "fcs_retail_prices": nonnegative((c.n_fcs,)),
            "fcs_discharge_prices": nonnegative((c.n_fcs,)),
            "fcs_base_prices": nonnegative((c.n_fcs,)),
            "mobile_price_menu": nonnegative((c.n_zones, c.n_price_levels)),
            "grid_buy_price": spaces.Box(0., max(c.grid_buy_per_kwh), (1,), dtype=np.float64),
            "pricing_mask": spaces.MultiBinary((c.n_zones, c.n_price_levels)),
            "action_mask": spaces.MultiBinary((c.n_mcv, len(Mode), c.n_zones)),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Share Gymnasium's RNG so its checker can verify stochastic transitions.
        self.simulator.rng = self.np_random
        return self.simulator.reset(options=options)

    def step(self, action):
        if not self.simulator._ready:
            raise RuntimeError("Call reset() first")
        if not self.action_space.contains(action):
            raise ValueError("Action does not match action_space (use correct shapes and dtypes)")
        if self.simulator.stage == 0:
            return self.simulator.quote_price_levels(action["price_levels"])
        return self.simulator.dispatch([
            DispatchAction(Mode(int(mode)), int(target), float(kwh))
            for mode, target, kwh in zip(action["modes"], action["targets"], action["energy_kwh"])
        ])

    def local_observation(self, m):
        return self.simulator.local_observation(m)

    def render(self):
        s = self.simulator
        if not s._ready:
            return "Call reset() to start."
        stages = ("pricing", "dispatch", "terminal")
        return (f"Period {s.period}/{s.config.horizon}, stage={stages[s.stage]}\n"
                f"MCV zones={s.locations.tolist()}, battery kWh={s.energy.round(2).tolist()}\n"
                f"Requests by zone={s.requests.tolist()}, cumulative profit=${s.cumulative_profit:.2f}")

