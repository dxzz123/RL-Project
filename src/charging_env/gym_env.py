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
            "prices": spaces.Box(c.price_min, c.price_max, (c.n_mcv,), dtype=np.float64),
            "modes": spaces.MultiDiscrete(np.full(c.n_mcv, len(Mode))),
            "targets": spaces.MultiDiscrete(np.full(c.n_mcv, c.n_zones)),
            "energy_kwh": spaces.Box(0., maximum_exchange, (c.n_mcv,), dtype=np.float64),
        })
        total_requests = c.n_zones*c.max_requests_per_zone
        def count(shape, maximum=total_requests):
            return spaces.Box(0, maximum, shape, dtype=np.int64)
        self.observation_space = spaces.Dict({
            "stage": spaces.Discrete(3), "period": count((1,), c.horizon),
            "locations": spaces.MultiDiscrete(np.full(c.n_mcv, c.n_zones)),
            "energy_kwh": spaces.Box(0., c.battery_capacity_kwh, (c.n_mcv,), dtype=np.float64),
            "requests_by_zone": count((c.n_zones,), c.max_requests_per_zone),
            "mobile_prices": spaces.Box(c.price_min, c.price_max, (c.n_mcv,), dtype=np.float64),
            "mobile_demand": count((c.n_mcv, c.n_zones), c.max_requests_per_zone),
            "fixed_demand": count((c.n_fcs,)), "outside_requests": count((1,)),
            "fcs_retail_prices": spaces.Box(0., max(c.fcs_retail_per_kwh), (c.n_fcs,), dtype=np.float64),
            "grid_buy_price": spaces.Box(0., max(c.grid_buy_per_kwh), (1,), dtype=np.float64),
            "grid_sell_price": spaces.Box(0., max(c.grid_sell_per_kwh), (1,), dtype=np.float64),
            "action_mask": spaces.MultiBinary((c.n_mcv, len(Mode), c.n_zones)),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Share Gymnasium's RNG so its checker can verify stochastic transitions.
        self.simulator.rng = self.np_random
        return self.simulator.reset(options=options)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action does not match action_space (use correct shapes and dtypes)")
        if self.simulator.stage == 0:
            return self.simulator.quote_prices(action["prices"])
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

