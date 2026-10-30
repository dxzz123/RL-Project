import unittest
import warnings
from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from charging_env.gym_env import ChargingEnv
from charging_env.config import Scenario


class GymInterfaceTests(unittest.TestCase):
    def test_official_checker(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*symmetric and normalized.*")
            check_env(ChargingEnv(), skip_render_check=True)

    def test_complete_episode_spaces_and_two_stage_timing(self):
        env = ChargingEnv()
        obs, info = env.reset(seed=7)
        env.action_space.seed(11)
        self.assertTrue(env.observation_space.contains(obs))
        for period in range(env.simulator.config.horizon):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(int(obs["period"][0]), period)
            self.assertEqual(obs["stage"], 1)
            self.assertEqual(reward, 0.)
            self.assertEqual(info["discount"], 1.)
            self.assertFalse(terminated or truncated)
            self.assertTrue(env.observation_space.contains(obs))
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            self.assertEqual(int(obs["period"][0]), period+1)
            self.assertEqual(terminated, period == env.simulator.config.horizon-1)
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
        with self.assertRaises(RuntimeError):
            env.step(env.action_space.sample())

    def test_render_and_seed_repeatability(self):
        env = ChargingEnv(render_mode="ansi")
        self.assertIn("reset", env.render())
        first, _ = env.reset(seed=4)
        second, _ = env.reset(seed=4)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertIn("pricing", env.render())

    def test_empty_requests_and_terminal_observations_have_exact_space_dtypes(self):
        scenario = replace(Scenario(), mean_requests_by_zone=(0., 0., 0., 0.),
                            fcs_pricing_mode="legacy", mcv_pricing_mode="absolute")
        env = ChargingEnv(scenario)
        obs, _ = env.reset(seed=7)
        for _ in range(2 * scenario.horizon + 1):
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs["request_details"].shape, (0, 3 + scenario.n_fcs))
            for key, space in env.observation_space.spaces.items():
                if isinstance(obs[key], np.ndarray) and hasattr(space, "dtype") and space.dtype is not None:
                    self.assertEqual(obs[key].dtype, space.dtype, key)
            if obs["stage"] == 2:
                break
            obs, _, _, _, _ = env.step(env.action_space.sample())
        self.assertEqual(obs["stage"], 2)

    def test_unbounded_poisson_request_sequence_and_discrete_node_action(self):
        env = ChargingEnv(replace(Scenario(), mean_requests_by_zone=(100., 100., 100., 100.),
                                        fcs_pricing_mode="legacy", mcv_pricing_mode="absolute"))
        obs, _ = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(obs["request_details"]), int(obs["requests_by_zone"].sum()))
        self.assertGreater(len(obs["request_details"]), 4 * env.simulator.config.max_requests_per_zone)
        action = env.action_space.sample()
        self.assertEqual(action["price_levels"].shape, (env.simulator.config.n_zones,))
        self.assertEqual(action["modes"].shape, (env.simulator.config.n_mcv,))
        obs, _, _, _, _ = env.step(action)
        np.testing.assert_array_equal(obs["mobile_prices"],
            np.asarray(env.simulator.config.price_levels)[action["price_levels"]])
        self.assertTrue(env.observation_space.contains(obs))


if __name__ == "__main__":
    unittest.main()
