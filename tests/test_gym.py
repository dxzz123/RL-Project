import unittest
import warnings

import numpy as np
from gymnasium.utils.env_checker import check_env

from charging_env.gym_env import ChargingEnv


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


if __name__ == "__main__":
    unittest.main()
