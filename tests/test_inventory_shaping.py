"""Potential-shaping telescoping and stage-advantage invariance."""
from copy import deepcopy
import unittest

import numpy as np

from charging_env.rl.returns import inventory_potential_rewards, stage_gae


def observations(energies):
    return [{'energy_kwh': np.asarray(energy, dtype=float)} for energy in energies]


class InventoryShapingTests(unittest.TestCase):
    def test_independent_discounted_telescoping_for_multiple_episodes(self):
        gamma, coefficient = .91, .3
        rewards = np.array([2., -3., 7., 11., -1., 4.])
        pre_obs = observations([[40., 20.], [48., 21.], [31., 10.],
                                [8., 9.], [25., 35.], [20., 12.]])
        dones = np.array([False, False, True, True, False, True])
        original = rewards.copy()
        original_obs = deepcopy(pre_obs)
        shaped = inventory_potential_rewards(rewards, pre_obs, dones, gamma, coefficient)
        for start, end in ((0, 3), (3, 4), (4, 6)):
            discounts = gamma ** np.arange(end-start)
            raw_return = float(np.dot(discounts, rewards[start:end]))
            shaped_return = float(np.dot(discounts, shaped[start:end]))
            expected_change = -coefficient*sum(pre_obs[start]['energy_kwh'])
            self.assertAlmostEqual(shaped_return-raw_return, expected_change, places=12)
        np.testing.assert_array_equal(rewards, original)
        for actual, expected in zip(pre_obs, original_obs):
            np.testing.assert_array_equal(actual['energy_kwh'], expected['energy_kwh'])
        self.assertFalse(np.shares_memory(shaped, rewards))

    def test_terminal_step_and_one_period_episode_have_zero_next_potential(self):
        # Inventory before termination does not imply a bonus for unused stock.
        # The second independent episode starts with a much larger inventory.
        rewards = np.array([6., 9.])
        pre_obs = observations([[20., 30.], [1000., 2000.]])
        shaped = inventory_potential_rewards(rewards, pre_obs, [1, 1], .99, .1)
        np.testing.assert_allclose(shaped, [1., -291.], rtol=0., atol=1e-13)
        changed_next_episode = observations([[20., 30.], [1., 2.]])
        changed = inventory_potential_rewards(rewards, changed_next_episode, [1, 1], .99, .1)
        self.assertEqual(shaped[0], changed[0])

    def test_different_inventory_paths_keep_identical_raw_objective_ranking(self):
        rewards = np.array([1., 3., 5.])
        # Same initial state, then very different replenishment and terminal stock.
        low_stock = observations([[50., 50.], [20., 15.], [5., 5.]])
        high_stock = observations([[50., 50.], [80., 70.], [75., 60.]])
        for gamma in (.9, 1.):
            with self.subTest(gamma=gamma):
                low = inventory_potential_rewards(rewards, low_stock, [0, 0, 1], gamma, .3)
                high = inventory_potential_rewards(rewards, high_stock, [0, 0, 1], gamma, .3)
                discounts = gamma**np.arange(3)
                self.assertAlmostEqual(float(discounts @ low), float(discounts @ high), places=12)
                self.assertAlmostEqual(float(discounts @ high), float(discounts @ rewards)-30., places=12)

    def test_zero_coefficient_is_a_copy_and_empty_input_is_supported(self):
        rewards = np.array([1., -2., 3.], dtype=np.float32)
        shaped = inventory_potential_rewards(rewards, observations([[5.], [9.], [7.]]),
                                             [0, 0, 1], .99, 0.)
        np.testing.assert_array_equal(shaped, rewards)
        shaped[0] = 99.
        self.assertEqual(rewards[0], 1.)
        empty = inventory_potential_rewards([], [], [], .99, .1)
        self.assertEqual(empty.shape, (0,))

    def test_consistently_transformed_critics_preserve_both_stage_advantages(self):
        rewards = np.array([3., -2., 8., 4., 9.])
        pre_obs = observations([[40., 10.], [50., 12.], [30., 15.], [9., 8.], [6., 7.]])
        dones = np.array([0, 0, 1, 0, 1])
        gamma, coefficient = .93, .2
        # Pricing leaves inventory unchanged within the physical period, so
        # both stage values receive the same minus-Phi transformation.
        potential = coefficient*np.array([sum(obs['energy_kwh']) for obs in pre_obs])
        next_potential = np.array([potential[1], potential[2], 0., potential[4], 0.])
        vp = np.array([7., 10., 3., 8., -1.])
        vd = np.array([6., 12., 4., 9., 2.])
        next_vp = np.array([vp[1], vp[2], 0., vp[4], 0.])
        shaped = inventory_potential_rewards(rewards, pre_obs, dones, gamma, coefficient)
        for trace in (0., .5, 1.):
            with self.subTest(trace=trace):
                raw = stage_gae(rewards, vp, vd, next_vp, dones, gamma, trace)
                adjusted = stage_gae(shaped, vp-potential, vd-potential,
                                     next_vp-next_potential, dones, gamma, trace)
                for stage in ('pricing', 'dispatch'):
                    np.testing.assert_allclose(adjusted[stage+'_advantages'], raw[stage+'_advantages'],
                                               rtol=0., atol=5e-15)
                    np.testing.assert_allclose(adjusted[stage+'_returns'], raw[stage+'_returns']-potential,
                                               rtol=0., atol=5e-15)

    def test_invalid_coefficient_discount_and_incomplete_episode_are_rejected(self):
        pre_obs = observations([[10., 20.]])
        for coefficient in (-.1, np.inf, -np.inf, np.nan, True, np.bool_(True), '0.1', None):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                inventory_potential_rewards([1.], pre_obs, [1], .99, coefficient)
        for gamma in (0., -1., 1.1, np.inf, np.nan, True, '0.99'):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                inventory_potential_rewards([1.], pre_obs, [1], gamma, .1)
        with self.assertRaisesRegex(ValueError, 'complete episodes'):
            inventory_potential_rewards([1.], pre_obs, [0], .99, .1)
        for rewards, obs, dones in (([np.nan], pre_obs, [1]), ([1.], [], [1]),
                                    ([1.], pre_obs, [2]), ([[1.]], pre_obs, [[1]])):
            with self.subTest(rewards=rewards, dones=dones), self.assertRaises(ValueError):
                inventory_potential_rewards(rewards, obs, dones, .99, .1)
        with self.assertRaisesRegex(ValueError, 'finite energy'):
            inventory_potential_rewards([1.], observations([[np.inf]]), [1], .99, .1)


if __name__ == '__main__':
    unittest.main()
