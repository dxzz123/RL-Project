"""Large-fleet local search preserves full-fleet physical allocation."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks.policies import coordinated_dispatch, dispatch_outcome
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, Mode


def scenario(vehicles=6, **changes):
    base = dict(horizon=3, zone_xy_km=((0., 0.), (2., 0.), (4., 0.)),
        # Dispatch/rollout/history regressions: keep the fixed absolute menu and
        # profile station prices. tests/test_station_pricing.py covers the LADWP
        # base-price bands and the premium menu.
        fcs_pricing_mode='legacy', mcv_pricing_mode='absolute',
                fcs_zones=(0,), initial_locations=(0,)*vehicles,
                initial_energy_kwh=(40.,)*vehicles,
                mean_requests_by_zone=(3., 4., 3.), arrival_profile=(1.,)*3,
                grid_buy_per_kwh=(.15, .2, .25), grid_sell_per_kwh=(.1,)*3,
                fcs_retail_per_kwh=(.5,), fcs_ev_capacity_kwh=(40.,),
                fcs_fleet_exchange_capacity_kwh=(30.,),
                charge_efficiency=.8, discharge_efficiency=.8,
                mobile_service_efficiency=1., fixed_service_efficiency=1.,
                price_noise_distribution='none', charging_utility_intercept=3.)
    return replace(Scenario(), **(base | changes)).validate()


def observed(config, seed=7, *, counts=None, energies=None, choices=None):
    sim = ChargingSimulator(config)
    options = None if counts is None else dict(requests=counts, request_energy_kwh=energies)
    sim.reset(seed=seed, options=options)
    sim.quote_prices(np.full(config.n_zones, .6))
    if choices is not None:
        sim.request_choices = np.asarray(choices, dtype=np.int64)
        sim._refresh_demand()
    return sim


class ScalableInventoryTests(unittest.TestCase):
    def compare_physics(self, sim, commands):
        expected = dispatch_outcome(sim.observe(), sim.config, commands)
        obs, reward, _, _, info = sim.dispatch(commands)
        self.assertAlmostEqual(reward, expected['reward'], places=10)
        self.assertEqual(info['invalid_actions'], [])
        np.testing.assert_allclose(expected['next_energy_kwh'], obs['energy_kwh'], atol=1e-10)
        np.testing.assert_array_equal(expected['served_request_mask'], info['served_request_mask'])
        np.testing.assert_array_equal(expected['assigned_mcv'], info['assigned_mcv'])
        np.testing.assert_allclose(expected['charge_grid_kwh'], info['energy_ledger']['charge_grid_kwh'])
        np.testing.assert_allclose(expected['export_grid_kwh'], info['energy_ledger']['export_grid_kwh'])
        np.testing.assert_allclose(info['energy_ledger']['balance_residual_kwh'], 0., atol=1e-10)
        return info

    def test_six_vehicle_mixed_exchange_shares_gross_station_capacity(self):
        config = scenario(initial_energy_kwh=(20.,)*3+(60.,)*3)
        sim = observed(config, counts=[2, 0, 0], energies=[10., 10.], choices=[1, 1])
        mask = np.zeros_like(sim.observe()['action_mask'])
        mask[:3, Mode.RECHARGE, 0] = 1
        mask[3:, Mode.DISCHARGE, 0] = 1
        commands = coordinate_inventory_dispatch(sim.observe(), config, action_mask=mask)
        self.assertEqual([a.energy_kwh for a in commands], [30.]*3+[20.]*3)
        info = self.compare_physics(sim, commands)
        np.testing.assert_allclose(info['energy_ledger']['charge_grid_kwh'], [6., 6., 6., 0., 0., 0.])
        np.testing.assert_allclose(info['energy_ledger']['export_grid_kwh'], [0., 0., 0., 4., 4., 4.])
        self.assertAlmostEqual(float(info['energy_ledger']['fixed_delivered_kwh'].sum()), 20.)

    def test_competing_service_does_not_duplicate_requests(self):
        config = scenario()
        sim = observed(config, counts=[7, 0, 0], energies=[10.]*7, choices=[0]*7)
        mask = np.zeros_like(sim.observe()['action_mask'])
        mask[:, Mode.SERVE, 0] = 1
        commands = coordinate_inventory_dispatch(sim.observe(), config, action_mask=mask)
        info = self.compare_physics(sim, commands)
        self.assertEqual(int(info['served_mobile'].sum()), 7)
        np.testing.assert_array_equal(info['assigned_mcv'], [0, 0, 0, 1, 1, 1, 2])

    def test_search_is_monotone_bounded_deterministic_and_does_not_use_joint_enumeration(self):
        config = scenario(initial_locations=(0, 1, 2, 0, 1, 2))
        sim = observed(config)
        obs = sim.observe()
        old = deepcopy(obs)
        rng_before = np.random.get_state()
        sim_rng_before = deepcopy(sim.rng.bit_generator.state)
        diagnostics = {}
        with patch('charging_env.benchmarks.policies.joint_dispatch_outcomes',
                   side_effect=AssertionError('Exhaustive enumeration is forbidden')):
            first = coordinate_inventory_dispatch(obs, config, sweeps=2, diagnostics=diagnostics)
            count = diagnostics['candidate_outcomes']
            second = coordinate_inventory_dispatch(obs, config, sweeps=2, diagnostics=diagnostics)
        self.assertEqual(first, second)
        self.assertEqual(diagnostics['calls'], 2)
        self.assertEqual(diagnostics['candidate_outcomes'], 2*count)
        call = diagnostics['last_call']
        self.assertLessEqual(call['candidate_outcomes'], call['candidate_budget'])
        self.assertLessEqual(call['sweeps_completed'], 2*call['starts'])
        self.assertLessEqual(call['starts'], 2)
        for trace in call['accepted_score_traces']:
            self.assertTrue(np.all(np.diff(trace) > 0))
        self.assertGreaterEqual(call['selected_score'], max(call['initial_scores'])-1e-10)
        predicted = dispatch_outcome(obs, config, first)
        expected_score = predicted['reward']+.3*(predicted['next_energy_kwh'].sum()-obs['energy_kwh'].sum())
        self.assertAlmostEqual(call['selected_score'], expected_score, places=10)
        for key in obs:
            np.testing.assert_array_equal(obs[key], old[key])
        self.assertEqual(sim_rng_before, sim.rng.bit_generator.state)
        rng_after = np.random.get_state()
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])

    def test_terminal_shadow_is_zero_and_preserves_wait_on_exact_ties(self):
        config = scenario(grid_sell_per_kwh=(0.,)*3, travel_cost_per_km=0.,
                          travel_kwh_per_km=0.)
        sim = observed(config, counts=[0, 0, 0], energies=[])
        sim.period = config.horizon-1
        obs = sim.observe()
        low, high = {}, {}
        a = coordinate_inventory_dispatch(obs, config, shadow_price=0., diagnostics=low)
        b = coordinate_inventory_dispatch(obs, config, shadow_price=100., diagnostics=high)
        self.assertEqual(a, b)
        self.assertTrue(all(x.mode == Mode.WAIT for x in a))
        self.assertEqual(high['last_call']['effective_shadow_price'], 0.)
        self.assertAlmostEqual(high['last_call']['selected_score'], dispatch_outcome(obs, config, b)['reward'])

    def test_restricted_mask_and_maximum_exchange_are_respected(self):
        config = scenario(initial_energy_kwh=(20.,)*6)
        sim = observed(config, counts=[0, 0, 0], energies=[])
        obs = sim.observe()
        mask = np.zeros_like(obs['action_mask'])
        mask[:, Mode.RECHARGE, 0] = 1
        commands = coordinate_inventory_dispatch(obs, config, action_mask=mask)
        self.assertTrue(all(a.mode == Mode.RECHARGE and a.target_zone == 0 for a in commands))
        self.assertEqual([a.energy_kwh for a in commands], [30.]*6)
        # Projected service-first would request only residual station capacity;
        # full-grid labels must still request30 before shared allocation.
        info = self.compare_physics(sim, commands)
        self.assertAlmostEqual(float(info['energy_ledger']['charge_grid_kwh'].sum()), 30.)

    def test_two_vehicle_score_does_not_exceed_exhaustive_optimum(self):
        config = scenario(vehicles=2, initial_locations=(0, 2), initial_energy_kwh=(20., 55.))
        for seed in [7, 19, 37]:
            obs = observed(config, seed=seed).observe()
            local = coordinate_inventory_dispatch(obs, config, shadow_price=.3, energy_fractions=(.5, 1.))
            exhaustive = coordinated_dispatch(obs, config, shadow_price=.3, energy_fractions=(.5, 1.))
            def score(actions):
                result = dispatch_outcome(obs, config, actions)
                return result['reward']+.3*result['next_energy_kwh'].sum()
            self.assertLessEqual(score(local), score(exhaustive)+1e-9)

    def test_invalid_search_parameters_and_masks_are_rejected(self):
        config = scenario()
        obs = observed(config).observe()
        for change in [dict(sweeps=0), dict(sweeps=True), dict(sweeps=1.5),
                       dict(shadow_price=-1), dict(shadow_price=float('nan')),
                       dict(energy_fractions=()), dict(energy_fractions=(1.1,)),
                       dict(energy_fractions=(True,)), dict(diagnostics=[])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                coordinate_inventory_dispatch(obs, config, **change)
        for mask in [np.zeros((2, 2)), np.full_like(obs['action_mask'], 2),
                     np.ones_like(obs['action_mask']), np.zeros_like(obs['action_mask'])]:
            with self.assertRaises(ValueError):
                coordinate_inventory_dispatch(obs, config, action_mask=mask)
        with self.assertRaises(ValueError):
            coordinate_inventory_dispatch(dict(obs, stage=0), config)


if __name__ == '__main__':
    unittest.main()
