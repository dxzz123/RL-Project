"""Online rollout scoring, shared model draws, physical validity, and replay."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks.policies import dispatch_outcome, service_first_dispatch
from charging_env.config import Scenario
from charging_env.rl.fitted_value import sample_dispatch_observation
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy, evaluate_rollout_policy, main
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode


def small_scenario(vehicles=2, horizon=3, **changes):
    values = dict(horizon=horizon, zone_xy_km=((0., 0.), (2., 0.), (4., 0.)),
        fcs_zones=(0,), initial_locations=tuple(i % 3 for i in range(vehicles)),
        initial_energy_kwh=(35.,)*vehicles, mean_requests_by_zone=(2., 3., 2.),
        arrival_profile=(1.,)*horizon, grid_buy_per_kwh=(.15,)*horizon,
        grid_sell_per_kwh=(.1,)*horizon, fcs_retail_per_kwh=(.5,),
        fcs_ev_capacity_kwh=(40.,), fcs_fleet_exchange_capacity_kwh=(30.,),
        price_noise_distribution='none', charging_utility_intercept=3.)
    return replace(Scenario(), **(values | changes)).validate()


def observed(scenario):
    simulator = ChargingSimulator(scenario)
    obs, _ = simulator.reset(seed=19)
    policy = RolloutADPPolicy(scenario, RolloutADPConfig(rollout_samples=2))
    post, _, _, _, _ = simulator.quote_price_levels(policy.prices(obs))
    return simulator, post, policy


class RolloutADPTests(unittest.TestCase):
    def test_one_two_three_ten_vehicles_match_simulator_and_query_budget(self):
        for vehicles in (1, 2, 3, 10):
            with self.subTest(vehicles=vehicles):
                scenario = small_scenario(vehicles)
                simulator, obs, policy = observed(scenario)
                actions = policy.dispatch(obs)
                expected = dispatch_outcome(obs, scenario, actions)
                next_obs, reward, _, _, info = simulator.dispatch(actions)
                self.assertEqual(info['invalid_actions'], [])
                self.assertAlmostEqual(reward, expected['reward'], places=10)
                np.testing.assert_allclose(next_obs['energy_kwh'], expected['next_energy_kwh'])
                trace = policy.decisions[-1]
                self.assertEqual(trace['future_model_queries'], len(trace['candidates'])*2*2)
                self.assertEqual(trace['query_budget'], trace['future_model_queries'])
                self.assertEqual(policy.statistics['future_model_queries'], trace['future_model_queries'])
                self.assertEqual(policy.statistics['exogenous_draws'], 4)

    def test_candidates_share_exact_future_draws_and_saved_scores_replay(self):
        scenario = small_scenario(3)
        _, obs, policy = observed(scenario)
        names, plans, _ = policy._candidate_plans(obs)
        with patch('charging_env.rl.rollout_adp.sample_dispatch_observation',
                   wraps=sample_dispatch_observation) as sampler:
            selected = policy.dispatch(obs)
        trace = policy.decisions[-1]
        seeds = np.asarray(trace['future_draw_seeds'])
        self.assertTrue(np.all(seeds % 2 == 0))
        expected_calls = [int(seeds[offset, draw]) for draw in range(2) for offset in range(2)]
        called = [call.args[4] for call in sampler.call_args_list]
        self.assertEqual(called, expected_calls*len(plans))
        replayed = []
        for commands in plans:
            current = dispatch_outcome(obs, scenario, commands)
            scores = []
            for draw in range(2):
                value, discount = current['reward'], scenario.discount_per_period
                locations, energies = current['next_locations'], current['next_energy_kwh']
                for offset, period in enumerate((1, 2)):
                    future = sample_dispatch_observation(scenario, period, locations,
                        np.clip(energies, scenario.reserve_kwh, scenario.battery_capacity_kwh),
                        int(seeds[offset, draw]), policy.config.resolved_price_index(scenario))
                    step = dispatch_outcome(future, scenario, service_first_dispatch(future, scenario, .2))
                    value += discount*step['reward']
                    discount *= scenario.discount_per_period
                    locations, energies = step['next_locations'], step['next_energy_kwh']
                scores.append(value)
            replayed.append(scores)
        np.testing.assert_array_equal(trace['sampled_total_scores'], replayed)
        winner = int(np.argmax(np.mean(replayed, axis=1)))
        self.assertEqual(selected, plans[winner])
        self.assertEqual(trace['selected'], names[winner])

    def test_terminal_uses_only_current_profit_including_salvage(self):
        scenario = small_scenario(3, 1, terminal_value_per_stored_kwh=1.)
        _, obs, policy = observed(scenario)
        with patch('charging_env.rl.rollout_adp.sample_dispatch_observation',
                   side_effect=AssertionError('No future draws at terminal')):
            actions = policy.dispatch(obs)
        outcome = dispatch_outcome(obs, scenario, actions)
        self.assertGreater(outcome['reward_terms']['terminal_inventory_value'], 0.)
        trace = policy.decisions[-1]
        np.testing.assert_array_equal(trace['mean_total_scores'], trace['current_profit'])
        self.assertEqual(max(trace['mean_total_scores']), outcome['reward'])
        self.assertEqual(trace['future_model_queries'], 0)
        self.assertEqual(trace['future_draw_seeds'], [])

    def test_deduplication_retains_base_and_negative_raw_costs(self):
        scenario = small_scenario(2, 3, initial_locations=(1, 1), initial_energy_kwh=(5., 5.),
            mean_requests_by_zone=(0., 0., 0.), discount_per_period=.5,
            operating_cost_per_mcv_period=2.)
        _, obs, policy = observed(scenario)
        actions = policy.dispatch(obs)
        self.assertEqual(actions, [DispatchAction(Mode.WAIT, 1)]*2)
        trace = policy.decisions[-1]
        self.assertEqual(trace['candidates'], ['service_first_base'])
        self.assertEqual(trace['selected'], 'service_first_base')
        np.testing.assert_array_equal(trace['sampled_total_scores'], [[-7., -7.]])
        self.assertFalse(policy.provenance()['offline_training'])
        self.assertFalse(policy.provenance()['learned_value_bootstrap'])

    def test_global_policy_and_environment_randomness_are_unchanged(self):
        simulator, obs, policy = observed(small_scenario())
        observation_before = deepcopy(obs)
        env_rng = deepcopy(simulator.rng.bit_generator.state)
        external = np.random.default_rng(37)
        external_before = deepcopy(external.bit_generator.state)
        numpy_before, python_before = np.random.get_state(), random.getstate()
        policy.dispatch(obs, external)
        self.assertEqual(env_rng, simulator.rng.bit_generator.state)
        self.assertEqual(external_before, external.bit_generator.state)
        for key in obs:
            np.testing.assert_array_equal(obs[key], observation_before[key])
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_episode_reset_makes_evaluation_order_irrelevant(self):
        scenario, config = small_scenario(), RolloutADPConfig(seed=37, rollout_samples=2)
        first, second = RolloutADPPolicy(scenario, config), RolloutADPPolicy(scenario, config)
        a = evaluate_rollout_policy(first, [17, 23])
        b = evaluate_rollout_policy(second, [23, 17])
        self.assertEqual({e['seed']: e['discounted_profit'] for e in a['episodes']},
                         {e['seed']: e['discounted_profit'] for e in b['episodes']})
        def plans(policy):
            return {(d['episode_key'], d['period']): (d['future_draw_seeds'], d['selected'],
                d['sampled_total_scores']) for d in policy.decisions}
        self.assertEqual(plans(first), plans(second))

    def test_roundoff_correction_is_tiny_recorded_and_not_a_physical_override(self):
        scenario = small_scenario()
        policy = RolloutADPPolicy(scenario)
        corrected = policy._clip_roundoff(np.asarray([scenario.reserve_kwh-1e-12,
                                                      scenario.battery_capacity_kwh+1e-12]))
        np.testing.assert_array_equal(corrected, [scenario.reserve_kwh, scenario.battery_capacity_kwh])
        self.assertEqual(policy.statistics['boundary_roundoff_corrections'], 2)
        self.assertLess(policy.statistics['max_boundary_roundoff_kwh'], 1e-8)
        with self.assertRaises(ValueError):
            policy._clip_roundoff(np.asarray([scenario.reserve_kwh-1e-5, 40.]))

    def test_nonstandard_prices_and_resource_guards_before_sampling(self):
        scenario = small_scenario(price_levels=(.2, .7, .9))
        policy = RolloutADPPolicy(scenario)
        self.assertEqual(policy.config.resolved_price_index(scenario), 1)
        invalid = (dict(seed=True), dict(rollout_samples=0), dict(rollout_samples=1_000_000),
            dict(planner_sweeps=1000), dict(price_index=3), dict(base_recharge_threshold=1.1),
            dict(inventory_early_shadow=-.1), dict(alternative_recharge_thresholds=(.2,)*14),
            dict(energy_fractions=(float('nan'),)), dict(energy_fractions=(True,)))
        with patch('numpy.random.default_rng', side_effect=AssertionError('Validate before sampling')):
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    RolloutADPPolicy(scenario, replace(RolloutADPConfig(), **changes))
            with self.assertRaises(ValueError):
                RolloutADPPolicy(small_scenario(horizon=201), RolloutADPConfig(rollout_samples=256))
            with self.assertRaises(ValueError):
                RolloutADPPolicy(small_scenario(horizon=1), RolloutADPConfig(rollout_samples=1_000_000))

    def test_optional_candidate_is_checked_and_not_used_as_continuation(self):
        scenario = small_scenario()
        _, obs, _ = observed(scenario)
        policy = RolloutADPPolicy(scenario, RolloutADPConfig(rollout_samples=1),
            extra_proposal=lambda observation: [DispatchAction(Mode.WAIT, 99)]*scenario.n_mcv)
        with self.assertRaises(ValueError):
            policy.dispatch(obs)
        self.assertEqual(policy.statistics['future_model_queries'], 0)

    def test_cli_saves_online_evaluation_trace_and_numbers_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'scenario.json'
            path.write_text(json.dumps(asdict(small_scenario(1, 1, price_levels=(.2, .7, .9)))))
            args = ['--scenario', str(path), '--price-index', '1', '--episodes', '1',
                '--eval-seed-base', '831000001', '--rollout-samples', '1', '--output-dir', str(root/'run')]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
            for name in ('run', 'run_2'):
                result = json.loads((root/name/'evaluation.json').read_text())
                provenance = json.loads((root/name/'provenance.json').read_text())
                self.assertEqual(result['evaluation_seeds'], [831000001])
                self.assertEqual(result['invalid_actions'], 0)
                self.assertFalse(provenance['offline_training'])
                self.assertEqual(provenance['fixed_price_index'], 1)
                self.assertTrue((root/name/'planning_trace.json').exists())


if __name__ == '__main__':
    unittest.main()
