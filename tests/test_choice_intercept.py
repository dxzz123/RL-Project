"""Analytic outside-option checks, benchmark consistency, and saved scenarios."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from charging_env.choice import charging_choice_probabilities
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator

try:
    import torch
except ImportError:
    torch = None


def tiny_scenario(intercept=0., **overrides):
    settings = dict(
        horizon=2, zone_xy_km=((0., 0.), (2., 0.)), fcs_zones=(0, 1),
        initial_locations=(0, 1), initial_energy_kwh=(40., 45.),
        mean_requests_by_zone=(3., 4.), arrival_profile=(1., 1.),
        grid_buy_per_kwh=(.12, .18), grid_sell_per_kwh=(.09, .14),
        fcs_retail_per_kwh=(.42, .55), fcs_ev_capacity_kwh=(30., 40.),
        fcs_fleet_exchange_capacity_kwh=(20., 25.),
        price_noise_distribution='none', charging_utility_intercept=intercept,
    )
    settings.update(overrides)
    return replace(Scenario(), **settings).validate()


def analytical_row(config, mobile_cost, fixed_cost, mobile_available, accessible):
    """Direct scalar MNL calculation, independent of the shared vectorized helper."""
    weights = [np.exp(config.charging_utility_intercept
                      - config.price_sensitivity_per_dollar * mobile_cost)
               if mobile_available else 0.]
    weights.extend(np.exp(config.charging_utility_intercept
                          - config.price_sensitivity_per_dollar * cost) if available else 0.
                   for cost, available in zip(fixed_cost, accessible))
    weights.append(1.)
    return np.asarray(weights) / sum(weights)


class ChoiceInterceptTests(unittest.TestCase):
    def test_heterogeneous_request_probabilities_match_independent_cost_formula(self):
        access = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
        prices = np.array([.3, .8])
        for intercept in (0., 3.):
            with self.subTest(intercept=intercept):
                config = tiny_scenario(intercept)
                sim = ChargingSimulator(config)
                obs, _ = sim.reset(seed=91, options={
                    'requests': [2, 2], 'request_energy_kwh': [5., 14., 7., 12.],
                    'accessible_fcs': access,
                })
                expected = []
                for row in obs['request_details']:
                    zone, energy = int(row[0]), row[1]
                    mobile_cost = prices[zone]*energy + config.mobile_wait_cost_per_hour*obs['mobile_wait_hours'][zone]
                    fixed_cost = [energy*obs['fcs_retail_prices'][f]
                                  + config.fixed_travel_cost_per_km*config.distances[zone, fz]
                                  + config.fixed_wait_cost_per_hour*obs['fcs_wait_hours'][f]
                                  for f, fz in enumerate(config.fcs_zones)]
                    expected.append(analytical_row(config, mobile_cost, fixed_cost,
                                                   obs['mobile_available'][zone], row[3:]))
                expected = np.asarray(expected)
                actual = sim.choice_probabilities(prices)
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
                self.assertFalse(np.allclose(actual[0], actual[1]))
                _, _, _, _, info = sim.quote_prices(prices)
                np.testing.assert_array_equal(info['choice_probabilities'], actual)
                np.testing.assert_array_equal(actual[:, 1:-1][~access], 0.)

    def test_intercept_changes_participation_but_preserves_conditional_provider_shares(self):
        mobile_cost = np.array([1., 3., 8.])
        fixed_cost = np.array([[2., 4.], [5., 2.], [4., 7.]])
        access = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
        mobile_available = np.array([1, 0, 1], dtype=bool)
        zero = charging_choice_probabilities(tiny_scenario(), mobile_cost, fixed_cost,
                                             mobile_available, access)
        positive = charging_choice_probabilities(tiny_scenario(3.), mobile_cost, fixed_cost,
                                                 mobile_available, access)
        negative = charging_choice_probabilities(tiny_scenario(-2.), mobile_cost, fixed_cost,
                                                 mobile_available, access)
        self.assertTrue((positive[:, -1] < zero[:, -1]).all())
        self.assertTrue((zero[:, -1] < negative[:, -1]).all())
        np.testing.assert_allclose(positive[:, :-1]/positive[:, :-1].sum(axis=1, keepdims=True),
                                   zero[:, :-1]/zero[:, :-1].sum(axis=1, keepdims=True),
                                   rtol=1e-14, atol=1e-15)
        # The common service intercept multiplies total service/outside odds by exp(b).
        np.testing.assert_allclose((1.-positive[:, -1])/positive[:, -1],
                                   np.exp(3.)*(1.-zero[:, -1])/zero[:, -1], rtol=1e-14)

    def test_scalar_and_broadcast_price_batches_keep_request_accessibility(self):
        config = tiny_scenario(3.)
        mobile_cost = np.array([[1., 4., 8.], [2., 5., 9.]])
        fixed_cost = np.array([[3., 5.], [7., 2.], [6., 4.]])
        available = np.array([True, False, True])
        access = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
        actual = charging_choice_probabilities(config, mobile_cost, fixed_cost, available, access)
        self.assertEqual(actual.shape, (2, 3, 4))
        for price in range(2):
            for request in range(3):
                expected = analytical_row(config, mobile_cost[price, request], fixed_cost[request],
                                          available[request], access[request])
                np.testing.assert_allclose(actual[price, request], expected, rtol=1e-14, atol=1e-15)
        scalar = charging_choice_probabilities(config, 1., np.array([3., 5.]), True, [True, False])
        self.assertEqual(scalar.shape, (4,))
        np.testing.assert_allclose(scalar, actual[0, 0], rtol=0., atol=0.)

    def test_no_available_provider_has_certain_outside_choice_at_every_intercept(self):
        for intercept in (-10000., 0., 3., 10000.):
            with self.subTest(intercept=intercept):
                config = tiny_scenario(intercept, initial_energy_kwh=(5., 5.))
                sim = ChargingSimulator(config)
                obs, _ = sim.reset(seed=81, options={
                    'requests': [1, 1], 'request_energy_kwh': [5., 15.],
                    'accessible_fcs': [[0, 0], [0, 0]],
                })
                self.assertFalse(obs['mobile_available'].any())
                expected = np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]])
                np.testing.assert_array_equal(sim.choice_probabilities([.3, .8]), expected)
                _, _, _, _, info = sim.quote_prices([.3, .8])
                np.testing.assert_array_equal(info['request_choices'], [3, 3])

    def test_empty_request_sets_work_in_helper_and_simulator(self):
        config = tiny_scenario(3.)
        actual = charging_choice_probabilities(config, np.empty((5, 0)), np.empty((0, 2)),
                                               np.empty(0, dtype=bool), np.empty((0, 2), dtype=bool))
        self.assertEqual(actual.shape, (5, 0, 4))
        sim = ChargingSimulator(config)
        sim.reset(seed=71, options={'requests': [0, 0]})
        self.assertEqual(sim.choice_probabilities([.3, .8]).shape, (0, 4))
        post, reward, terminated, truncated, info = sim.quote_prices([.3, .8])
        self.assertEqual(info['choice_probabilities'].shape, (0, 4))
        self.assertEqual(post['request_details'].shape, (0, 5))
        self.assertEqual(reward, 0.)
        self.assertFalse(terminated or truncated)

    def test_extreme_finite_intercepts_remain_normalized_and_keep_masks(self):
        mobile_cost = np.array([2., 3., 1.])
        fixed_cost = np.array([[4., 1.], [2., 8.], [3., 7.]])
        available = np.array([True, False, False])
        access = np.array([[1, 1], [0, 1], [0, 0]], dtype=bool)
        allowed = np.column_stack((available, access, np.ones(3, dtype=bool)))
        for intercept in (-10000., 10000.):
            with self.subTest(intercept=intercept), np.errstate(over='raise', invalid='raise'):
                actual = charging_choice_probabilities(tiny_scenario(intercept), mobile_cost, fixed_cost,
                                                       available, access)
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual >= 0.).all())
                np.testing.assert_allclose(actual.sum(axis=1), 1., rtol=0., atol=1e-15)
                np.testing.assert_array_equal(actual[~allowed], 0.)
                np.testing.assert_array_equal(actual[2], [0., 0., 0., 1.])
                if intercept < 0:
                    np.testing.assert_array_equal(actual[:, -1], 1.)
                else:
                    np.testing.assert_array_equal(actual[:2, -1], 0.)
                    costs = np.array([mobile_cost[0], *fixed_cost[0]])
                    expected = np.exp(-tiny_scenario().price_sensitivity_per_dollar*costs)
                    # Subtracting utilities near 10,000 loses a few low bits.
                    np.testing.assert_allclose(actual[0, :-1], expected/expected.sum(), rtol=1e-12, atol=2e-13)

    def test_config_rejects_nonfinite_or_nonreal_intercepts_and_accepts_negative_values(self):
        for invalid in (np.nan, np.inf, -np.inf, True, np.bool_(False), '3', None, 1+0j):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                replace(Scenario(), charging_utility_intercept=invalid).validate()
        for valid in (-3., 0, 3., np.float64(2.5)):
            with self.subTest(value=valid):
                config = replace(Scenario(), charging_utility_intercept=valid).validate()
                self.assertEqual(config.charging_utility_intercept, valid)
                self.assertEqual(config.price_sensitivity_per_dollar, Scenario().price_sensitivity_per_dollar)

    def test_json_roundtrip_and_missing_legacy_field_default_to_zero(self):
        self.assertEqual(Scenario().charging_utility_intercept, 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scenario.json'
            for intercept in (-2., 0., 3.):
                scenario = tiny_scenario(intercept)
                scenario.to_json(path)
                self.assertEqual(json.loads(path.read_text())['charging_utility_intercept'], intercept)
                self.assertEqual(Scenario.from_json(path), scenario)
            legacy = asdict(tiny_scenario())
            legacy.pop('charging_utility_intercept')
            path.write_text(json.dumps(legacy))
            self.assertEqual(Scenario.from_json(path), tiny_scenario())

    def test_adaptive_benchmark_matches_independent_expected_margin_at_nonzero_intercept(self):
        from charging_env.benchmarks.policies import adaptive_prices
        # Nonbinding capacities isolate the pricing rule's MNL expectation from
        # its fluid capacity approximation, with heterogeneous energy and access.
        for intercept in (0., 3.):
            with self.subTest(intercept=intercept):
                config = tiny_scenario(intercept, battery_capacity_kwh=1000.,
                    initial_energy_kwh=(1000., 1000.), mobile_service_kw=1000.,
                    fcs_ev_capacity_kwh=(1000., 1000.))
                sim = ChargingSimulator(config)
                obs, _ = sim.reset(seed=61, options={
                    'requests': [2, 1], 'request_energy_kwh': [5., 9., 14.],
                    'accessible_fcs': [[1, 0], [1, 1], [0, 1]],
                })
                expected = []
                for zone in range(config.n_zones):
                    scores = []
                    for price in config.price_levels:
                        score = 0.
                        for row in obs['request_details']:
                            if int(row[0]) != zone:
                                continue
                            energy = row[1]
                            mc = price*energy + config.mobile_wait_cost_per_hour*obs['mobile_wait_hours'][zone]
                            fc = [energy*obs['fcs_retail_prices'][f]
                                  + config.fixed_travel_cost_per_km*config.distances[zone, fz]
                                  + config.fixed_wait_cost_per_hour*obs['fcs_wait_hours'][f]
                                  for f, fz in enumerate(config.fcs_zones)]
                            probs = analytical_row(config, mc, fc, obs['mobile_available'][zone], row[3:])
                            replacement = obs['grid_buy_price'][0]/(config.charge_efficiency*config.mobile_service_efficiency)
                            score += (price-replacement)*energy*probs[0]
                            score += sum((obs['fcs_retail_prices'][f]
                                          - obs['grid_buy_price'][0]/config.fixed_service_efficiency)
                                         * energy * probs[f+1] for f in range(config.n_fcs))
                        scores.append(score)
                    expected.append(int(np.argmax(scores)))
                np.testing.assert_array_equal(adaptive_prices(obs, config), expected)


@unittest.skipIf(torch is None, 'Install the train extra to test saved scenarios')
class ChoiceInterceptCheckpointTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        self.Trainer = Trainer
        self.config = TrainingConfig(seed=53, updates=2, episodes_per_update=2,
            epochs=2, minibatch_size=4, hidden_size=16, eval_episodes=2,
            dispatch_final_activation='silu', device='cpu', torch_threads=1)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_legacy_checkpoint_missing_scenario_intercept_preserves_exact_policy(self):
        trainer = self.Trainer(tiny_scenario(), replace(self.config, dispatch_final_activation='tanh'))
        expected = {mode: trainer.evaluate([3101, 3103], deterministic=mode, action_seed=671)
                    for mode in (False, True)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['scenario'].pop('charging_utility_intercept')
            payload['training_config'].pop('dispatch_final_activation')
            torch.save(payload, path)
            restored = self.Trainer.load_checkpoint(path)
            self.assertEqual(restored.scenario.charging_utility_intercept, 0.)
            self.assertEqual(restored.config.dispatch_final_activation, 'tanh')
            for mode in (False, True):
                self.assertEqual(restored.evaluate([3101, 3103], deterministic=mode, action_seed=671),
                                 expected[mode])

    def test_tiny_silu_intercept_three_training_and_saved_rng_resume_are_exact(self):
        trainer = self.Trainer(tiny_scenario(3.), self.config)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run = Path(directory)/'tiny'
            summary = trainer.train(run)
            path = run/'checkpoint.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            expected_evaluation = trainer.evaluate([4101, 4103], deterministic=False, action_seed=971)
            expected_rollout = trainer.collect_rollout()
            expected_metrics = trainer.update(expected_rollout)
            expected_torch_rng = torch.get_rng_state().clone()
            expected_numpy_rng = deepcopy(trainer.rng.bit_generator.state)
            restored = self.Trainer.load_checkpoint(path)
            self.assertEqual(saved['scenario']['charging_utility_intercept'], 3.)
            self.assertEqual(restored.scenario, tiny_scenario(3.))
            self.assertEqual(restored.config.dispatch_final_activation, 'silu')
            self.assertEqual(restored.evaluate([4101, 4103], deterministic=False, action_seed=971),
                             expected_evaluation)
            actual_rollout = restored.collect_rollout()
            for field in ('pricing_actions', 'dispatch_actions', 'energy_fractions',
                          'old_price_log_probs', 'old_dispatch_log_probs', 'rewards',
                          'price_values', 'dispatch_values', 'next_price_values', 'dones'):
                np.testing.assert_array_equal(getattr(actual_rollout, field), getattr(expected_rollout, field))
            self.assertEqual(restored.update(actual_rollout), expected_metrics)
            torch.testing.assert_close(torch.get_rng_state(), expected_torch_rng, rtol=0., atol=0.)
            self.assertEqual(restored.rng.bit_generator.state, expected_numpy_rng)
            for name, model in trainer._modules().items():
                self.assert_nested_equal(restored._modules()[name].state_dict(), model.state_dict())
            self.assert_nested_equal(restored.actor_optimizer.state_dict(), trainer.actor_optimizer.state_dict())
            self.assert_nested_equal(restored.critic_optimizer.state_dict(), trainer.critic_optimizer.state_dict())
        self.assertTrue(summary['all_metrics_finite'])
        self.assertTrue(summary['checkpoint_reload_matches'])
        self.assertEqual(summary['invalid_evaluation_actions'], 0)
        self.assertEqual(summary['training_physical_periods'], 8)


if __name__ == '__main__':
    unittest.main()
