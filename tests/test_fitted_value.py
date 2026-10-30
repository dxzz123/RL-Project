"""Independent numerical and physical checks for fitted continuation values."""
from copy import deepcopy
from dataclasses import replace
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks.evaluation import evaluate_policy
from charging_env.benchmarks.policies import joint_dispatch_outcomes
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode
from charging_env.rl import fitted_value as fitted


def tiny_scenario(horizon=2, zones=1, **changes):
    settings = dict(
        horizon=horizon, zone_xy_km=tuple((float(z), 0.) for z in range(zones)),
        fcs_zones=(0,), initial_locations=(0, 0),
        battery_capacity_kwh=15., reserve_kwh=5., initial_energy_kwh=(15., 15.),
        mean_requests_by_zone=tuple(2. for _ in range(zones)),
        arrival_profile=tuple(1. for _ in range(horizon)),
        grid_buy_per_kwh=tuple(.2 for _ in range(horizon)),
        grid_sell_per_kwh=tuple(.3 for _ in range(horizon)),
        fcs_retail_per_kwh=(.5,), fcs_ev_capacity_kwh=(15.,),
        fcs_fleet_exchange_capacity_kwh=(10.,),
        charge_efficiency=.8, discharge_efficiency=.8,
        mobile_service_efficiency=.8, fixed_service_efficiency=.8,
        price_noise_distribution='uniform', price_noise_scale=.1,
        charging_utility_intercept=3.,
        # Physics and legacy-pricing regressions: keep the fixed absolute menu
        # and the profile/shock station prices. See tests/test_station_pricing.py
        # for the LADWP base-price and premium-menu behaviour.
        fcs_pricing_mode="legacy", mcv_pricing_mode="absolute",
    )
    settings.update(changes)
    return replace(Scenario(), **settings).validate()


def small_config(**changes):
    settings = dict(seed=7, grid_step=5., samples_per_period=2,
                    price_index=4, training_seed_base=130000000)
    settings.update(changes)
    return fitted.FittedValueConfig(**settings)


def policy_fixture(scenario=None, config=None):
    scenario = scenario or tiny_scenario(zones=2)
    config = config or small_config()
    grid = np.array([5., 10., 15.])
    values = np.zeros((scenario.horizon + 1, scenario.n_zones, scenario.n_zones, 3, 3))
    for t in range(scenario.horizon):
        for first in range(scenario.n_zones):
            for second in range(scenario.n_zones):
                x, y = np.meshgrid(grid, grid, indexing='ij')
                values[t, first, second] = 1000*t + 100*first + 10*second + 2*x + 3*y + .1*x*y
    return fitted.FittedValuePolicy(scenario, config, grid, values)


def without_timing(result):
    result = deepcopy(result)
    result.pop('mean_policy_decision_ms', None)
    for episode in result['episodes']:
        episode.pop('mean_policy_decision_ms', None)
    return result


class FittedValueInterpolationTests(unittest.TestCase):
    def test_bilinear_interpolation_reproduces_affine_and_product_values(self):
        policy = policy_fixture()
        locations = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        energies = np.array([[7., 12.], [5., 15.], [15., 5.], [10., 10.]])
        expected = (1000 + 100*locations[:, 0] + 10*locations[:, 1]
                    + 2*energies[:, 0] + 3*energies[:, 1]
                    + .1*energies[:, 0]*energies[:, 1])
        np.testing.assert_allclose(policy.continuation(1, locations, energies), expected, atol=1e-12)
        self.assertAlmostEqual(float(policy.continuation(1, locations[0], energies[0])), expected[0])

    def test_location_and_vehicle_order_are_preserved(self):
        policy = policy_fixture()
        first = float(policy.continuation(0, (0, 1), (5., 15.)))
        swapped = float(policy.continuation(0, (1, 0), (15., 5.)))
        self.assertAlmostEqual(first, 10 + 2*5 + 3*15 + .1*5*15)
        self.assertAlmostEqual(swapped, 100 + 2*15 + 3*5 + .1*15*5)
        self.assertNotEqual(first, swapped)

    def test_terminal_continuation_is_zero_at_every_boundary_and_between_grid_points(self):
        policy = policy_fixture()
        for locations in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for energies in ((5., 5.), (15., 15.), (7., 12.)):
                self.assertEqual(float(policy.continuation(2, locations, energies)), 0.)

    def test_interpolation_rejects_invalid_time_locations_and_extrapolation(self):
        policy = policy_fixture()
        bad_queries = (
            (-1, (0, 1), (5., 10.)), (3, (0, 1), (5., 10.)),
            (.5, (0, 1), (5., 10.)), (True, (0, 1), (5., 10.)),
            (0, (0, 2), (5., 10.)), (0, (0., 1.), (5., 10.)),
            (0, (0,), (5., 10.)), (0, (0, 1), (4.9, 10.)),
            (0, (0, 1), (5., 15.1)), (0, (0, 1), (np.nan, 10.)),
        )
        for query in bad_queries:
            with self.subTest(query=query), self.assertRaises((ValueError, TypeError)):
                policy.continuation(*query)

    def test_constructor_rejects_wrong_grid_shape_nonfinite_and_nonzero_terminal_values(self):
        policy = policy_fixture()
        grid = np.array([5., 10., 15.])
        values = policy.values.copy()
        bad_inputs = [
            (np.array([5., 9., 15.]), values),
            (grid[::-1], values),
            (grid, values[:, :, :, :-1]),
            (grid, np.full_like(values, np.nan)),
        ]
        terminal = values.copy()
        terminal[-1, 0, 0, 0, 0] = 1.
        bad_inputs.append((grid, terminal))
        for bad_grid, bad_values in bad_inputs:
            with self.subTest(grid=bad_grid, shape=bad_values.shape), self.assertRaises(ValueError):
                fitted.FittedValuePolicy(policy.scenario, policy.config, bad_grid, bad_values)


class FittedValueSamplingAndPhysicsTests(unittest.TestCase):
    def test_synthetic_queries_replay_without_mutating_global_rng_or_scenario(self):
        scenario = tiny_scenario(zones=2)
        original = deepcopy(scenario)
        before = np.random.get_state()
        first = fitted.sample_dispatch_observation(scenario, 1, (0, 1), (10., 15.), 12345)
        second = fitted.sample_dispatch_observation(scenario, 1, (0, 1), (10., 15.), 12345)
        after = np.random.get_state()
        self.assertEqual(scenario, original)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(set(first), set(second))
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
        self.assertEqual(int(first['period'][0]), 1)
        self.assertEqual(int(first['stage']), 1)
        np.testing.assert_array_equal(first['locations'], [0, 1])
        np.testing.assert_array_equal(first['energy_kwh'], [10., 15.])

    def test_draws_reuse_arrivals_request_energies_and_station_shocks_across_fleet_states(self):
        scenario = tiny_scenario(zones=2, mean_requests_by_zone=(10., 10.))
        first = fitted.sample_dispatch_observation(scenario, 0, (0, 1), (15., 15.), 54321)
        second = fitted.sample_dispatch_observation(scenario, 0, (1, 0), (5., 10.), 54321)
        self.assertGreater(len(first['request_details']), 0)
        # Current choices can differ when travel times and mobile availability
        # change. The exogenous customers and station shocks must stay paired.
        np.testing.assert_array_equal(first['request_details'][:, :2], second['request_details'][:, :2])
        np.testing.assert_array_equal(first['request_details'][:, 3:], second['request_details'][:, 3:])
        for key in ('requests_by_zone', 'fcs_retail_prices', 'fcs_discharge_prices'):
            np.testing.assert_array_equal(first[key], second[key])

    def test_training_query_seed_blocks_are_deterministic_unique_and_independent_by_training_seed(self):
        scenario = tiny_scenario()
        trainer = fitted.FittedValueTrainer(scenario, small_config())
        seeds = trainer.query_seeds()
        self.assertEqual(np.asarray(seeds).shape, (2, 2))
        self.assertEqual(seeds, fitted.FittedValueTrainer(scenario, small_config()).query_seeds())
        flat = np.asarray(seeds).ravel().tolist()
        other = np.asarray(fitted.FittedValueTrainer(scenario, small_config(seed=19)).query_seeds()).ravel().tolist()
        self.assertEqual(len(set(flat)), 4)
        self.assertTrue(all(seed % 2 == 0 for seed in flat))
        self.assertTrue(set(flat).isdisjoint(other))

    def test_all_joint_predictions_match_actual_dispatch_with_shared_exchange_and_ordered_packing(self):
        scenario = tiny_scenario(horizon=1, zones=2, terminal_value_per_stored_kwh=.17)
        simulator = ChargingSimulator(scenario)
        simulator.reset(seed=191, options={'requests': [3, 0], 'request_energy_kwh': [5., 8., 7.]})
        simulator.quote_prices([.6, .6])
        simulator.request_choices = np.array([0, 0, 0], dtype=np.int64)
        simulator._refresh_demand()
        obs = simulator.observe()
        before = deepcopy(obs)
        rng_state = deepcopy(simulator.rng.bit_generator.state)
        modes, targets, amounts, outcomes = joint_dispatch_outcomes(obs, scenario)
        self.assertGreater(len(modes), 1)
        for index in range(len(modes)):
            trial = deepcopy(simulator)
            actions = [DispatchAction(Mode(int(modes[index, m])), int(targets[index, m]), float(amounts[index, m]))
                       for m in range(scenario.n_mcv)]
            next_obs, reward, _, _, info = trial.dispatch(actions)
            with self.subTest(index=index):
                self.assertAlmostEqual(outcomes['reward'][index], reward, places=10)
                np.testing.assert_allclose(outcomes['next_energy_kwh'][index], info['energy_ledger']['after_kwh'], atol=1e-10)
                np.testing.assert_array_equal(outcomes['next_locations'][index], next_obs['locations'])
                np.testing.assert_array_equal(outcomes['served_mask'][index], info['served_request_mask'])
                self.assertFalse(info['invalid_actions'])
        self.assertEqual(simulator.rng.bit_generator.state, rng_state)
        for key in obs:
            np.testing.assert_array_equal(obs[key], before[key])

    def test_online_dispatch_uses_public_current_state_and_saved_values_without_sampling(self):
        policy = policy_fixture()
        obs = fitted.sample_dispatch_observation(policy.scenario, 0, (0, 1), (10., 15.), 987)
        before = deepcopy(obs)
        with patch.object(fitted, 'ChargingSimulator', side_effect=AssertionError('Online policy sampled a simulator')):
            first = policy.dispatch(obs)
            second = policy.dispatch(obs, np.random.default_rng(999))
        self.assertEqual(first, second)
        for m, action in enumerate(first):
            self.assertTrue(obs['action_mask'][m, action.mode, action.target_zone])
        for key in before:
            np.testing.assert_array_equal(obs[key], before[key])


class FittedValueTrainingAndCheckpointTests(unittest.TestCase):
    def test_one_period_bellman_values_match_exhaustive_actual_simulator_rewards(self):
        scenario = tiny_scenario(horizon=1, mean_requests_by_zone=(0.,),
                                 price_noise_distribution='none', terminal_value_per_stored_kwh=.17)
        config = small_config(grid_step=10.)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'one_period'
            fitted.FittedValueTrainer(scenario, config).train(directory)
            policy = fitted.FittedValuePolicy.load_checkpoint(directory / 'checkpoint.npz')
        np.testing.assert_array_equal(policy.values[-1], 0.)
        for first_index, first_energy in enumerate((5., 15.)):
            for second_index, second_energy in enumerate((5., 15.)):
                query_scenario = replace(scenario, initial_energy_kwh=(first_energy, second_energy)).validate()
                simulator = ChargingSimulator(query_scenario)
                obs, _ = simulator.reset(seed=887)
                obs, *_ = simulator.quote_prices([.6])
                modes, targets, amounts, _ = joint_dispatch_outcomes(obs, scenario)
                actual_rewards = []
                for index in range(len(modes)):
                    actions = [DispatchAction(Mode(int(modes[index, m])), int(targets[index, m]), float(amounts[index, m]))
                               for m in range(2)]
                    trial = deepcopy(simulator)
                    _, reward, terminated, _, _ = trial.dispatch(actions)
                    self.assertTrue(terminated)
                    actual_rewards.append(reward)
                with self.subTest(energies=(first_energy, second_energy)):
                    self.assertAlmostEqual(policy.values[0, 0, 0, first_index, second_index], max(actual_rewards), places=10)

    def test_stochastic_terminal_backup_averages_observed_action_optima(self):
        scenario = tiny_scenario(horizon=1, mean_requests_by_zone=(5.,),
                                 terminal_value_per_stored_kwh=.17)
        config = small_config(grid_step=10., samples_per_period=3)
        trainer = fitted.FittedValueTrainer(scenario, config)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer.train(Path(temporary) / 'stochastic')
        for i, e0 in enumerate((5., 15.)):
            for j, e1 in enumerate((5., 15.)):
                observed_best = []
                for seed in trainer.query_seeds()[0]:
                    physical = replace(scenario, initial_energy_kwh=(e0, e1)).validate()
                    simulator = ChargingSimulator(physical)
                    obs, _ = simulator.reset(seed=seed)
                    levels = np.where(obs['mobile_available'], config.price_index, 0).astype(int)
                    obs, *_ = simulator.quote_price_levels(levels)
                    modes, targets, amounts, _ = joint_dispatch_outcomes(obs, scenario)
                    trial_rewards = []
                    for k in range(len(modes)):
                        actions = [DispatchAction(Mode(int(modes[k, m])), int(targets[k, m]), float(amounts[k, m]))
                                   for m in range(2)]
                        trial = deepcopy(simulator)
                        _, reward, *_ = trial.dispatch(actions)
                        trial_rewards.append(reward)
                    observed_best.append(max(trial_rewards))
                self.assertAlmostEqual(trainer.policy.values[0, 0, 0, i, j], float(np.mean(observed_best)), places=10)

    def test_two_period_backup_applies_period_discount_to_next_value_once(self):
        scenario = tiny_scenario(mean_requests_by_zone=(0.,), price_noise_distribution='none',
                                 grid_buy_per_kwh=(.1, .8), grid_sell_per_kwh=(.05, .7),
                                 discount_per_period=.4, terminal_value_per_stored_kwh=.09)
        config = small_config(grid_step=10., samples_per_period=1)
        trainer = fitted.FittedValueTrainer(scenario, config)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer.train(Path(temporary) / 'discount')
        terminal_scenario = replace(scenario, horizon=1, arrival_profile=(1.,),
                                    grid_buy_per_kwh=(.8,), grid_sell_per_kwh=(.7,)).validate()
        next_values = np.zeros((2, 2))
        for i, e0 in enumerate((5., 15.)):
            for j, e1 in enumerate((5., 15.)):
                simulator = ChargingSimulator(replace(terminal_scenario, initial_energy_kwh=(e0, e1)))
                simulator.reset(seed=1)
                obs, *_ = simulator.quote_prices([.6])
                modes, targets, amounts, _ = joint_dispatch_outcomes(obs, terminal_scenario)
                rewards = []
                for k in range(len(modes)):
                    actions = [DispatchAction(Mode(int(modes[k, m])), int(targets[k, m]), float(amounts[k, m]))
                               for m in range(2)]
                    trial = deepcopy(simulator)
                    _, reward, *_ = trial.dispatch(actions)
                    rewards.append(reward)
                next_values[i, j] = max(rewards)
        np.testing.assert_allclose(trainer.policy.values[1, 0, 0], next_values, atol=1e-10)
        for i, e0 in enumerate((5., 15.)):
            for j, e1 in enumerate((5., 15.)):
                simulator = ChargingSimulator(replace(scenario, initial_energy_kwh=(e0, e1)))
                simulator.reset(seed=2)
                obs, *_ = simulator.quote_prices([.6])
                modes, targets, amounts, _ = joint_dispatch_outcomes(obs, scenario)
                scores = []
                for k in range(len(modes)):
                    actions = [DispatchAction(Mode(int(modes[k, m])), int(targets[k, m]), float(amounts[k, m]))
                               for m in range(2)]
                    trial = deepcopy(simulator)
                    next_obs, reward, *_ = trial.dispatch(actions)
                    u, v = (next_obs['energy_kwh'] - 5.) / 10.
                    future = ((1-u)*(1-v)*next_values[0, 0] + u*(1-v)*next_values[1, 0]
                              + (1-u)*v*next_values[0, 1] + u*v*next_values[1, 1])
                    scores.append(reward + .4*future)
                self.assertAlmostEqual(trainer.policy.values[0, 0, 0, i, j], max(scores), places=10)

    def test_training_cli_repeats_without_overwriting_existing_results(self):
        from charging_env.rl.fitted_train import main
        scenario = tiny_scenario(horizon=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_file = root / 'scenario.json'
            scenario.to_json(scenario_file)
            requested = root / 'run'
            arguments = ['--scenario', str(scenario_file), '--output-dir', str(requested),
                         '--grid-step', '10', '--samples', '1']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            original = {path.name: path.read_bytes() for path in requested.iterdir()}
            occupied = root / 'run_2'
            occupied.write_text('Preserve this unrelated file.')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertEqual({path.name: path.read_bytes() for path in requested.iterdir()}, original)
            self.assertEqual(occupied.read_text(), 'Preserve this unrelated file.')
            self.assertTrue((root / 'run_3/checkpoint.npz').is_file())

    def test_two_period_training_checkpoint_and_evaluation_replay_exactly(self):
        scenario = tiny_scenario()
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'training'
            summary = fitted.FittedValueTrainer(scenario, config).train(directory)
            first = fitted.FittedValuePolicy.load_checkpoint(directory / 'checkpoint.npz')
            copied = Path(temporary) / 'copy.npz'
            first.save_checkpoint(copied)
            original_bytes = copied.read_bytes()
            with self.assertRaises(FileExistsError):
                first.save_checkpoint(copied)
            self.assertEqual(copied.read_bytes(), original_bytes)
            second = fitted.FittedValuePolicy.load_checkpoint(copied)
            self.assertTrue(isinstance(summary, dict))
            self.assertEqual(first.scenario, scenario)
            self.assertEqual(first.config, config)
            np.testing.assert_array_equal(first.energy_grid, second.energy_grid)
            np.testing.assert_array_equal(first.values, second.values)
            result = evaluate_policy(scenario, [601, 603, 605], first.prices, first.dispatch)
            replay = evaluate_policy(scenario, [601, 603, 605], second.prices, second.dispatch)
            self.assertEqual(without_timing(result), without_timing(replay))
            self.assertEqual(result['invalid_actions'], 0)
            self.assertTrue(np.isfinite(result['mean_discounted_profit']))

    def test_checkpoint_rejects_corrupt_values_and_object_payload_without_pickle(self):
        policy = policy_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'policy.npz'
            policy.save_checkpoint(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as archive:
                payload = {name: archive[name].copy() for name in archive.files}
            self.assertIn('values', payload)
            for replacement in (np.full_like(payload['values'], np.nan), payload['values'][..., :-1]):
                changed = dict(payload, values=replacement)
                np.savez(checkpoint, **changed)
                with self.assertRaises(ValueError):
                    fitted.FittedValuePolicy.load_checkpoint(checkpoint)
            np.savez(checkpoint, values=np.array([{'unsafe': 'object'}], dtype=object))
            with self.assertRaises((ValueError, KeyError)):
                fitted.FittedValuePolicy.load_checkpoint(checkpoint)

    def test_scenario_mismatch_requires_explicit_distribution_shift_and_rejects_physics_changes(self):
        policy = policy_fixture()
        shifted = replace(policy.scenario, mean_requests_by_zone=(3., 3.)).validate()
        with self.assertRaises(ValueError):
            policy.with_scenario(shifted)
        compatible = policy.with_scenario(shifted, allow_distribution_shift=True)
        np.testing.assert_array_equal(compatible.values, policy.values)
        incompatible = replace(policy.scenario, battery_capacity_kwh=16.).validate()
        with self.assertRaises(ValueError):
            policy.with_scenario(incompatible, allow_distribution_shift=True)

    def test_configs_reject_wrong_fleet_invalid_settings_and_unbounded_query_budgets(self):
        scenario = tiny_scenario()
        invalid_configs = (
            small_config(seed=True), small_config(seed=-1), small_config(grid_step=0.),
            small_config(grid_step=np.nan), small_config(samples_per_period=0),
            small_config(samples_per_period=True), small_config(price_index=99),
            small_config(energy_fractions=()), small_config(energy_fractions=(1.1,)),
            small_config(max_joint_actions=0), small_config(training_seed_base=-1),
            small_config(grid_step=1e-6), small_config(samples_per_period=10**9),
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate(scenario)
        one_vehicle = replace(scenario, initial_locations=(0,), initial_energy_kwh=(10.,)).validate()
        with self.assertRaisesRegex(ValueError, '(two|2)'):
            small_config().validate(one_vehicle)

    def test_joint_enumeration_cap_rejects_before_silent_action_pruning(self):
        scenario = tiny_scenario()
        obs = fitted.sample_dispatch_observation(scenario, 0, (0, 0), (10., 10.), 2468)
        with self.assertRaisesRegex(ValueError, 'max_joint_actions'):
            joint_dispatch_outcomes(obs, scenario, max_joint_actions=1)


if __name__ == '__main__':
    unittest.main()
