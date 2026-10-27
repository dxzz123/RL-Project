"""Larger-fleet ordinary imitation and exact two-vehicle compatibility."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, Mode

try:
    import torch
except ImportError:
    torch = None


def small_scenario(vehicles=3, **changes):
    values = dict(horizon=3, zone_xy_km=((0., 0.), (2., 0.)), fcs_zones=(0,),
        initial_locations=(0,) + (1,) * (vehicles - 1), initial_energy_kwh=(35.,) * vehicles,
        mean_requests_by_zone=(3., 3.), arrival_profile=(1., 1., 1.),
        grid_buy_per_kwh=(.12, .18, .14), grid_sell_per_kwh=(.09, .14, .1),
        fcs_retail_per_kwh=(.45,), fcs_ev_capacity_kwh=(30.,),
        fcs_fleet_exchange_capacity_kwh=(30.,), price_noise_distribution='none',
        charging_utility_intercept=3., price_sensitivity_per_dollar=.1)
    values.update(changes)
    return replace(Scenario(), **values).validate()


@unittest.skipIf(torch is None, 'Install the train extra for scalable initialization')
class ScalableWarmstartTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl import paper_ppo, paper_warmstart
        from charging_env.benchmarks import scalable
        self.ppo, self.warm, self.scalable = paper_ppo, paper_warmstart, scalable
        self.scenario = small_scenario()
        self.config = paper_ppo.PaperPPOConfig(seed=31, updates=1, episodes_per_update=2,
            epochs=1, minibatch_size=6, hidden_size=8, eval_episodes=1, eval_interval=0,
            gae_lambda=1., reward_scale=20., normalize_advantages=True,
            warmstart_episodes=2, warmstart_epochs=2, warmstart_minibatch_size=6,
            warmstart_dispatch_solver='coordinate', warmstart_coordinate_sweeps=2)

    def trainer(self, scenario=None, **changes):
        return self.ppo.PaperPPOTrainer(scenario or self.scenario, replace(self.config, **changes))

    def assert_parameters_equal(self, first, second):
        for name, model in first._modules().items():
            for a, b in zip(model.parameters(), second._modules()[name].parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_larger_fleet_requires_explicit_coordinate_solver(self):
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            self.trainer(warmstart_dispatch_solver='exhaustive')
        self.trainer()
        with self.assertRaisesRegex(ValueError, 'solver'):
            self.warm.validate_teacher_scenario(self.scenario, 'unknown')
        no_price = replace(self.scenario, price_levels=(.2, .4, .8, 1.))
        with self.assertRaisesRegex(ValueError, '0.60'):
            self.trainer(no_price)

    def test_native_mask_matches_paper_operations_and_preserves_stranded_wait(self):
        scenario = small_scenario(initial_locations=(1, 0, 1), initial_energy_kwh=(5., 10., 35.))
        sim = ChargingSimulator(scenario); obs, _ = sim.reset(seed=921, options={'requests': [0, 0]})
        post, *_ = sim.quote_price_levels(self.warm.inventory_teacher_prices(obs, scenario, solver='coordinate'))
        original = self.scalable.coordinate_inventory_dispatch
        captured = []
        def inspect(obs, config, shadow_price, **kwargs):
            captured.append(kwargs['action_mask'].copy())
            return original(obs, config, shadow_price, **kwargs)
        with patch.object(self.scalable, 'coordinate_inventory_dispatch', side_effect=inspect):
            actions = self.warm.inventory_teacher_actions(post, scenario, solver='coordinate')
        native = captured[0]; paper = self.ppo.paper_dispatch_mask(post, scenario)
        np.testing.assert_array_equal(native[:, Mode.SERVE], paper[:, self.ppo.PaperOperation.SERVICE])
        np.testing.assert_array_equal(native[:, Mode.RECHARGE], paper[:, self.ppo.PaperOperation.CHARGE])
        np.testing.assert_array_equal(native[:, Mode.DISCHARGE], paper[:, self.ppo.PaperOperation.DISCHARGE])
        self.assertTrue((~native | np.asarray(post['action_mask'], dtype=bool)).all())
        for m, location in enumerate(post['locations']):
            self.assertTrue(native[m, Mode.WAIT, location])
            self.assertFalse(native[m, Mode.REPOSITION, location])
        commands = self.ppo.paper_dispatch_commands(post, actions, scenario)
        self.assertEqual(commands[0].mode, Mode.WAIT)
        self.assertEqual(commands[0].target_zone, 1)
        self.assertFalse(sim.dispatch(commands)[-1]['invalid_actions'])

    def test_coordinate_teacher_does_not_exhaustively_enumerate_and_reports_actual_work(self):
        trainer = self.trainer()
        with patch.object(self.warm, 'joint_dispatch_outcomes', side_effect=AssertionError('No exhaustive fleet search')):
            report = trainer.warm_start()
        periods = self.config.warmstart_episodes * self.scenario.horizon
        self.assertEqual(report['dispatch_solver'], 'coordinate')
        self.assertEqual(report['coordinate_sweeps'], 2)
        self.assertFalse(report['solver_exact'])
        self.assertEqual(report['teacher'], self.warm.COORDINATE_TEACHER)
        self.assertEqual(report['teacher_pricing_queries'], periods)
        self.assertEqual(report['teacher_dispatch_queries'], periods)
        self.assertEqual(report['teacher_solver_statistics']['calls'], periods)
        self.assertGreater(report['teacher_evaluated_joint_candidates'], 0)
        self.assertEqual(report['teacher_evaluated_joint_candidates'],
                         report['teacher_solver_statistics']['candidate_outcomes'])
        self.assertIsNone(report['teacher_joint_candidate_outcomes'])
        self.assertIsNone(report['teacher_admissible_joint_candidates'])
        self.assertFalse(report['extra_training_reward'])
        self.assertEqual(trainer.update_count, 0)
        self.assertEqual(trainer.episode_count, 0)

    def test_demonstration_returns_use_unmodified_simulator_profit_and_rng(self):
        trainer = self.trainer()
        torch_before = torch.get_rng_state().clone(); numpy_before = np.random.get_state()
        report = trainer.warm_start()
        torch.testing.assert_close(torch.get_rng_state(), torch_before, atol=0, rtol=0)
        numpy_after = np.random.get_state()
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        profits = []
        for seed in report['seeds']:
            sim = ChargingSimulator(self.scenario); obs, _ = sim.reset(seed=seed)
            rewards = []
            while sim.stage != 2:
                levels = self.warm.inventory_teacher_prices(obs, self.scenario, solver='coordinate')
                post, *_ = sim.quote_price_levels(levels)
                action = self.warm.inventory_teacher_actions(post, self.scenario,
                    solver='coordinate', sweeps=self.config.warmstart_coordinate_sweeps)
                obs, reward, _, _, info = sim.dispatch(self.ppo.paper_dispatch_commands(post, action, self.scenario))
                self.assertFalse(info['invalid_actions'])
                rewards.append(reward)
            profits.append(float(np.dot(self.scenario.discount_per_period**np.arange(len(rewards)), rewards)))
        self.assertAlmostEqual(report['mean_demonstration_discounted_profit'], float(np.mean(profits)))
        self.assertEqual(report['critic_targets'], 'complete discounted raw returns divided by reward_scale')

    def test_default_exact_fit_is_unaffected_by_coordinate_settings(self):
        scenario = small_scenario(vehicles=2)
        first = self.trainer(scenario, warmstart_dispatch_solver='exhaustive', warmstart_coordinate_sweeps=1)
        second = self.trainer(scenario, warmstart_dispatch_solver='exhaustive', warmstart_coordinate_sweeps=8)
        report = first.warm_start(); changed = second.warm_start()
        self.assert_parameters_equal(first, second)
        self.assertEqual(report['teacher'], self.warm.TEACHER)
        self.assertTrue(report['solver_exact'])
        self.assertIsNone(report['coordinate_sweeps'])
        self.assertEqual(report['demonstration_sha256'], changed['demonstration_sha256'])
        self.assertEqual(report['final_metrics'], changed['final_metrics'])
        self.assertEqual(report['teacher_evaluated_joint_candidates'], report['teacher_joint_candidate_outcomes'])

    def test_coordinate_checkpoint_continues_with_identical_rollout_and_update(self):
        trainer = self.trainer(); trainer.warm_start(); trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coordinate.pt'; trainer.save_checkpoint(path)
            restored = self.ppo.PaperPPOTrainer.load_checkpoint(path)
        self.assertEqual(restored.warmstart_metadata, trainer.warmstart_metadata)
        self.assertEqual(restored.config.warmstart_dispatch_solver, 'coordinate')
        self.assertEqual(restored.config.warmstart_coordinate_sweeps, 2)
        expected, actual = trainer.collect_rollout(), restored.collect_rollout()
        for field in ('pricing_actions', 'dispatch_actions', 'rewards', 'pricing_rewards',
                      'old_price_log_probs', 'old_dispatch_log_probs', 'price_values', 'dispatch_values'):
            np.testing.assert_array_equal(getattr(expected, field), getattr(actual, field))
        trainer.update(expected); restored.update(actual)
        self.assert_parameters_equal(trainer, restored)

    def test_checkpoint_rejects_mismatched_solver_provenance(self):
        trainer = self.trainer(); trainer.warm_start()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coordinate.pt'; trainer.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            payload['warmstart_metadata']['summary']['coordinate_sweeps'] = 3
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                self.ppo.PaperPPOTrainer.load_checkpoint(path)

    def test_coordinate_prices_still_hide_current_realized_requests(self):
        sim = ChargingSimulator(self.scenario); obs, _ = sim.reset(seed=923)
        changed = deepcopy(obs)
        for key in ('request_details', 'potential_energy_kwh', 'mobile_wait_hours', 'fcs_wait_hours', 'pricing_mask'):
            changed[key] = np.full_like(changed[key], 999.)
        np.testing.assert_array_equal(
            self.warm.inventory_teacher_prices(obs, self.scenario, solver='coordinate'),
            self.warm.inventory_teacher_prices(changed, self.scenario, solver='coordinate'))


if __name__ == '__main__':
    unittest.main()
