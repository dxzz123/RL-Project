"""Checks for inventory initialization, training units, and checkpoint continuation."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from itertools import product
import io
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


def small_scenario(**changes):
    values = dict(
        # Dispatch/rollout/history regressions: keep the fixed absolute menu and
        # profile station prices. tests/test_station_pricing.py covers the LADWP
        # base-price bands and the premium menu.
        fcs_pricing_mode='legacy', mcv_pricing_mode='absolute',
        horizon=3, zone_xy_km=((0., 0.), (2., 0.)), fcs_zones=(0,),
        initial_locations=(0, 1), initial_energy_kwh=(35., 40.),
        mean_requests_by_zone=(3., 3.), arrival_profile=(1., 1., 1.),
        grid_buy_per_kwh=(.12, .18, .14), grid_sell_per_kwh=(.09, .14, .1),
        fcs_retail_per_kwh=(.45,), fcs_ev_capacity_kwh=(30.,),
        fcs_fleet_exchange_capacity_kwh=(30.,), price_noise_distribution='none',
        charging_utility_intercept=3., price_sensitivity_per_dollar=.1,
    )
    values.update(changes)
    return replace(Scenario(), **values).validate()


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class InventoryInitializationTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl import paper_ppo, paper_warmstart
        self.ppo = paper_ppo
        self.warm = paper_warmstart
        self.scenario = small_scenario()
        self.config = paper_ppo.PaperPPOConfig(
            seed=29, updates=1, episodes_per_update=2, epochs=1,
            minibatch_size=6, hidden_size=8, eval_episodes=1,
            eval_interval=0, gae_lambda=1., reward_scale=20.,
            normalize_advantages=True, warmstart_episodes=2,
            warmstart_epochs=8, warmstart_minibatch_size=6,
            warmstart_learning_rate=.003, torch_threads=1,
        )

    def trainer(self, **changes):
        return self.ppo.PaperPPOTrainer(self.scenario, replace(self.config, **changes))

    def parameters(self, trainer):
        return {name: [p.detach().clone() for p in model.parameters()]
                for name, model in trainer._modules().items()}

    def assert_parameters_equal(self, first, second):
        for name in first:
            for a, b in zip(first[name], second[name]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def teacher_demonstrations(self, seeds):
        pre, post, prices, actions = [], [], [], []
        for seed in seeds:
            sim = ChargingSimulator(self.scenario)
            obs, _ = sim.reset(seed=int(seed))
            while sim.stage != 2:
                before = self.ppo.paper_pricing_observation(obs, self.scenario)
                levels = self.warm.inventory_teacher_prices(before, self.scenario)
                after, *_ = sim.quote_price_levels(levels)
                flat = self.warm.inventory_teacher_actions(after, self.scenario)
                pre.append(before); post.append(after)
                prices.append(levels); actions.append(flat)
                obs, *_ = sim.dispatch(self.ppo.paper_dispatch_commands(after, flat, self.scenario))
        return pre, post, np.asarray(prices), np.asarray(actions)

    def test_teacher_prices_ignore_hidden_requests_and_use_admissible_price_levels(self):
        sim = ChargingSimulator(self.scenario)
        obs, _ = sim.reset(seed=901)
        expected = self.warm.inventory_teacher_prices(obs, self.scenario)
        changed = deepcopy(obs)
        for key in ('requests_by_zone', 'potential_energy_kwh', 'request_details',
                    'mobile_prices', 'mobile_demand', 'fixed_demand',
                    'mobile_requested_kwh', 'fixed_requested_kwh', 'outside_requests',
                    'mobile_wait_hours', 'fcs_wait_hours', 'pricing_mask', 'action_mask'):
            changed[key] = np.full(np.shape(changed[key]), 999.)
        np.testing.assert_array_equal(self.warm.inventory_teacher_prices(changed, self.scenario), expected)
        mask = self.ppo.paper_pricing_observation(obs, self.scenario)['pricing_mask']
        self.assertTrue(mask[np.arange(self.scenario.n_zones), expected].all())
        np.testing.assert_array_equal(np.asarray(self.scenario.price_levels)[expected], [.6, .6])

    def test_teacher_charge_labels_share_the_actor_mask_and_exchange_convention(self):
        scenario = small_scenario(initial_locations=(0, 0), initial_energy_kwh=(10., 10.))
        sim = ChargingSimulator(scenario)
        obs, _ = sim.reset(seed=903, options={'requests': [0, 0]})
        post, *_ = sim.quote_price_levels(self.warm.inventory_teacher_prices(obs, scenario))
        flat = self.warm.inventory_teacher_actions(post, scenario)
        mask = self.ppo.paper_dispatch_mask(post, scenario).reshape(scenario.n_mcv, -1)
        self.assertTrue(mask[np.arange(scenario.n_mcv), flat].all())
        commands = self.ppo.paper_dispatch_commands(post, flat, scenario)
        self.assertTrue(any(command.mode == Mode.RECHARGE for command in commands))
        self.assertTrue(all(command.energy_kwh == 30. for command in commands
                            if command.mode == Mode.RECHARGE))
        selected = deepcopy(sim).dispatch(commands)[-1]
        self.assertFalse(selected['invalid_actions'])
        self.assertAlmostEqual(float(sum(selected['energy_ledger']['charge_grid_kwh'])), 30.)
        # Either one charger or two sharing capacity can tie for the teacher.
        # Explicitly exercise the shared allocation for the actor's two labels.
        both = np.full(scenario.n_mcv, int(self.ppo.PaperOperation.CHARGE)*scenario.n_zones)
        simultaneous = self.ppo.paper_dispatch_commands(post, both, scenario)
        self.assertTrue(all(command.energy_kwh == 30. for command in simultaneous))
        _, _, _, _, info = sim.dispatch(simultaneous)
        self.assertFalse(info['invalid_actions'])
        self.assertAlmostEqual(float(sum(info['energy_ledger']['charge_grid_kwh'])), 30.)
        np.testing.assert_allclose(info['energy_ledger']['after_kwh'], [24.25, 24.25])

    def test_terminal_teacher_maximizes_actual_reward_without_extra_inventory_value(self):
        scenario = small_scenario(initial_locations=(0, 0), initial_energy_kwh=(15., 18.))
        sim = ChargingSimulator(scenario)
        sim.reset(seed=905, options={'requests': [0, 0]})
        sim.period = scenario.horizon - 1
        sim._prepare_period({'requests': np.zeros(scenario.n_zones, dtype=int)})
        post, *_ = sim.quote_price_levels(self.warm.inventory_teacher_prices(sim.observe(), scenario))
        mask = self.ppo.paper_dispatch_mask(post, scenario).reshape(scenario.n_mcv, -1)
        best = -np.inf
        for joint in product(*(np.flatnonzero(row) for row in mask)):
            trial = deepcopy(sim)
            commands = self.ppo.paper_dispatch_commands(post, np.asarray(joint), scenario)
            _, reward, done, _, info = trial.dispatch(commands)
            self.assertTrue(done)
            self.assertFalse(info['invalid_actions'])
            best = max(best, reward)
        selected = self.warm.inventory_teacher_actions(post, scenario)
        commands = self.ppo.paper_dispatch_commands(post, selected, scenario)
        self.assertFalse(any(command.mode == Mode.RECHARGE for command in commands))
        _, actual, _, _, _ = sim.dispatch(commands)
        self.assertAlmostEqual(actual, best)

    def test_teacher_preserves_idle_fallback_when_vehicles_are_stranded(self):
        scenario = small_scenario(initial_locations=(1, 1), initial_energy_kwh=(5., 5.))
        sim = ChargingSimulator(scenario)
        obs, _ = sim.reset(seed=907, options={'requests': [0, 0]})
        post, *_ = sim.quote_price_levels(self.warm.inventory_teacher_prices(obs, scenario))
        actions = self.warm.inventory_teacher_actions(post, scenario)
        commands = self.ppo.paper_dispatch_commands(post, actions, scenario)
        self.assertTrue(all(a.mode == Mode.WAIT and a.target_zone == 1 for a in commands))
        self.assertFalse(sim.dispatch(commands)[-1]['invalid_actions'])

    def test_reward_scaling_keeps_economic_rewards_and_reports_in_dollars(self):
        original = self.trainer(reward_scale=1., warmstart_episodes=0)
        scaled = self.trainer(reward_scale=20., warmstart_episodes=0)
        first, second = original.collect_rollout(), scaled.collect_rollout()
        for field in ('pricing_actions', 'dispatch_actions', 'rewards', 'pricing_rewards'):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        self.assertEqual(first.episodes, second.episodes)
        offset = 0
        for episode in second.episodes:
            end = offset + self.scenario.horizon
            raw = second.rewards[offset:end]
            self.assertAlmostEqual(episode['profit'], float(raw.sum()))
            self.assertAlmostEqual(episode['discounted_profit'],
                                   float(np.dot(self.scenario.discount_per_period**np.arange(len(raw)), raw)))
            offset = end

    def test_lambda_one_training_targets_are_discounted_returns_in_scaled_units(self):
        trainer = self.trainer(warmstart_episodes=0)
        rollout = trainer.collect_rollout()
        original_rewards = rollout.rewards.copy()
        expected = np.zeros(len(rollout))
        continuation = 0.
        for i in range(len(rollout)-1, -1, -1):
            continuation = rollout.rewards[i]/trainer.config.reward_scale + (
                0. if rollout.dones[i] else self.scenario.discount_per_period*continuation)
            expected[i] = continuation
        captured = []
        function = self.ppo.critic_targets
        def capture(*args, **kwargs):
            result = function(*args, **kwargs)
            captured.append(deepcopy(result))
            return result
        with patch.object(self.ppo, 'critic_targets', side_effect=capture):
            metrics = trainer.update(rollout)
        self.assertEqual(len(captured), 1)
        np.testing.assert_allclose(captured[0]['pricing_returns'], expected, atol=1e-5)
        np.testing.assert_allclose(captured[0]['dispatch_returns'], expected, atol=1e-5)
        np.testing.assert_array_equal(rollout.rewards, original_rewards)
        self.assertAlmostEqual(metrics['mean_training_profit'],
                               float(np.mean([e['profit'] for e in rollout.episodes])))

    def test_supervised_initialization_improves_demonstration_likelihood_without_ppo_updates(self):
        trainer = self.trainer()
        original = deepcopy(trainer)
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        with patch.object(trainer, 'update', side_effect=AssertionError('Demonstrations must not enter PPO')):
            report = trainer.warm_start()
        self.assertTrue(trainer.warmstart_metadata['completed'])
        self.assertEqual(trainer.warmstart_metadata['summary'], report)
        self.assertEqual(trainer.update_count, 0)
        self.assertEqual(trainer.episode_count, 0)
        self.assertEqual(trainer.environment_periods, 0)
        data = self.teacher_demonstrations(report['seeds'])
        with torch.no_grad():
            def loss(model):
                return -(model.pricing_actor.evaluate_actions(data[0], data[2])['log_prob'].mean()
                         + model.dispatch_actor.evaluate_actions(data[1], data[3])['log_prob'].mean())
            self.assertLess(float(loss(trainer)), float(loss(original)))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        torch.testing.assert_close(torch_before, torch.get_rng_state(), atol=0, rtol=0)
        before = self.parameters(trainer)
        self.assertEqual(trainer.warm_start(), report)
        self.assert_parameters_equal(before, self.parameters(trainer))

    def test_initialization_and_scaled_training_resume_with_identical_next_rollout_and_update(self):
        trainer = self.trainer(warmstart_epochs=2)
        trainer.warm_start()
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            trainer.save_checkpoint(path)
            restored = self.ppo.PaperPPOTrainer.load_checkpoint(path)
        self.assertEqual(restored.warmstart_metadata, trainer.warmstart_metadata)
        before = self.parameters(restored)
        restored.warm_start()
        self.assert_parameters_equal(before, self.parameters(restored))
        expected, actual = trainer.collect_rollout(), restored.collect_rollout()
        for field in ('pricing_actions', 'dispatch_actions', 'rewards', 'pricing_rewards',
                      'old_price_log_probs', 'old_dispatch_log_probs', 'price_values', 'dispatch_values'):
            np.testing.assert_array_equal(getattr(expected, field), getattr(actual, field))
        trainer.update(expected); restored.update(actual)
        self.assert_parameters_equal(self.parameters(trainer), self.parameters(restored))
        self.assertEqual(trainer.update_count, restored.update_count)
        self.assertEqual(trainer.episode_count, restored.episode_count)

    def test_legacy_checkpoint_loads_with_original_units_and_no_implicit_initialization(self):
        trainer = self.trainer(reward_scale=1., warmstart_episodes=0)
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['format_version'] = 1
            payload.pop('warmstart_metadata')
            for name in list(payload['training_config']):
                if name == 'reward_scale' or name.startswith('warmstart_'):
                    payload['training_config'].pop(name)
            torch.save(payload, path)
            restored = self.ppo.PaperPPOTrainer.load_checkpoint(path)
        self.assertEqual(restored.config.reward_scale, 1.)
        self.assertEqual(restored.config.warmstart_episodes, 0)
        self.assert_parameters_equal(self.parameters(trainer), self.parameters(restored))
        first, second = trainer.collect_rollout(), restored.collect_rollout()
        np.testing.assert_array_equal(first.dispatch_actions, second.dispatch_actions)
        np.testing.assert_array_equal(first.rewards, second.rewards)

    def test_evaluation_does_not_trigger_initialization_and_cli_rejects_changed_units_or_teacher(self):
        from charging_env.rl.paper_train import main
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            trainer.save_checkpoint(path)
            with patch.object(self.ppo.PaperPPOTrainer, 'warm_start', side_effect=AssertionError('Evaluation must not train')):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['--evaluate-only', str(path), '--eval-episodes', '1']), 0)
            for args in (['--reward-scale', '1'], ['--warmstart-episodes', '5'],
                         ['--warmstart-epochs', '9'], ['--warmstart-seed-base', '412000002']):
                with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['--resume', str(path)] + args)

    def test_pending_initialization_checkpoint_cannot_silently_resume_without_demonstrations(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pending.pt'
            trainer.save_checkpoint(path)
            restored = self.ppo.PaperPPOTrainer.load_checkpoint(path)
            output = Path(directory)/'resume'
            with self.assertRaises(ValueError):
                restored.train(output)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                restored.warm_start()

    def test_teacher_rejects_scenarios_outside_its_declared_prior(self):
        no_price = small_scenario(price_levels=(.2, .4, .8, 1.))
        one_vehicle = small_scenario(initial_locations=(0,), initial_energy_kwh=(35.,))
        for scenario in (no_price, one_vehicle):
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                self.ppo.PaperPPOTrainer(scenario, self.config)

    def test_invalid_scale_and_initialization_settings_are_rejected(self):
        for changes in ({'reward_scale': 0.}, {'reward_scale': -1.}, {'reward_scale': True},
                        {'reward_scale': float('nan')}, {'warmstart_episodes': -1},
                        {'warmstart_episodes': True}, {'warmstart_epochs': 0},
                        {'warmstart_learning_rate': 0.}, {'warmstart_minibatch_size': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes).validate()


if __name__ == '__main__':
    unittest.main()
