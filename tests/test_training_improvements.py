"""Training-only improvements retain objective metrics and saved-run semantics."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.config import Scenario
from charging_env.rl.returns import inventory_potential_rewards

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the train extra to test PPO improvements')
class TrainingImprovementTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        self.Trainer, self.Config = Trainer, TrainingConfig
        self.scenario = replace(Scenario(), horizon=3,
            zone_xy_km=((0., 0.), (2., 0.)), fcs_zones=(0,),
            initial_locations=(0, 1), initial_energy_kwh=(40., 45.),
            mean_requests_by_zone=(4., 4.), arrival_profile=(1., 1., 1.),
            grid_buy_per_kwh=(.12, .18, .14), grid_sell_per_kwh=(.09, .14, .10),
            fcs_retail_per_kwh=(.45,), fcs_ev_capacity_kwh=(40.,),
            fcs_fleet_exchange_capacity_kwh=(30.,), price_noise_distribution='none',
            charging_utility_intercept=3.).validate()
        self.base = TrainingConfig(seed=29, updates=5, episodes_per_update=2,
            epochs=2, minibatch_size=6, hidden_size=16, eval_episodes=2,
            dispatch_final_activation='silu', device='cpu', torch_threads=1)
        self.improved = replace(self.base, dispatch_agent_id=True, encoder_layer_norm=True,
            inventory_potential=.3, dispatch_entropy_coef=.03, dispatch_entropy_final_coef=.005,
            dispatch_entropy_anneal_updates=5, separate_network_grad_clip=True,
            training_seed_base=200000)

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

    def test_role_layernorm_and_shaped_checkpoint_continue_exactly(self):
        trainer = self.Trainer(self.scenario, self.improved)
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved.pt'
            trainer.save_checkpoint(path)
            expected_rollout = trainer.collect_rollout()
            expected = trainer.update(expected_rollout)
            expected_rng = torch.get_rng_state().clone()
            expected_numpy_rng = deepcopy(trainer.rng.bit_generator.state)
            restored = self.Trainer.load_checkpoint(path)
            self.assertEqual(restored.config, self.improved)
            self.assertTrue(restored.dispatch_actor.encoder.include_agent_id)
            for model in restored._modules().values():
                self.assertEqual(sum(isinstance(layer, torch.nn.LayerNorm) for layer in model.encoder.body), 2)
            actual_rollout = restored.collect_rollout()
            for field in ('pricing_actions', 'dispatch_actions', 'energy_fractions',
                          'old_price_log_probs', 'old_dispatch_log_probs', 'rewards',
                          'price_values', 'dispatch_values', 'next_price_values', 'dones'):
                np.testing.assert_array_equal(getattr(actual_rollout, field), getattr(expected_rollout, field))
            self.assertEqual(restored.update(actual_rollout), expected)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0., atol=0.)
            self.assertEqual(restored.rng.bit_generator.state, expected_numpy_rng)
            for name, model in trainer._modules().items():
                self.assert_nested_equal(restored._modules()[name].state_dict(), model.state_dict())
            self.assert_nested_equal(restored.actor_optimizer.state_dict(), trainer.actor_optimizer.state_dict())
            self.assert_nested_equal(restored.critic_optimizer.state_dict(), trainer.critic_optimizer.state_dict())

    def test_legacy_missing_fields_load_disabled_without_changing_outputs(self):
        fields = ('dispatch_agent_id', 'encoder_layer_norm', 'inventory_potential',
                  'dispatch_entropy_coef', 'dispatch_entropy_final_coef',
                  'dispatch_entropy_anneal_updates', 'separate_network_grad_clip', 'training_seed_base')
        trainer = self.Trainer(self.scenario, self.base)
        before = {mode: trainer.evaluate([1001, 1003], deterministic=mode, action_seed=921)
                  for mode in (False, True)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            for field in fields:
                payload['training_config'].pop(field)
            torch.save(payload, path)
            restored = self.Trainer.load_checkpoint(path)
            for field in fields:
                self.assertEqual(getattr(restored.config, field), getattr(self.base, field))
            self.assertFalse(restored.dispatch_actor.encoder.include_agent_id)
            for model in restored._modules().values():
                self.assertFalse(any(isinstance(layer, torch.nn.LayerNorm) for layer in model.encoder.body))
            for mode in (False, True):
                self.assertEqual(restored.evaluate([1001, 1003], deterministic=mode, action_seed=921), before[mode])

    def test_cli_rejects_role_layernorm_objective_and_seed_schedule_changes(self):
        from charging_env.rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved.pt'
            self.Trainer(self.scenario, self.improved).save_checkpoint(path)
            changes = [(['--no-dispatch-agent-id'], 'dispatch-agent-id'),
                       (['--no-encoder-layer-norm'], 'encoder-layer-norm'),
                       (['--inventory-potential', '0.'], 'inventory-potential'),
                       (['--training-seed-base', '0'], 'training-seed-base')]
            for mode in ('--resume', '--evaluate-only'):
                for flags, name in changes:
                    with self.subTest(mode=mode, flags=flags):
                        error = io.StringIO()
                        with patch.object(self.Trainer, 'train') as train, patch.object(
                                self.Trainer, 'evaluate') as evaluate, redirect_stderr(error):
                            with self.assertRaises(SystemExit) as caught:
                                main([mode, str(path), *flags])
                        self.assertEqual(caught.exception.code, 2)
                        self.assertIn(name, error.getvalue())
                        self.assertIn('cannot change', error.getvalue())
                        train.assert_not_called()
                        evaluate.assert_not_called()

    def test_shifted_training_seed_schedule_is_even_disjoint_and_continues(self):
        legacy = self.Trainer(self.scenario, self.base)
        old = legacy.collect_rollout()
        shifted = self.Trainer(self.scenario, replace(self.base, training_seed_base=200000))
        first, second = shifted.collect_rollout(), shifted.collect_rollout()
        old_seeds = [e['seed'] for e in old.episodes]
        new_seeds = [e['seed'] for e in first.episodes+second.episodes]
        self.assertEqual(old_seeds, [58, 60])
        self.assertEqual(new_seeds, [400058, 400060, 400062, 400064])
        self.assertTrue(all(seed % 2 == 0 for seed in new_seeds))
        self.assertFalse(set(old_seeds) & set(new_seeds))
        self.assertFalse(set(new_seeds) & {self.base.eval_seed_base+2*i for i in range(self.base.eval_episodes)})

    def test_training_shaping_never_changes_raw_rollout_or_profit_metrics(self):
        trainer = self.Trainer(self.scenario, self.improved)
        rollout = trainer.collect_rollout()
        raw, episodes = rollout.rewards.copy(), deepcopy(rollout.episodes)
        expected_learning = inventory_potential_rewards(raw, rollout.pre_obs, rollout.dones,
                                                        self.scenario.discount_per_period, .3)
        from charging_env.rl import trainer as trainer_module
        with patch.object(trainer_module, 'stage_gae', wraps=trainer_module.stage_gae) as gae:
            metrics = trainer.update(rollout)
        np.testing.assert_array_equal(gae.call_args.args[0], expected_learning)
        np.testing.assert_array_equal(rollout.rewards, raw)
        self.assertEqual(rollout.episodes, episodes)
        self.assertAlmostEqual(metrics['mean_learning_reward'], float(expected_learning.mean()))
        self.assertEqual(metrics['mean_training_profit'], float(np.mean([row['profit'] for row in episodes])))
        self.assertEqual(metrics['mean_training_discounted_profit'],
                         float(np.mean([row['discounted_profit'] for row in episodes])))

    def test_dispatch_entropy_schedule_hits_endpoints_and_preserves_pricing_coefficient(self):
        trainer = self.Trainer(self.scenario, self.improved)
        for count, expected in ((0, .03), (2, .0175), (4, .005), (9, .005)):
            with self.subTest(count=count):
                trainer.update_count = count
                result = trainer.update(trainer.collect_rollout())
                self.assertAlmostEqual(result['dispatch_entropy_coefficient'], expected, places=14)
                self.assertEqual(trainer.config.entropy_coef, self.base.entropy_coef)
                self.assertTrue(all(np.isfinite(value) for value in result.values()))
        default = self.Trainer(self.scenario, self.base)
        self.assertEqual(default.update(default.collect_rollout())['dispatch_entropy_coefficient'], self.base.entropy_coef)

    def test_cli_additional_update_count_does_not_retime_saved_entropy_schedule(self):
        from charging_env.rl.train import main
        trainer = self.Trainer(self.scenario, self.improved)
        trainer.update(trainer.collect_rollout())
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scheduled.pt'
            trainer.save_checkpoint(path)
            expected = trainer.update(trainer.collect_rollout())['dispatch_entropy_coefficient']
            observed = {}
            def probe(resumed, *args, **kwargs):
                observed['horizon'] = resumed.config.dispatch_entropy_anneal_updates
                observed['coefficient'] = resumed.update(resumed.collect_rollout())['dispatch_entropy_coefficient']
                return observed
            with patch.object(self.Trainer, 'train', autospec=True, side_effect=probe), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--resume', str(path), '--updates', '11',
                                       '--output-dir', str(Path(directory)/'resumed')]), 0)
            self.assertEqual(observed['horizon'], 5)
            self.assertEqual(observed['coefficient'], expected)

    def test_separate_clipping_limits_each_network_and_all_updates_stay_finite(self):
        trainer = self.Trainer(self.scenario, replace(self.improved, max_grad_norm=.01))
        expected_groups = {frozenset(id(parameter) for parameter in model.parameters())
                           for model in trainer._modules().values()}
        original_clip = torch.nn.utils.clip_grad_norm_
        groups, clipped_norms = [], []
        def record_clip(parameters, max_norm, **kwargs):
            parameters = list(parameters)
            groups.append(frozenset(id(parameter) for parameter in parameters))
            before = original_clip(parameters, max_norm, **kwargs)
            clipped_norms.append(float(torch.sqrt(sum(parameter.grad.square().sum()
                for parameter in parameters if parameter.grad is not None))))
            return before
        with patch('charging_env.rl.trainer.nn.utils.clip_grad_norm_', side_effect=record_clip):
            metrics = trainer.update(trainer.collect_rollout())
        self.assertEqual(set(groups), expected_groups)
        self.assertEqual(len(groups), 4*metrics['optimizer_steps'])
        self.assertTrue(all(norm <= .010001 for norm in clipped_norms))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        for name in trainer._modules():
            self.assertGreater(metrics[name+'_parameter_delta'], 0.)

    def test_new_configuration_rejects_invalid_fields_and_missing_schedule_horizon(self):
        invalid = [('dispatch_agent_id', 1), ('encoder_layer_norm', 'yes'),
                   ('separate_network_grad_clip', 0), ('training_seed_base', -1),
                   ('training_seed_base', 1.5), ('inventory_potential', -.1),
                   ('inventory_potential', np.nan), ('dispatch_entropy_coef', True),
                   ('dispatch_entropy_final_coef', np.inf), ('dispatch_entropy_anneal_updates', -1)]
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(self.base, **{field: value}).validate()
        with self.assertRaises(ValueError):
            replace(self.base, dispatch_entropy_final_coef=.005, dispatch_entropy_anneal_updates=0).validate()


if __name__ == '__main__':
    unittest.main()
