"""Dispatch-only activation selection, checkpoint compatibility, and PPO replay."""

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
from charging_env.simulator import ChargingSimulator, Mode

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the train extra to test dispatch activation')
class DispatchActivationTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        self.Trainer, self.TrainingConfig = Trainer, TrainingConfig
        self.scenario = replace(
            Scenario(), horizon=3, zone_xy_km=((0., 0.), (2., 0.)),
            fcs_zones=(0,), initial_locations=(0, 1), initial_energy_kwh=(40., 45.),
            mean_requests_by_zone=(4., 4.), arrival_profile=(1., 1., 1.),
            grid_buy_per_kwh=(.12, .18, .14), grid_sell_per_kwh=(.09, .14, .10),
            fcs_retail_per_kwh=(.45,), fcs_ev_capacity_kwh=(40.,),
            fcs_fleet_exchange_capacity_kwh=(30.,), price_noise_distribution='none',
            price_sensitivity_per_dollar=.1,
        ).validate()
        self.config = TrainingConfig(seed=29, updates=2, episodes_per_update=2,
                                     epochs=2, minibatch_size=6, hidden_size=16,
                                     eval_episodes=2, device='cpu', torch_threads=1)

    def trainer(self, activation='tanh'):
        return self.Trainer(self.scenario, replace(self.config, dispatch_final_activation=activation))

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

    def test_default_is_tanh_and_only_the_final_dispatch_body_activation_changes(self):
        self.assertEqual(self.config.dispatch_final_activation, 'tanh')
        original, modified = self.trainer(), self.trainer('silu')
        changes = []
        for model_name, original_model in original._modules().items():
            modified_model = modified._modules()[model_name]
            expected_types = {name: type(module) for name, module in original_model.named_modules()}
            actual_types = {name: type(module) for name, module in modified_model.named_modules()}
            self.assertEqual(set(actual_types), set(expected_types))
            changes.extend((model_name, name) for name in expected_types
                           if actual_types[name] is not expected_types[name])
            self.assert_nested_equal(modified_model.state_dict(), original_model.state_dict())
            self.assertIsInstance(modified_model.encoder.request_encoder[1], torch.nn.Tanh)
            self.assertIsInstance(modified_model.encoder.request_encoder[3], torch.nn.Tanh)
            self.assertIsInstance(modified_model.encoder.body[1], torch.nn.Tanh)
        self.assertEqual(changes, [('dispatch_actor', 'encoder.body.3')])
        self.assertIsInstance(original.dispatch_actor.encoder.body[3], torch.nn.Tanh)
        self.assertIsInstance(modified.dispatch_actor.encoder.body[3], torch.nn.SiLU)
        for name in ('pricing_actor', 'pricing_critic', 'dispatch_critic'):
            self.assertIsInstance(modified._modules()[name].encoder.body[3], torch.nn.Tanh)
        self.assertEqual(modified.scenario, original.scenario)

    def test_dispatch_constructor_default_and_supported_explicit_activations(self):
        from charging_env.rl.networks import DispatchActor
        default = DispatchActor(self.scenario, hidden_size=16)
        self.assertIsInstance(default.encoder.body[3], torch.nn.Tanh)
        for activation, expected in [('tanh', torch.nn.Tanh), ('silu', torch.nn.SiLU)]:
            with self.subTest(activation=activation):
                actor = DispatchActor(self.scenario, hidden_size=16, final_activation=activation)
                self.assertIsInstance(actor.encoder.body[3], expected)
                self.assertIsInstance(actor.encoder.body[1], torch.nn.Tanh)

    def test_invalid_constructor_and_configuration_values_are_rejected(self):
        from charging_env.rl.networks import DispatchActor
        for invalid in ('relu', 'SiLU', '', None, 7, True):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    replace(self.config, dispatch_final_activation=invalid).validate()
                with self.assertRaises(ValueError):
                    DispatchActor(self.scenario, hidden_size=16, final_activation=invalid)

    def test_legacy_checkpoint_without_activation_field_reproduces_tanh_outputs(self):
        trainer = self.trainer()
        trainer.update(trainer.collect_rollout())
        expected = {mode: trainer.evaluate([1001, 1003], deterministic=mode, action_seed=917)
                    for mode in (False, True)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['training_config'].pop('dispatch_final_activation')
            torch.save(payload, path)
            restored = self.Trainer.load_checkpoint(path)
            self.assertEqual(restored.config.dispatch_final_activation, 'tanh')
            self.assertIsInstance(restored.dispatch_actor.encoder.body[3], torch.nn.Tanh)
            for name, model in trainer._modules().items():
                self.assert_nested_equal(restored._modules()[name].state_dict(), model.state_dict())
            for mode in (False, True):
                self.assertEqual(restored.evaluate([1001, 1003], deterministic=mode, action_seed=917),
                                 expected[mode])

    def test_silu_checkpoint_restores_optimizer_rng_and_exact_training_continuation(self):
        trainer = self.trainer('silu')
        trainer.update(trainer.collect_rollout())
        expected_eval = trainer.evaluate([2001, 2003], deterministic=False, action_seed=301)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'silu.pt'
            trainer.save_checkpoint(path)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            expected_rollout = trainer.collect_rollout()
            expected_metrics = trainer.update(expected_rollout)
            expected_torch_rng = torch.get_rng_state().clone()
            expected_numpy_rng = deepcopy(trainer.rng.bit_generator.state)
            restored = self.Trainer.load_checkpoint(path)
            self.assertEqual(restored.config.dispatch_final_activation, 'silu')
            self.assertIsInstance(restored.dispatch_actor.encoder.body[3], torch.nn.SiLU)
            torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], rtol=0., atol=0.)
            self.assertEqual(restored.rng.bit_generator.state, saved['numpy_rng'])
            self.assert_nested_equal(restored.actor_optimizer.state_dict(), saved['actor_optimizer'])
            self.assert_nested_equal(restored.critic_optimizer.state_dict(), saved['critic_optimizer'])
            self.assertEqual(restored.evaluate([2001, 2003], deterministic=False, action_seed=301),
                             expected_eval)
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

    def test_silu_preserves_masks_likelihood_replay_and_finite_gradients(self):
        trainer = self.trainer('silu')
        sim = ChargingSimulator(self.scenario)
        sim.reset(seed=421)
        sim.quote_price_levels(np.zeros(self.scenario.n_zones, dtype=int))
        observations = [sim.local_observation(m) for m in range(self.scenario.n_mcv)]
        expected_actions = []
        for obs, mode in zip(observations, (Mode.DISCHARGE, Mode.RECHARGE)):
            self.assertTrue(obs['action_mask'][0, mode, 0])
            obs['action_mask'].fill(0)
            obs['action_mask'][0, mode, 0] = 1
            expected_actions.append(int(mode)*self.scenario.n_zones)
        actor = trainer.dispatch_actor
        sample = actor.sample(observations)
        torch.testing.assert_close(sample['actions'], torch.tensor(expected_actions))
        self.assertTrue(torch.isfinite(sample['log_prob']).all())
        self.assertTrue(torch.isfinite(sample['entropy']).all())
        self.assertTrue(((sample['energy_fraction'] > 0.) & (sample['energy_fraction'] < 1.)).all())
        replay = actor.evaluate_actions(observations, sample['actions'], sample['energy_fraction'])
        torch.testing.assert_close(replay['log_prob'], sample['log_prob'], rtol=0., atol=0.)
        actor.zero_grad()
        loss = -(replay['log_prob']*torch.tensor([1., -.4])).mean() - .01*replay['entropy'].mean()
        loss.backward()
        for parameter in actor.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(actor.encoder.body[0].weight.grad.norm()), 0.)
        self.assertGreater(float(actor.encoder.body[2].weight.grad.norm()), 0.)

    def test_cli_rejects_activation_changes_on_resume_and_evaluate(self):
        from charging_env.rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            for saved_activation, different in [('tanh', 'silu'), ('silu', 'tanh')]:
                path = Path(directory)/(saved_activation+'.pt')
                self.trainer(saved_activation).save_checkpoint(path)
                for mode in ('--resume', '--evaluate-only'):
                    with self.subTest(saved=saved_activation, mode=mode):
                        error = io.StringIO()
                        with patch.object(self.Trainer, 'train') as train, patch.object(
                                self.Trainer, 'evaluate') as evaluate, redirect_stderr(error):
                            with self.assertRaises(SystemExit) as caught:
                                main([mode, str(path), '--dispatch-final-activation', different])
                        self.assertEqual(caught.exception.code, 2)
                        self.assertIn('--dispatch-final-activation', error.getvalue())
                        self.assertIn('cannot change', error.getvalue())
                        train.assert_not_called()
                        evaluate.assert_not_called()

    def test_cli_evaluation_keeps_saved_silu_when_flag_is_omitted_or_unchanged(self):
        from charging_env.rl.train import main
        trainer = self.trainer('silu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'silu.pt'
            trainer.save_checkpoint(path)
            for extra in ([], ['--dispatch-final-activation', 'silu']):
                with self.subTest(extra=extra), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['--evaluate-only', str(path), *extra]), 0)
                self.assertIn('mean_discounted_profit', output.getvalue())

    def test_tiny_silu_training_is_finite_and_checkpoint_reload_matches(self):
        trainer = self.trainer('silu')
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summary = trainer.train(Path(directory)/'tiny_run')
            restored = self.Trainer.load_checkpoint(Path(directory)/'tiny_run/checkpoint.pt')
        self.assertTrue(summary['all_metrics_finite'])
        self.assertTrue(summary['checkpoint_reload_matches'])
        self.assertEqual(summary['invalid_evaluation_actions'], 0)
        self.assertEqual(summary['training_physical_periods'], 12)
        self.assertEqual(restored.config.dispatch_final_activation, 'silu')
        for delta in summary['parameter_change_by_module'].values():
            self.assertGreater(delta, 0.)


if __name__ == '__main__':
    unittest.main()
