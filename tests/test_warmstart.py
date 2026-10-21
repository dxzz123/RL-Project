"""Small imitation checks, separate from PPO and independent evaluation."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import Mode

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the train extra to test warm starts')
class InventoryWarmStartTests(unittest.TestCase):
    def trainer(self, **overrides):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        config = dict(seed=43, updates=1, episodes_per_update=1, epochs=1,
                      minibatch_size=4, hidden_size=16, dispatch_agent_id=True,
                      dispatch_final_activation='silu')
        # Both remaining physical periods are cheap enough for exact teacher search.
        scenario = replace(Scenario(), horizon=2, initial_locations=(0, 0),
                           initial_energy_kwh=(5., 5.), mean_requests_by_zone=(0., 0., 0., 0.),
                           arrival_profile=(1., 1.), grid_buy_per_kwh=(.10, .10),
                           grid_sell_per_kwh=(.08, .08), charging_utility_intercept=3.).validate()
        config.update(overrides)
        return Trainer(scenario, TrainingConfig(**config))

    def test_actor_only_updates_preserve_critics_counters_and_ppo_optimizers(self):
        from charging_env.rl.trainer import TrainingConfig
        from charging_env.rl.warmstart import warm_start_from_inventory
        options = {'encoder_layer_norm': True} if 'encoder_layer_norm' in TrainingConfig.__dataclass_fields__ else {}
        trainer = self.trainer(**options)
        before = {name: deepcopy(module.state_dict()) for name, module in trainer._modules().items()}
        optimizers = (deepcopy(trainer.actor_optimizer.state_dict()), deepcopy(trainer.critic_optimizer.state_dict()))
        metrics = warm_start_from_inventory(trainer, episodes=2, epochs=2,
                                            seed_base=60000000, minibatch_size=2)
        self.assertEqual((trainer.update_count, trainer.episode_count, trainer.environment_periods), (0, 0, 0))
        self.assertEqual(trainer.actor_optimizer.state_dict(), optimizers[0])
        self.assertEqual(trainer.critic_optimizer.state_dict(), optimizers[1])
        for name in ('pricing_critic', 'dispatch_critic'):
            for key, value in getattr(trainer, name).state_dict().items():
                torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before['dispatch_actor'][key])
                            for key, value in trainer.dispatch_actor.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in trainer.actor_parameters))
        self.assertTrue(metrics['all_gradients_finite'])
        self.assertTrue(np.isfinite(metrics['largest_actor_gradient_norm']))
        self.assertEqual(metrics['demonstration_episodes'], 2)
        self.assertEqual(metrics['demonstration_periods'], 4)
        self.assertEqual(metrics['optimizer_steps'], 4)
        self.assertEqual(metrics['environment_seeds'], [60000000, 60000002])
        self.assertEqual(metrics['teacher_invalid_actions'], 0)
        self.assertGreater(metrics['in_sample_energy_labels'], 0)
        self.assertTrue(np.isfinite(metrics['in_sample_energy_fraction_mae']))
        json.dumps(metrics, allow_nan=False)

    def test_teacher_labels_are_legal_and_boundary_energy_mse_has_finite_gradients(self):
        from charging_env.rl.warmstart import _batch, _collect_demonstrations
        trainer = self.trainer()
        data = _collect_demonstrations(trainer, 1, 60000000, 4, .30)
        labels = data['dispatch_labels'].reshape(-1)
        for obs, label in zip(data['local_obs'], labels):
            self.assertTrue(obs['action_mask'].reshape(-1)[label])
        mask = np.isin(labels // trainer.scenario.n_zones, [Mode.RECHARGE, Mode.DISCHARGE])
        self.assertTrue(mask.any())
        self.assertTrue(((data['energy_targets'] >= 0) & (data['energy_targets'] <= 1)).all())
        for boundary in (0., 1.):
            with self.subTest(boundary=boundary):
                targets = deepcopy(data)
                targets['energy_targets'].reshape(-1)[mask] = boundary
                values = _batch(trainer, targets, np.arange(len(data['pre_obs'])))
                self.assertTrue(torch.isfinite(values['loss']))
                trainer.dispatch_actor.zero_grad(set_to_none=True)
                trainer.pricing_actor.zero_grad(set_to_none=True)
                values['loss'].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all()
                                    for p in trainer.actor_parameters if p.grad is not None))

    def test_invalid_arguments_are_rejected_before_any_actor_update(self):
        from charging_env.rl.warmstart import warm_start_from_inventory
        trainer = self.trainer()
        before = deepcopy(trainer.dispatch_actor.state_dict())
        invalid = [dict(episodes=0), dict(epochs=True), dict(minibatch_size=0),
                   dict(seed_base=3), dict(seed_base=-2), dict(seed_base=True),
                   dict(price_index=-1), dict(price_index=999), dict(price_index=True),
                   dict(shadow_price=-.1), dict(shadow_price=float('nan')),
                   dict(learning_rate=0), dict(learning_rate=float('inf'))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                warm_start_from_inventory(trainer, **kwargs)
        for key, value in trainer.dispatch_actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'dispatch_agent_id'):
            warm_start_from_inventory(self.trainer(dispatch_agent_id=False), episodes=1, epochs=1)
        trainer.update_count = 1
        with self.assertRaisesRegex(ValueError, 'zero PPO counters'):
            warm_start_from_inventory(trainer, episodes=1, epochs=1)


if __name__ == '__main__':
    unittest.main()
