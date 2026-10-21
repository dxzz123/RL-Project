"""Regression checks for reproducible evaluation and validation-only selection."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.config import Scenario

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the train extra to test PPO/MAPPO monitoring")
class TrainingMonitoringTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        self.Trainer = Trainer
        self.TrainingConfig = TrainingConfig
        self.scenario = replace(
            Scenario(), horizon=3, zone_xy_km=((0., 0.), (2., 0.)),
            fcs_zones=(0,), initial_locations=(0, 1),
            initial_energy_kwh=(40., 45.), mean_requests_by_zone=(4., 4.),
            arrival_profile=(1., 1., 1.), grid_buy_per_kwh=(.12, .18, .14),
            grid_sell_per_kwh=(.09, .14, .10), fcs_retail_per_kwh=(.45,),
            fcs_ev_capacity_kwh=(40.,), fcs_fleet_exchange_capacity_kwh=(30.,),
            price_noise_distribution="none", price_sensitivity_per_dollar=.1,
        ).validate()
        self.config = TrainingConfig(
            seed=29, updates=2, episodes_per_update=2, epochs=1,
            minibatch_size=6, hidden_size=16, learning_rate=.001,
            eval_episodes=3, device="cpu", torch_threads=1,
        )

    def assert_rollouts_equal(self, actual, expected):
        for name in ("pricing_actions", "dispatch_actions", "energy_fractions",
                     "old_price_log_probs", "old_dispatch_log_probs", "rewards"):
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_stochastic_evaluation_is_reproducible_and_does_not_mutate_training(self):
        trainer = self.Trainer(self.scenario, self.config)
        torch_state = torch.get_rng_state().clone()
        numpy_state = deepcopy(trainer.rng.bit_generator.state)
        counters = (trainer.update_count, trainer.episode_count, trainer.environment_periods)
        weights = {name: deepcopy(model.state_dict()) for name, model in trainer._modules().items()}

        actual = trainer.evaluate([1001, 1003, 1005], deterministic=False, action_seed=917)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        self.assertEqual(trainer.rng.bit_generator.state, numpy_state)
        self.assertEqual((trainer.update_count, trainer.episode_count, trainer.environment_periods), counters)
        for name, model in trainer._modules().items():
            for parameter_name, value in model.state_dict().items():
                torch.testing.assert_close(value, weights[name][parameter_name], rtol=0, atol=0)

        # Evaluation must depend on its own action seed, not preceding training draws.
        torch.rand(31)
        repeated = trainer.evaluate([1001, 1003, 1005], deterministic=False, action_seed=917)
        self.assertEqual(actual, repeated)

    def test_stochastic_validation_does_not_change_checkpoint_continuation(self):
        trainer = self.Trainer(self.scenario, self.config)
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            trainer.save_checkpoint(path)
            expected = trainer.collect_rollout()
            restored = self.Trainer.load_checkpoint(path, device="cpu")
            restored.evaluate([1001, 1003], deterministic=False, action_seed=301)
            actual = restored.collect_rollout()
        self.assert_rollouts_equal(actual, expected)

    def test_legacy_checkpoint_without_new_config_fields_loads_defaults(self):
        trainer = self.Trainer(self.scenario, self.config)
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for name in ("critic_learning_rate", "eval_interval", "eval_seed_base", "eval_policy",
                         "eval_action_seed", "discount_policy_loss"):
                payload["training_config"].pop(name, None)
            torch.save(payload, path)
            expected = trainer.collect_rollout()
            restored = self.Trainer.load_checkpoint(path, device="cpu")
            actual = restored.collect_rollout()
        self.assertEqual(restored.config.eval_interval, 0)
        self.assertIsNone(restored.config.critic_learning_rate)
        self.assertFalse(restored.config.discount_policy_loss)
        self.assertEqual(restored.config.eval_seed_base, 2 * self.config.seed + 1)
        self.assertEqual(restored.actor_optimizer.param_groups[0]["lr"], self.config.learning_rate)
        self.assertEqual(restored.critic_optimizer.param_groups[0]["lr"], self.config.learning_rate)
        self.assert_rollouts_equal(actual, expected)

    def test_actor_and_critic_learning_rates_are_independent(self):
        config = replace(self.config, learning_rate=.0002, critic_learning_rate=.0013)
        trainer = self.Trainer(self.scenario, config)
        self.assertEqual(trainer.actor_optimizer.param_groups[0]["lr"], .0002)
        self.assertEqual(trainer.critic_optimizer.param_groups[0]["lr"], .0013)

    def test_value_diagnostics_are_finite_when_targets_have_zero_variance(self):
        trainer = self.Trainer(self.scenario, self.config)
        rollout = trainer.collect_rollout()
        for name in ("rewards", "price_values", "dispatch_values", "next_price_values"):
            getattr(rollout, name).fill(0.)
        metrics = trainer.update(rollout)
        for stage in ("pricing", "dispatch"):
            self.assertEqual(metrics[f"{stage}_explained_variance"], 0.)
            self.assertTrue(np.isfinite(metrics[f"{stage}_value_rmse"]))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_discounted_policy_weighting_resets_each_episode_for_both_stages(self):
        from charging_env.rl.trainer import _normalize
        late_weights = torch.tensor([.5, .5], dtype=torch.float32).pow(1000)
        torch.testing.assert_close(_normalize(torch.tensor([1., 3.]), late_weights), torch.zeros(2))
        # At the unchanged behavior policy, weighted normalized advantages must
        # have zero surrogate mean. Entropy exposes the outer weights directly,
        # checking one discount per physical period and the per-episode reset.
        for enabled in (True, False):
            with self.subTest(discount_policy_loss=enabled):
                scenario = replace(self.scenario, discount_per_period=.5)
                config = replace(self.config, discount_policy_loss=enabled, gae_lambda=1.)
                trainer = self.Trainer(scenario, config)
                rollout = trainer.collect_rollout()
                rollout.rewards[:] = [1., 2., 4., 1., 2., 4.]
                for name in ("price_values", "dispatch_values", "next_price_values"):
                    getattr(rollout, name).fill(0.)
                weights = torch.tensor([1., .5, .25, 1., .5, .25] if enabled else [1.] * 6)
                active = torch.tensor([bool(obs["pricing_mask"].any()) for obs in rollout.pre_obs])
                with torch.no_grad():
                    pricing = trainer.pricing_actor.evaluate_actions(rollout.pre_obs, rollout.pricing_actions)
                    dispatch = trainer.dispatch_actor.evaluate_actions(
                        rollout.local_obs, rollout.dispatch_actions.reshape(-1),
                        rollout.energy_fractions.reshape(-1))
                    expected_pricing_entropy = float((pricing["entropy"] * weights)[active].mean())
                    expected_dispatch_entropy = float((dispatch["entropy"] * weights.repeat_interleave(2)).mean())
                metrics = trainer.update(rollout)
                self.assertAlmostEqual(metrics["pricing_policy_loss"], 0., places=5)
                self.assertAlmostEqual(metrics["dispatch_policy_loss"], 0., places=5)
                self.assertAlmostEqual(metrics["pricing_entropy"], expected_pricing_entropy, places=5)
                self.assertAlmostEqual(metrics["dispatch_entropy"], expected_dispatch_entropy, places=5)

    def test_resume_cli_actor_override_preserves_explicit_critic_rate(self):
        from charging_env.rl.train import main
        trainer = self.Trainer(self.scenario, replace(self.config, critic_learning_rate=.0013))
        captured = {}

        def record_training(resumed, output_dir, updates=None):
            captured["actor_lr"] = resumed.actor_optimizer.param_groups[0]["lr"]
            captured["critic_lr"] = resumed.critic_optimizer.param_groups[0]["lr"]
            return {"checked": True}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint)
            with patch.object(self.Trainer, "train", record_training), redirect_stdout(io.StringIO()):
                code = main(["--resume", str(checkpoint), "--learning-rate", ".0004",
                             "--output-dir", str(Path(directory) / "resumed")])
        self.assertEqual(code, 0)
        self.assertEqual(captured, {"actor_lr": .0004, "critic_lr": .0013})

    def test_resume_cli_critic_override_preserves_actor_rate(self):
        from charging_env.rl.train import main
        trainer = self.Trainer(self.scenario, self.config)
        captured = {}

        def record_training(resumed, output_dir, updates=None):
            captured["actor_lr"] = resumed.actor_optimizer.param_groups[0]["lr"]
            captured["critic_lr"] = resumed.critic_optimizer.param_groups[0]["lr"]
            return {"checked": True}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint)
            with patch.object(self.Trainer, "train", record_training), redirect_stdout(io.StringIO()):
                code = main(["--resume", str(checkpoint), "--critic-learning-rate", ".0017",
                             "--output-dir", str(Path(directory) / "resumed")])
        self.assertEqual(code, 0)
        self.assertEqual(captured, {"actor_lr": self.config.learning_rate, "critic_lr": .0017})

    def test_evaluation_uncertainty_counts_and_reward_components_reconcile(self):
        from charging_env.simulator import Mode
        trainer = self.Trainer(self.scenario, self.config)
        result = trainer.evaluate([1001, 1003, 1005], deterministic=False, action_seed=83)
        profits = np.asarray([episode["discounted_profit"] for episode in result["episodes"]])
        expected_se = float(profits.std(ddof=1) / np.sqrt(len(profits)))
        self.assertAlmostEqual(result["standard_error_discounted_profit"], expected_se)
        np.testing.assert_allclose(
            result["confidence_interval_95"],
            [profits.mean() - 1.96 * expected_se, profits.mean() + 1.96 * expected_se],
        )
        self.assertEqual(set(result["action_counts"]), {mode.name for mode in Mode})
        self.assertEqual(sum(result["action_counts"].values()),
                         len(profits) * self.scenario.horizon * self.scenario.n_mcv)
        for name, actual in result["mean_reward_terms"].items():
            self.assertAlmostEqual(actual, np.mean([e["reward_terms"][name] for e in result["episodes"]]))
        terms = result["mean_reward_terms"]
        reconstructed = sum((-1 if name.endswith("_cost") else 1) * value
                            for name, value in terms.items())
        self.assertAlmostEqual(result["mean_profit"], reconstructed)
        single = trainer.evaluate([1001])
        self.assertIsNone(single["confidence_interval_95"])

    def test_best_checkpoint_uses_configured_validation_policy_including_initial_policy(self):
        # Greedy validation peaks at update 1; sampled validation peaks before training.
        # The last update is worse in both cases, and must not replace the selected policy.
        evaluations = {True: [10., 20., 5.], False: [100., 90., 80.]}
        real_evaluate = self.Trainer.evaluate

        def controlled_evaluate(trainer, seeds, deterministic=True, action_seed=None):
            result = real_evaluate(trainer, seeds, deterministic=deterministic, action_seed=action_seed)
            result["mean_discounted_profit"] = evaluations[deterministic][trainer.update_count]
            return result

        for policy, expected_update, expected_score in (("deterministic", 1, 20.), ("stochastic", 0, 100.)):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                trainer = self.Trainer(self.scenario, replace(self.config, eval_interval=1, eval_policy=policy))
                with patch.object(self.Trainer, "evaluate", controlled_evaluate), redirect_stdout(io.StringIO()):
                    summary = trainer.train(output)
                selection = json.loads((output / "best_validation.json").read_text())
                best = self.Trainer.load_checkpoint(output / "best_checkpoint.pt")
                final = self.Trainer.load_checkpoint(output / "checkpoint.pt")
                self.assertEqual(summary["best_update"], expected_update)
                self.assertEqual(summary["best_validation_mean_discounted_profit"], expected_score)
                self.assertEqual(selection["update"], expected_update)
                self.assertEqual(selection["policy"], policy)
                self.assertEqual(selection["mean_discounted_profit"], expected_score)
                self.assertEqual(best.update_count, expected_update)
                self.assertEqual(final.update_count, self.config.updates)
                self.assertTrue((output / "validation_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
