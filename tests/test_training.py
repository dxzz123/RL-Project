"""Analytic stage-credit and executable PPO/MAPPO regression checks.

The torch checks are optional so the simulator-only installation remains usable.
They run on tiny synthetic scenarios; they do not establish policy convergence.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, Mode
from charging_env.rl.returns import stage_gae

try:
    import torch
except ImportError:
    torch = None


def training_scenario(**overrides):
    """A nontrivial four-period, two-vehicle fleet for fast policy checks."""
    values = dict(
        horizon=4,
        zone_xy_km=((0., 0.), (2., 0.)),
        fcs_zones=(0,),
        initial_locations=(0, 1),
        initial_energy_kwh=(40., 45.),
        mean_requests_by_zone=(4., 4.),
        arrival_profile=(1., 1., 1., 1.),
        grid_buy_per_kwh=(0.12, 0.18, 0.20, 0.14),
        grid_sell_per_kwh=(0.09, 0.14, 0.16, 0.10),
        fcs_retail_per_kwh=(0.45,),
        fcs_ev_capacity_kwh=(40.,),
        fcs_fleet_exchange_capacity_kwh=(30.,),
        price_noise_distribution="none",
        price_sensitivity_per_dollar=0.1,
    )
    values.update(overrides)
    return replace(Scenario(), **values).validate()


class StageCreditTests(unittest.TestCase):
    def test_discount_is_applied_once_per_physical_period(self):
        values = stage_gae([2., 4.], [0., 0.], [0., 0.], [0., 0.],
                           [False, True], gamma=0.5, gae_lambda=1.)
        for name, actual in values.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(actual, [4., 4.])

    def test_lambda_one_recovers_monte_carlo_for_both_stages(self):
        vp = np.array([1., 5., -4., 9.])
        vd = np.array([2., -3., 7., 1.])
        values = stage_gae([3., 7., 2., 11.], vp, vd,
                           [5., 999., 9., 888.], [False, True, False, True],
                           gamma=0.9, gae_lambda=1.)
        expected = np.array([9.3, 7., 11.9, 11.])
        np.testing.assert_allclose(values["pricing_returns"], expected)
        np.testing.assert_allclose(values["dispatch_returns"], expected)
        np.testing.assert_allclose(values["pricing_advantages"], expected - vp)
        np.testing.assert_allclose(values["dispatch_advantages"], expected - vd)

    def test_lambda_zero_bootstraps_at_the_next_pricing_state(self):
        vp = np.array([11., 13., 17.])
        vd = np.array([19., 23., 29.])
        values = stage_gae([2., 7., 99.], vp, vd, [13., 444., 55.],
                           [False, True, True], gamma=0.5, gae_lambda=0.)
        expected = np.array([8.5, 7., 99.])
        np.testing.assert_allclose(values["pricing_returns"], expected)
        np.testing.assert_allclose(values["dispatch_returns"], expected)
        np.testing.assert_allclose(values["pricing_advantages"], expected - vp)
        np.testing.assert_allclose(values["dispatch_advantages"], expected - vd)

    def test_episode_boundaries_cut_both_advantage_traces(self):
        kwargs = dict(price_values=[1., 2., 3.], dispatch_values=[5., 6., 7.],
                      next_price_values=[2., 999., 999.], dones=[0, 1, 1],
                      gamma=0.8, gae_lambda=0.7)
        first = stage_gae([1., 2., 3.], **kwargs)
        changed = stage_gae([1., 2., 1_000_000.], **kwargs)
        for name in first:
            with self.subTest(name=name):
                np.testing.assert_array_equal(first[name][:2], changed[name][:2])

    def test_malformed_rollouts_are_rejected(self):
        valid = dict(rewards=[1.], price_values=[0.], dispatch_values=[0.],
                     next_price_values=[0.], dones=[True], gamma=0.9, gae_lambda=0.95)
        for changes in ({"rewards": [[1.]]}, {"price_values": []},
                        {"rewards": [np.nan]}, {"dones": [2]},
                        {"gamma": 0.}, {"gae_lambda": 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                stage_gae(**(valid | changes))


@unittest.skipIf(torch is None, "Install the train extra to test PPO/MAPPO")
class PolicyNetworkTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.networks import DispatchActor, PricingActor, ValueNetwork
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.scenario = training_scenario()
        self.pricing = PricingActor(self.scenario, hidden_size=16)
        self.dispatch = DispatchActor(self.scenario, hidden_size=16)
        self.critic = ValueNetwork(self.scenario, hidden_size=16)

    def observations(self, counts=(4, 3)):
        sim = ChargingSimulator(self.scenario)
        pre = sim.reset(seed=5, options={"requests": list(counts)})[0]
        post = sim.quote_price_levels([0] * self.scenario.n_zones)[0]
        return pre, post, [sim.local_observation(m) for m in range(self.scenario.n_mcv)]

    def test_unavailable_pricing_factors_have_no_likelihood_or_gradient(self):
        pre, _, _ = self.observations()
        pre["pricing_mask"].fill(0)
        result = self.pricing.sample([pre])
        torch.testing.assert_close(result["actions"], torch.zeros((1, 2), dtype=torch.long))
        torch.testing.assert_close(result["log_prob"], torch.zeros(1))
        torch.testing.assert_close(result["entropy"], torch.zeros(1))
        self.assertFalse(result["active"].any())
        (result["log_prob"].sum() + result["entropy"].sum()).backward()
        for parameter in self.pricing.parameters():
            if parameter.grad is not None:
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_pricing_likelihood_is_joint_sum_over_active_nodes(self):
        pre, _, _ = self.observations()
        # A partially available level set also checks the actual categorical mask.
        pre["pricing_mask"][0].fill(0)
        pre["pricing_mask"][1].fill(0)
        pre["pricing_mask"][1, [1, 4]] = 1
        observations = [deepcopy(pre) for _ in range(20)]
        result = self.pricing.sample(observations)
        self.assertTrue((result["actions"][:, 0] == 0).all())
        self.assertTrue(torch.isin(result["actions"][:, 1], torch.tensor([1, 4])).all())
        logits = self.pricing(observations)
        categorical = torch.distributions.Categorical(logits=logits)
        expected = categorical.log_prob(result["actions"]).sum(-1)
        torch.testing.assert_close(result["log_prob"], expected)
        replay = self.pricing.evaluate_actions(observations, result["actions"])
        torch.testing.assert_close(torch.exp(replay["log_prob"] - result["log_prob"]), torch.ones(20))

    def test_dispatch_masks_and_conditional_beta_likelihood_replay(self):
        _, _, local = self.observations()
        modes = (Mode.WAIT, Mode.REPOSITION, Mode.RECHARGE, Mode.DISCHARGE)
        observations = []
        expected_actions = []
        for mode in modes:
            obs = deepcopy(local[0])
            obs["action_mask"].fill(0)
            target = 1 if mode == Mode.REPOSITION else 0
            obs["action_mask"][0, int(mode), target] = 1
            observations.append(obs)
            expected_actions.append(int(mode) * self.scenario.n_zones + target)
        sample = self.dispatch.sample(observations)
        torch.testing.assert_close(sample["actions"], torch.tensor(expected_actions))
        torch.testing.assert_close(sample["energy_fraction"][:2], torch.zeros(2))
        self.assertTrue(((sample["energy_fraction"][2:] > 0.)
                         & (sample["energy_fraction"][2:] < 1.)).all())
        logits, concentrations = self.dispatch(observations)
        expected_log_prob = torch.distributions.Categorical(logits=logits).log_prob(sample["actions"])
        for index in (2, 3):
            selected = concentrations[index, sample["actions"][index]]
            expected_log_prob[index] += torch.distributions.Beta(*selected).log_prob(
                sample["energy_fraction"][index])
        torch.testing.assert_close(sample["log_prob"], expected_log_prob)
        replay = self.dispatch.evaluate_actions(observations, sample["actions"], sample["energy_fraction"])
        torch.testing.assert_close(replay["log_prob"], sample["log_prob"])
        torch.testing.assert_close(replay["entropy"], sample["entropy"])
        self.assertTrue(torch.isfinite(sample["entropy"]).all())

    def test_ignored_energy_has_no_action_likelihood_gradient(self):
        _, _, local = self.observations()
        wait = torch.tensor([int(Mode.WAIT) * self.scenario.n_zones
                             + int(local[0]["locations"][0])])
        zero = self.dispatch.evaluate_actions([local[0]], wait, torch.tensor([0.]))
        other = self.dispatch.evaluate_actions([local[0]], wait, torch.tensor([0.8]))
        torch.testing.assert_close(zero["log_prob"], other["log_prob"])
        zero["log_prob"].sum().backward()
        for parameter in self.dispatch.energy_head.parameters():
            if parameter.grad is not None:
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_empty_and_large_request_sets_are_finite_and_batch_padding_is_inert(self):
        empty, empty_post, empty_local = self.observations(counts=(0, 0))
        large, large_post, large_local = self.observations(counts=(129, 127))
        self.assertGreater(len(large["request_details"]), self.scenario.max_requests_per_zone)
        with torch.no_grad():
            for actor, empty_obs, large_obs in ((self.pricing, empty, large),
                                                (self.dispatch, empty_local[0], large_local[0])):
                batch = actor.sample([empty_obs, large_obs], deterministic=True)
                individual = actor.sample([empty_obs], deterministic=True)
                self.assertTrue(torch.isfinite(batch["log_prob"]).all())
                self.assertTrue(torch.isfinite(batch["entropy"]).all())
                torch.testing.assert_close(batch["log_prob"][:1], individual["log_prob"], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(batch["actions"][:1], individual["actions"])
            values = self.critic([empty, empty_post, large, large_post])
            self.assertEqual(values.shape, (4,))
            self.assertTrue(torch.isfinite(values).all())
            torch.testing.assert_close(values[:1], self.critic([empty]), rtol=1e-5, atol=1e-6)
            # A request well past the old metadata cap still affects the encoder.
            modified = deepcopy(large)
            modified["request_details"][-1, 1] += 3.
            encoded = self.pricing.encoder([large, modified])
            self.assertGreater(torch.max(torch.abs(encoded[0] - encoded[1])).item(), 1e-8)

    def test_dispatch_actor_rejects_global_observations_and_empty_action_masks(self):
        _, post, local = self.observations()
        with self.assertRaisesRegex(ValueError, "local_observation"):
            self.dispatch.sample([post])
        invalid = deepcopy(local[0])
        invalid["action_mask"].fill(0)
        with self.assertRaisesRegex(ValueError, "feasible action"):
            self.dispatch.sample([invalid])


@unittest.skipIf(torch is None, "Install the train extra to test PPO/MAPPO")
class TrainerIntegrationTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.trainer import Trainer, TrainingConfig
        self.Trainer = Trainer
        self.config = TrainingConfig(seed=29, updates=1, episodes_per_update=2,
                                     epochs=2, minibatch_size=8, hidden_size=16,
                                     learning_rate=0.001, eval_episodes=2,
                                     target_kl=0.2, device="cpu", torch_threads=1)
        self.scenario = training_scenario()

    def assert_finite_numbers(self, value):
        if isinstance(value, dict):
            for child in value.values():
                self.assert_finite_numbers(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.assert_finite_numbers(child)
        elif isinstance(value, (int, float, np.number)):
            self.assertTrue(np.isfinite(value), repr(value))
        elif isinstance(value, np.ndarray):
            self.assertTrue(np.isfinite(value).all())

    def test_rollout_preserves_two_stage_timing_masks_and_on_policy_likelihoods(self):
        trainer = self.Trainer(self.scenario, self.config)
        rollout = trainer.collect_rollout()
        periods = self.config.episodes_per_update * self.scenario.horizon
        self.assertEqual(len(rollout.pre_obs), periods)
        self.assertEqual(len(rollout.post_obs), periods)
        self.assertEqual(len(rollout.local_obs), periods * self.scenario.n_mcv)
        self.assertEqual(np.asarray(rollout.dispatch_actions).shape, (periods, self.scenario.n_mcv))
        self.assertEqual(np.asarray(rollout.pricing_actions).shape, (periods, self.scenario.n_zones))
        self.assertEqual(np.count_nonzero(rollout.dones), self.config.episodes_per_update)
        np.testing.assert_array_equal(np.flatnonzero(rollout.dones), [3, 7])
        for t in range(periods):
            self.assertEqual(rollout.pre_obs[t]["stage"], 0)
            self.assertEqual(rollout.post_obs[t]["stage"], 1)
            self.assertEqual(int(rollout.pre_obs[t]["period"][0]), t % self.scenario.horizon)
            self.assertEqual(int(rollout.post_obs[t]["period"][0]), t % self.scenario.horizon)
            for m in range(self.scenario.n_mcv):
                local = rollout.local_obs[t * self.scenario.n_mcv + m]
                self.assertEqual(int(local["agent_id"][0]), m)
                mask = local["action_mask"].reshape(-1)
                self.assertTrue(mask[int(rollout.dispatch_actions[t, m])])
            for zone, action in enumerate(rollout.pricing_actions[t]):
                mask = rollout.pre_obs[t]["pricing_mask"][zone]
                self.assertTrue(mask[int(action)] if mask.any() else action == 0)
        with torch.no_grad():
            pricing = trainer.pricing_actor.evaluate_actions(rollout.pre_obs, rollout.pricing_actions)
            dispatch = trainer.dispatch_actor.evaluate_actions(
                rollout.local_obs, rollout.dispatch_actions.reshape(-1), rollout.energy_fractions.reshape(-1))
        np.testing.assert_allclose(np.exp(pricing["log_prob"].numpy() - rollout.old_price_log_probs),
                                   1., rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.exp(dispatch["log_prob"].numpy().reshape(periods, -1)
                                         - rollout.old_dispatch_log_probs),
                                   1., rtol=1e-5, atol=1e-6)

    def test_one_update_changes_both_actors_and_both_centralized_critics(self):
        trainer = self.Trainer(self.scenario, self.config)
        modules = {name: getattr(trainer, name) for name in
                   ("pricing_actor", "dispatch_actor", "pricing_critic", "dispatch_critic")}
        before = {name: [p.detach().clone() for p in module.parameters()]
                  for name, module in modules.items()}
        rollout = trainer.collect_rollout()
        metrics = trainer.update(rollout)
        self.assertIsInstance(metrics, dict)
        self.assertTrue(metrics)
        self.assert_finite_numbers(metrics)
        self.assertEqual(trainer.update_count, 1)
        for name, module in modules.items():
            with self.subTest(module=name):
                self.assertTrue(any(not torch.equal(previous, current)
                                    for previous, current in zip(before[name], module.parameters())))
                self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))

    def test_update_is_finite_when_every_pricing_decision_is_inactive(self):
        # At most 4 kWh can be delivered, less than every request's 5 kWh minimum.
        scenario = training_scenario(mobile_service_kw=4.)
        trainer = self.Trainer(scenario, self.config)
        rollout = trainer.collect_rollout()
        self.assertTrue(all(not obs["pricing_mask"].any() for obs in rollout.pre_obs))
        np.testing.assert_array_equal(rollout.old_price_log_probs, np.zeros(len(rollout)))
        metrics = trainer.update(rollout)
        self.assert_finite_numbers(metrics)
        self.assertEqual(metrics["pricing_policy_loss"], 0.)
        self.assertEqual(metrics["pricing_entropy"], 0.)
        self.assertEqual(metrics["pricing_actor_parameter_delta"], 0.)
        self.assertGreater(metrics["dispatch_actor_parameter_delta"], 0.)

    def test_checkpoint_restores_deterministic_evaluation_and_stochastic_continuation(self):
        trainer = self.Trainer(self.scenario, self.config)
        trainer.update(trainer.collect_rollout())
        seeds = [1001, 1002]
        before_evaluation = trainer.evaluate(seeds, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint)
            self.assertTrue(checkpoint.is_file())
            expected_continuation = trainer.collect_rollout()
            restored = self.Trainer.load_checkpoint(checkpoint, device="cpu")
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(restored.episode_count, self.config.episodes_per_update)
            self.assertEqual(restored.evaluate(seeds, deterministic=True), before_evaluation)
            self.assertTrue(restored.actor_optimizer.state)
            self.assertTrue(restored.critic_optimizer.state)
            actual_continuation = restored.collect_rollout()
            for name in ("pricing_actions", "dispatch_actions", "energy_fractions",
                         "old_price_log_probs", "old_dispatch_log_probs", "rewards"):
                with self.subTest(field=name):
                    np.testing.assert_array_equal(getattr(actual_continuation, name),
                                                  getattr(expected_continuation, name))

    def test_small_training_run_writes_checkpoint_and_finite_report(self):
        trainer = self.Trainer(self.scenario, self.config)
        with tempfile.TemporaryDirectory() as directory:
            summary = trainer.train(Path(directory))
            self.assertIsInstance(summary, dict)
            self.assert_finite_numbers(summary)
            self.assertEqual(trainer.update_count, 1)
            self.assertTrue(list(Path(directory).rglob("*.pt")))
            self.assertTrue(list(Path(directory).rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
