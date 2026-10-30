"""Independent checks of the two-stage Joint PPO equations.

These small numerical and physical counterexamples check implementation
correctness; they make no claim about long-run policy convergence.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode

try:
    import torch
except ImportError:
    torch = None


def paper_scenario(**changes):
    settings = dict(
        horizon=3, zone_xy_km=((0., 0.), (2., 0.)), fcs_zones=(0,),
        initial_locations=(0, 1), initial_energy_kwh=(40., 45.),
        mean_requests_by_zone=(4., 4.), arrival_profile=(1., 1., 1.),
        grid_buy_per_kwh=(.12, .18, .14), grid_sell_per_kwh=(.09, .14, .1),
        fcs_retail_per_kwh=(.45,), fcs_ev_capacity_kwh=(30.,),
        fcs_fleet_exchange_capacity_kwh=(30.,), price_noise_distribution='none',
        charging_utility_intercept=3., price_sensitivity_per_dollar=.1,
        # Physics and legacy-pricing regressions: keep the fixed absolute menu
        # and the profile/shock station prices. See tests/test_station_pricing.py
        # for the LADWP base-price and premium-menu behaviour.
        fcs_pricing_mode="legacy", mcv_pricing_mode="absolute",
    )
    settings.update(changes)
    return replace(Scenario(), **settings).validate()


def observations(scenario=None, counts=(4, 3)):
    scenario = scenario or paper_scenario()
    simulator = ChargingSimulator(scenario)
    pre = simulator.reset(seed=91, options={'requests': list(counts)})[0]
    post = simulator.quote_price_levels([4] * scenario.n_zones)[0]
    return pre, post


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(child) for key, child in value.items() if key != 'mean_policy_decision_ms'}
    if isinstance(value, list):
        return [without_timing(child) for child in value]
    return value


def assert_finite(test, value):
    if isinstance(value, dict):
        for child in value.values():
            assert_finite(test, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite(test, child)
    elif torch is not None and torch.is_tensor(value):
        test.assertTrue(torch.isfinite(value).all(), repr(value))
    elif isinstance(value, (float, int, np.number, np.ndarray)):
        test.assertTrue(np.isfinite(value).all(), repr(value))


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class PaperGAETests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.paper_ppo import interleaved_gae
        self.gae = interleaved_gae
        self.arguments = dict(pricing_rewards=[1., 2.], dispatch_rewards=[4., 8.],
            price_values=[3., 5.], dispatch_values=[6., 7.], next_price_values=[5., 999.],
            dones=[False, True], gamma=.8, gae_lambda=.5)

    def test_exact_equations_apply_lambda_on_both_substage_edges(self):
        actual = self.gae(**self.arguments)
        np.testing.assert_allclose(actual['pricing_advantages'], [5.9, 4.5], atol=1e-12)
        np.testing.assert_allclose(actual['dispatch_advantages'], [3.8, 1.], atol=1e-12)
        np.testing.assert_allclose(actual['pricing_returns'], [8.9, 9.5], atol=1e-12)
        np.testing.assert_allclose(actual['dispatch_returns'], [9.8, 8.], atol=1e-12)

    def test_zero_lambda_pricing_target_is_immediate_pricing_reward_plus_dispatch_value(self):
        actual = self.gae(**(self.arguments | {'gae_lambda': 0.}))
        np.testing.assert_allclose(actual['pricing_advantages'], [4., 4.])
        np.testing.assert_allclose(actual['dispatch_advantages'], [2., 1.])
        np.testing.assert_allclose(actual['pricing_returns'], [7., 9.])
        np.testing.assert_allclose(actual['dispatch_returns'], [8., 8.])

    def test_unit_lambda_recovers_distinct_stage_monte_carlo_returns(self):
        actual = self.gae(**(self.arguments | {'gae_lambda': 1.}))
        # The dispatch return excludes the pricing reward already received in
        # that period; both include both rewards in future periods.
        np.testing.assert_allclose(actual['pricing_returns'], [13., 10.])
        np.testing.assert_allclose(actual['dispatch_returns'], [12., 8.])

    def test_two_substage_traces_decay_by_gamma_times_lambda_squared_per_period(self):
        actual = self.gae([0., 0., 0.], [0., 0., 1.], [0., 0., 0.],
                          [0., 0., 0.], [0., 0., 0.], [False, False, True], .8, .5)
        np.testing.assert_allclose(actual['pricing_advantages'], [.02, .1, .5], atol=1e-12)
        np.testing.assert_allclose(actual['dispatch_advantages'], [.04, .2, 1.], atol=1e-12)

    def test_episode_boundaries_cut_both_advantage_traces_and_bootstraps(self):
        settings = dict(pricing_rewards=[1., 2., 3.], dispatch_rewards=[4., 5., 6.],
            price_values=[7., 8., 9.], dispatch_values=[10., 11., 12.],
            next_price_values=[8., 999., 999.], dones=[False, True, True], gamma=.8, gae_lambda=.7)
        first = self.gae(**settings)
        changed = self.gae(**(settings | {'pricing_rewards': [1., 2., 1e8],
                                        'dispatch_rewards': [4., 5., -1e8]}))
        for name in first:
            np.testing.assert_array_equal(first[name][:2], changed[name][:2])
        terminal = self.gae(**(self.arguments | {'dones': [True, True]}))
        np.testing.assert_allclose(terminal['dispatch_advantages'], [-2., 1.])

    def test_empty_misaligned_nonfinite_and_invalid_discount_inputs_are_rejected(self):
        cases = ({'pricing_rewards': []}, {'dispatch_rewards': [1.]},
                 {'next_price_values': [np.nan, 0.]}, {'dones': [2, 1]},
                 {'gamma': 0.}, {'gae_lambda': 1.1}, {'gae_lambda': -.1})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.gae(**(self.arguments | changes))
        with self.assertRaises(ValueError):
            self.gae([], [], [], [], [], [], .9, .95)


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class PaperJointObjectiveTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.paper_ppo import joint_ppo_loss
        self.loss = joint_ppo_loss

    def test_probability_ratio_is_product_of_vehicle_ratios_then_clipped_once(self):
        per_vehicle_ratios = torch.tensor([[2., .5], [2., 2.], [.5, .5]], dtype=torch.float64)
        old = torch.full_like(per_vehicle_ratios, -2.)
        new = old + per_vehicle_ratios.log()
        advantages = torch.tensor([1., 1., -1.], dtype=torch.float64)
        result = self.loss(new, old, advantages, .2)
        torch.testing.assert_close(result['ratio'], torch.tensor([1., 4., .25], dtype=torch.float64))
        torch.testing.assert_close(result['log_ratio'], per_vehicle_ratios.log().sum(-1))
        expected = torch.tensor([1., 1.2, -.8], dtype=torch.float64)
        torch.testing.assert_close(result['surrogate'], expected)
        torch.testing.assert_close(result['loss'], -expected.mean())
        # For the balanced first row, independent vehicle clipping gives .85;
        # the joint action is unchanged in probability and must contribute 1.
        independently_clipped = torch.minimum(per_vehicle_ratios[0], per_vehicle_ratios[0].clamp(.8, 1.2)).mean()
        self.assertNotAlmostEqual(float(result['surrogate'][0]), float(independently_clipped))

    def test_negative_advantages_keep_unclipped_penalty_above_upper_bound(self):
        old = torch.tensor([[-3., -3.]], dtype=torch.float64)
        new = old + torch.tensor([[2., 2.]], dtype=torch.float64).log()
        result = self.loss(new, old, torch.tensor([-1.], dtype=torch.float64), .2)
        torch.testing.assert_close(result['surrogate'], torch.tensor([-4.], dtype=torch.float64))
        self.assertAlmostEqual(float(result['loss']), 4.)

    def test_entropy_is_sum_over_vehicles_then_average_over_periods(self):
        log_probs = torch.tensor([[-1., -2.], [-2., -1.]], dtype=torch.float64, requires_grad=True)
        entropies = torch.tensor([[.2, .4], [.3, .5]], dtype=torch.float64)
        result = self.loss(log_probs, log_probs.detach(), torch.tensor([1., 2.], dtype=torch.float64),
                           .2, entropies=entropies)
        self.assertAlmostEqual(float(result['entropy']), .7)
        result['loss'].backward()
        torch.testing.assert_close(log_probs.grad, torch.tensor([[-.5, -.5], [-1., -1.]], dtype=torch.float64))

    def test_bad_joint_shapes_nonfinite_values_and_empty_batches_are_rejected(self):
        valid = torch.zeros((2, 2))
        advantages = torch.ones(2)
        for new, old, adv in ((valid, valid[:1], advantages), (valid, valid, torch.ones(3)),
                              (valid[:0], valid[:0], advantages[:0]),
                              (torch.full_like(valid, float('nan')), valid, advantages)):
            with self.assertRaises(ValueError):
                self.loss(new, old, adv, .2)


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class PaperCriticTargetTests(unittest.TestCase):
    def test_paper_and_optional_pricing_advantage_dispatch_targets_are_explicitly_distinct(self):
        from charging_env.rl.paper_ppo import critic_targets, interleaved_gae
        gae = interleaved_gae([1., 2.], [4., 8.], [3., 5.], [6., 7.],
                              [5., 999.], [False, True], .8, .5)
        consistent = critic_targets(gae, [3., 5.], [6., 7.])
        alternative = critic_targets(gae, [3., 5.], [6., 7.], 'pricing_advantage')
        np.testing.assert_allclose(consistent['pricing_returns'], [8.9, 9.5])
        np.testing.assert_allclose(alternative['pricing_returns'], [8.9, 9.5])
        np.testing.assert_allclose(consistent['dispatch_returns'], [9.8, 8.])
        np.testing.assert_allclose(alternative['dispatch_returns'], [11.9, 11.5])
        with self.assertRaises(ValueError):
            critic_targets(gae, [3., 5.], [6., 7.], 'auto')


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class PaperInformationAndActionsTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl import paper_ppo
        self.paper = paper_ppo
        self.scenario = paper_scenario()

    def known_mobile_requests(self):
        simulator = ChargingSimulator(self.scenario)
        simulator.reset(seed=13, options={'requests': [1, 1], 'request_energy_kwh': [5., 5.]})
        simulator.quote_prices([.6, .6])
        simulator.request_choices = np.array([0, 0])
        simulator._refresh_demand()
        return simulator, simulator.observe()

    def test_pricing_information_excludes_realized_customers_and_congestion(self):
        pre, _ = observations(self.scenario)
        original = deepcopy(pre)
        hidden_fields = ('requests_by_zone', 'potential_energy_kwh', 'request_details',
                         'mobile_prices', 'mobile_demand', 'fixed_demand', 'mobile_requested_kwh',
                         'fixed_requested_kwh', 'outside_requests', 'mobile_wait_hours',
                         'fcs_wait_hours', 'mobile_available', 'eligible_mcvs', 'pricing_mask', 'action_mask')
        sparse = {key: value for key, value in pre.items() if key not in hidden_fields}
        baseline = self.paper.pricing_features(pre, self.scenario)
        np.testing.assert_array_equal(self.paper.pricing_features(sparse, self.scenario), baseline)
        changed = deepcopy(pre)
        for key in hidden_fields:
            changed[key] = np.full(np.shape(changed[key]), 999., dtype=float)
        np.testing.assert_array_equal(self.paper.pricing_features(changed, self.scenario), baseline)
        sanitized = self.paper.paper_pricing_observation(pre, self.scenario)
        for key in ('requests_by_zone', 'request_details', 'mobile_wait_hours', 'fcs_wait_hours',
                    'mobile_demand', 'fixed_demand'):
            self.assertNotIn(key, sanitized)
        for key in pre:
            np.testing.assert_array_equal(pre[key], original[key])
        visible_changed = deepcopy(pre)
        visible_changed['energy_kwh'][0] -= 2.
        self.assertFalse(np.array_equal(self.paper.pricing_features(visible_changed, self.scenario), baseline))

    def test_four_operation_action_mask_preserves_feasibility_and_dynamic_request_nodes(self):
        _, obs = self.known_mobile_requests()
        mask = self.paper.paper_dispatch_mask(obs, self.scenario)
        operation = self.paper.PaperOperation
        self.assertEqual(mask.shape, (self.scenario.n_mcv, 4, self.scenario.n_zones))
        self.assertTrue(np.issubdtype(mask.dtype, np.bool_))
        np.testing.assert_array_equal(mask[:, operation.SERVICE], obs['action_mask'][:, Mode.SERVE])
        np.testing.assert_array_equal(mask[:, operation.CHARGE], obs['action_mask'][:, Mode.RECHARGE])
        np.testing.assert_array_equal(mask[:, operation.DISCHARGE], obs['action_mask'][:, Mode.DISCHARGE])
        self.assertTrue(mask[:, operation.IDLE].any(axis=1).all())
        self.assertEqual(len(operation), 4)
        with self.assertRaises(ValueError):
            self.paper.paper_dispatch_mask(observations(self.scenario)[0], self.scenario)

    def test_idle_at_current_node_waits_and_idle_elsewhere_repositions(self):
        _, obs = self.known_mobile_requests()
        idle = int(self.paper.PaperOperation.IDLE) * self.scenario.n_zones
        commands = self.paper.paper_dispatch_commands(obs, np.array([idle, idle]), self.scenario)
        self.assertEqual(commands, [DispatchAction(Mode.WAIT, 0), DispatchAction(Mode.REPOSITION, 0)])
        self.assertTrue(all(action.energy_kwh == 0. for action in commands))

    def test_charge_and_discharge_use_full_individual_physical_quantity_without_beta(self):
        simulator, obs = self.known_mobile_requests()
        before = deepcopy(obs)
        for operation, mode in ((self.paper.PaperOperation.CHARGE, Mode.RECHARGE),
                                (self.paper.PaperOperation.DISCHARGE, Mode.DISCHARGE)):
            flat = np.full(self.scenario.n_mcv, int(operation) * self.scenario.n_zones, dtype=int)
            commands = self.paper.paper_dispatch_commands(obs, flat, self.scenario)
            for m, command in enumerate(commands):
                self.assertEqual(command.mode, mode)
                self.assertEqual(command.target_zone, 0)
                distance = self.scenario.distances[obs['locations'][m], 0]
                hours = self.scenario.period_hours - distance / self.scenario.speed_kmph
                energy = obs['energy_kwh'][m] - distance*self.scenario.travel_kwh_per_km
                expected = (min(hours*self.scenario.mobile_charge_kw,
                                (self.scenario.battery_capacity_kwh-energy)/self.scenario.charge_efficiency)
                            if mode == Mode.RECHARGE else
                            min(hours*self.scenario.mobile_discharge_kw,
                                (energy-self.scenario.reserve_kwh)*self.scenario.discharge_efficiency))
                self.assertAlmostEqual(command.energy_kwh, expected)
            trial = deepcopy(simulator)
            _, _, _, _, info = trial.dispatch(commands)
            self.assertFalse(info['invalid_actions'])
            ledger = 'charge_grid_kwh' if mode == Mode.RECHARGE else 'export_grid_kwh'
            self.assertAlmostEqual(float(info['energy_ledger'][ledger].sum()),
                                   min(sum(action.energy_kwh for action in commands),
                                       self.scenario.fcs_fleet_exchange_capacity_kwh[0]))
        for key in obs:
            np.testing.assert_array_equal(obs[key], before[key])

    def test_zero_demand_reserve_energy_has_safe_idle_fallback_at_unreachable_nonstation(self):
        scenario = paper_scenario(initial_locations=(1, 1), initial_energy_kwh=(5., 5.))
        _, obs = observations(scenario, counts=(0, 0))
        mask = self.paper.paper_dispatch_mask(obs, scenario)
        operation = self.paper.PaperOperation
        self.assertFalse(mask[:, operation.SERVICE].any())
        self.assertTrue(mask.reshape(scenario.n_mcv, -1).any(axis=1).all())
        self.assertTrue(mask[:, operation.IDLE, 1].all())
        flat = np.full(scenario.n_mcv, int(operation.IDLE)*scenario.n_zones+1, dtype=int)
        commands = self.paper.paper_dispatch_commands(obs, flat, scenario)
        self.assertEqual(commands, [DispatchAction(Mode.WAIT, 1), DispatchAction(Mode.WAIT, 1)])

    def test_dispatch_commands_reject_wrong_shape_noninteger_and_masked_actions(self):
        _, obs = self.known_mobile_requests()
        bad = (np.array([0]), np.array([0., 0.]), np.array([True, False]), np.array([999, 0]),
               np.array([int(self.paper.PaperOperation.CHARGE)*self.scenario.n_zones+1, 0]))
        for actions in bad:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.paper.paper_dispatch_commands(obs, actions, self.scenario)


@unittest.skipIf(torch is None, 'Install the train extra for Joint PPO')
class PaperNetworkAndTrainerTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl import paper_ppo
        self.paper = paper_ppo
        self.scenario = paper_scenario()
        self.config = paper_ppo.PaperPPOConfig(seed=29, updates=1, episodes_per_update=2,
            epochs=2, minibatch_size=6, hidden_size=16, actor_learning_rate=.001,
            eval_episodes=2, target_kl=.2, torch_threads=1)

    def trainer(self, **changes):
        return self.paper.PaperPPOTrainer(self.scenario, replace(self.config, **changes))

    def modules(self, trainer):
        return {name: getattr(trainer, name) for name in
                ('pricing_actor', 'pricing_critic', 'dispatch_actor', 'dispatch_critic')}

    def test_pricing_actor_and_critic_are_invariant_to_realized_request_fields(self):
        trainer = self.trainer()
        pre, _ = observations(self.scenario)
        changed = deepcopy(pre)
        for name in ('request_details', 'requests_by_zone', 'potential_energy_kwh',
                     'mobile_demand', 'fixed_demand', 'mobile_requested_kwh',
                     'fixed_requested_kwh', 'outside_requests', 'mobile_wait_hours', 'fcs_wait_hours'):
            changed[name] = np.full(np.shape(changed[name]), 999., dtype=float)
        with torch.no_grad():
            actor = trainer.pricing_actor.sample([pre, changed], deterministic=True)
            critic = trainer.pricing_critic([pre, changed])
        torch.testing.assert_close(actor['actions'][0], actor['actions'][1], atol=0, rtol=0)
        torch.testing.assert_close(actor['log_prob'][0], actor['log_prob'][1], atol=0, rtol=0)
        torch.testing.assert_close(actor['entropy'][0], actor['entropy'][1], atol=0, rtol=0)
        torch.testing.assert_close(critic[0], critic[1], atol=0, rtol=0)

    def test_dispatch_sampling_is_a_masked_joint_categorical_with_summed_entropy(self):
        trainer = self.trainer()
        _, post = observations(self.scenario)
        with torch.no_grad():
            for parameter in trainer.dispatch_actor.parameters():
                parameter.zero_()
            result = trainer.dispatch_actor.sample([post] * 16)
            replay = trainer.dispatch_actor.evaluate_actions([post] * 16, result['actions'])
        mask = self.paper.paper_dispatch_mask(post, self.scenario).reshape(self.scenario.n_mcv, -1)
        counts = mask.sum(-1)
        expected_log_prob = -np.log(counts)
        self.assertNotIn('energy_fraction', result)
        self.assertNotIn('energy_fractions', result)
        self.assertEqual(tuple(result['actions'].shape), (16, self.scenario.n_mcv))
        for t in range(16):
            for m in range(self.scenario.n_mcv):
                self.assertTrue(mask[m, int(result['actions'][t, m])])
        np.testing.assert_allclose(result['agent_log_probs'].numpy(),
                                   np.repeat(expected_log_prob[None], 16, axis=0), rtol=1e-6)
        np.testing.assert_allclose(result['agent_entropies'].numpy(),
                                   np.repeat(-expected_log_prob[None], 16, axis=0), rtol=1e-6)
        torch.testing.assert_close(result['log_prob'], result['agent_log_probs'].sum(-1))
        torch.testing.assert_close(result['entropy'], result['agent_entropies'].sum(-1))
        torch.testing.assert_close(replay['log_prob'], result['log_prob'])
        torch.testing.assert_close(replay['entropy'], result['entropy'])

    def test_empty_requests_and_fully_inactive_pricing_are_finite_without_policy_gradient(self):
        scenario = paper_scenario(initial_locations=(1, 1), initial_energy_kwh=(5., 5.),
                                  mean_requests_by_zone=(0., 0.))
        trainer = self.paper.PaperPPOTrainer(scenario, self.config)
        before = [parameter.detach().clone() for parameter in trainer.pricing_actor.parameters()]
        rollout = trainer.collect_rollout()
        np.testing.assert_array_equal(rollout.old_price_log_probs, 0.)
        np.testing.assert_array_equal(rollout.old_dispatch_log_probs, 0.)
        self.assertTrue(all(not observation['pricing_mask'].any() for observation in rollout.pre_obs))
        self.assertTrue(all(len(observation['request_details']) == 0 for observation in rollout.post_obs))
        metrics = trainer.update(rollout)
        assert_finite(self, metrics)
        self.assertEqual(metrics['pricing_entropy'], 0.)
        self.assertEqual(metrics['dispatch_entropy'], 0.)
        for original, parameter in zip(before, trainer.pricing_actor.parameters()):
            torch.testing.assert_close(original, parameter, atol=0, rtol=0)

    def test_rollout_freezes_networks_and_replays_the_original_joint_likelihoods(self):
        trainer = self.trainer()
        modules = self.modules(trainer)
        before = {name: [parameter.detach().clone() for parameter in module.parameters()]
                  for name, module in modules.items()}
        rollout = trainer.collect_rollout()
        periods = self.scenario.horizon * self.config.episodes_per_update
        self.assertEqual(len(rollout.pre_obs), periods)
        self.assertEqual(len(rollout.post_obs), periods)
        self.assertEqual(np.asarray(rollout.dispatch_actions).shape, (periods, self.scenario.n_mcv))
        self.assertEqual(np.asarray(rollout.pricing_actions).shape, (periods, self.scenario.n_zones))
        np.testing.assert_array_equal(np.flatnonzero(rollout.dones), [2, 5])
        np.testing.assert_array_equal(rollout.pricing_rewards, np.zeros(periods))
        for t, pre in enumerate(rollout.pre_obs):
            self.assertEqual(int(pre['stage']), 0)
            self.assertNotIn('request_details', pre)
            self.assertEqual(int(pre['period'][0]), t % self.scenario.horizon)
            self.assertEqual(int(rollout.post_obs[t]['stage']), 1)
        with torch.no_grad():
            pricing = trainer.pricing_actor.evaluate_actions(rollout.pre_obs, rollout.pricing_actions)
            dispatch = trainer.dispatch_actor.evaluate_actions(rollout.post_obs, rollout.dispatch_actions)
        np.testing.assert_allclose(np.exp(pricing['log_prob'].numpy()-rollout.old_price_log_probs), 1., atol=1e-6)
        np.testing.assert_allclose(np.exp(dispatch['agent_log_probs'].numpy()-rollout.old_dispatch_log_probs), 1., atol=1e-6)
        for name, module in modules.items():
            for previous, parameter in zip(before[name], module.parameters()):
                torch.testing.assert_close(previous, parameter, atol=0, rtol=0)

    def test_update_changes_all_four_networks_with_finite_parameters_and_metrics(self):
        trainer = self.trainer()
        modules = self.modules(trainer)
        before = {name: [parameter.detach().clone() for parameter in module.parameters()]
                  for name, module in modules.items()}
        rollout = trainer.collect_rollout()
        snapshot = deepcopy(rollout)
        metrics = trainer.update(rollout)
        assert_finite(self, metrics)
        self.assertEqual(trainer.update_count, 1)
        for name, module in modules.items():
            self.assertTrue(any(not torch.equal(previous, parameter)
                                for previous, parameter in zip(before[name], module.parameters())), name)
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in module.parameters()))
        for name in ('pricing_actions', 'dispatch_actions', 'old_price_log_probs', 'old_dispatch_log_probs',
                     'pricing_rewards', 'dispatch_rewards', 'dones'):
            np.testing.assert_array_equal(getattr(rollout, name), getattr(snapshot, name))

    def test_paper_and_optional_pricing_advantage_critic_modes_are_saved_distinctly(self):
        self.assertEqual(self.config.dispatch_critic_target, 'dispatch')
        trained = {}
        for target in ('dispatch', 'pricing_advantage'):
            trainer = self.trainer(dispatch_critic_target=target)
            metrics = trainer.update(trainer.collect_rollout())
            assert_finite(self, metrics)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'paper.pt'
                trainer.save_checkpoint(path)
                restored = self.paper.PaperPPOTrainer.load_checkpoint(path)
            self.assertEqual(restored.config.dispatch_critic_target, target)
            trained[target] = trainer
        # With identical fresh data, the chosen critic target changes only the
        # dispatch critic in this update. The actors use the same frozen GAE.
        for name in ('pricing_actor', 'pricing_critic', 'dispatch_actor'):
            for first, second in zip(self.modules(trained['dispatch'])[name].parameters(),
                                     self.modules(trained['pricing_advantage'])[name].parameters()):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(first, second) for first, second in zip(
            trained['dispatch'].dispatch_critic.parameters(), trained['pricing_advantage'].dispatch_critic.parameters())))

    def test_constructor_rollouts_and_evaluation_preserve_global_random_generators(self):
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        trainer = self.trainer()
        trainer.collect_rollout()
        trainer.evaluate([701, 703], deterministic=False, action_seed=123)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_before, atol=0, rtol=0)

    def test_checkpoint_restores_deterministic_stochastic_and_optimizer_continuation(self):
        trainer = self.trainer()
        trainer.update(trainer.collect_rollout())
        deterministic = trainer.evaluate([711, 713], deterministic=True)
        stochastic = trainer.evaluate([711, 713], deterministic=False, action_seed=123)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'paper.pt'
            trainer.save_checkpoint(path)
            restored = self.paper.PaperPPOTrainer.load_checkpoint(path)
            self.assertEqual(without_timing(restored.evaluate([711, 713], deterministic=True)), without_timing(deterministic))
            self.assertEqual(without_timing(restored.evaluate([711, 713], deterministic=False, action_seed=123)), without_timing(stochastic))
            expected = trainer.collect_rollout()
            actual = restored.collect_rollout()
            for name in ('pricing_actions', 'dispatch_actions', 'pricing_rewards', 'dispatch_rewards',
                         'old_price_log_probs', 'old_dispatch_log_probs'):
                np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))
            trainer.update(expected)
            restored.update(actual)
            for name, module in self.modules(trainer).items():
                for expected_parameter, actual_parameter in zip(module.parameters(), self.modules(restored)[name].parameters()):
                    torch.testing.assert_close(expected_parameter, actual_parameter, atol=0, rtol=0)
            self.assertEqual(restored.update_count, 2)
            self.assertEqual(restored.episode_count, 4)
            self.assertEqual(restored.environment_periods, 12)

    def test_checkpoint_rejects_wrong_algorithm_missing_models_nonfinite_parameters_and_counters(self):
        trainer = self.trainer()
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'paper.pt'
            trainer.save_checkpoint(path)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            wrong_algorithm = deepcopy(payload)
            wrong_algorithm['algorithm'] = 'legacy_mappo'
            missing_model = deepcopy(payload)
            del missing_model['models']['dispatch_critic']
            nonfinite = deepcopy(payload)
            name = next(iter(nonfinite['models']['dispatch_actor']))
            nonfinite['models']['dispatch_actor'][name].reshape(-1)[0] = float('nan')
            bad_counter = deepcopy(payload)
            bad_counter['environment_periods'] = -1
            bad_target = deepcopy(payload)
            bad_target['training_config']['dispatch_critic_target'] = 'automatic'
            extra = deepcopy(payload)
            extra['unexpected'] = True
            for corrupted in (wrong_algorithm, missing_model, nonfinite, bad_counter, bad_target, extra):
                torch.save(corrupted, path)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.paper.PaperPPOTrainer.load_checkpoint(path)

    def test_distribution_shift_requires_explicit_evaluation_and_cannot_replace_physics(self):
        trainer = self.trainer()
        shifted = replace(self.scenario, mean_requests_by_zone=(6., 6.), initial_energy_kwh=(35., 35.)).validate()
        with self.assertRaises(ValueError):
            trainer.with_scenario(shifted)
        frozen = trainer.with_scenario(shifted, allow_distribution_shift=True)
        result = frozen.evaluate([721, 723], deterministic=False, action_seed=123)
        self.assertEqual(result['invalid_actions'], 0)
        self.assertEqual(trainer.scenario, self.scenario)
        for name, module in self.modules(trainer).items():
            for original, adapted in zip(module.parameters(), self.modules(frozen)[name].parameters()):
                torch.testing.assert_close(original, adapted, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            frozen.collect_rollout()
        changed_physics = replace(self.scenario, mobile_service_kw=40.).validate()
        with self.assertRaises(ValueError):
            trainer.with_scenario(changed_physics, allow_distribution_shift=True)

    def test_training_cli_preserves_existing_runs_and_resume_continues_exact_counters(self):
        from charging_env.rl.paper_train import main
        scenario = paper_scenario(horizon=1, arrival_profile=(1.,),
                                  grid_buy_per_kwh=(.12,), grid_sell_per_kwh=(.09,))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario_path = root / 'scenario.json'
            scenario.to_json(scenario_path)
            requested = root / 'run'
            small = ['--updates', '1', '--episodes-per-update', '1', '--epochs', '1',
                     '--minibatch-size', '2', '--hidden-size', '8', '--eval-episodes', '1',
                     '--eval-interval', '0', '--torch-threads', '1']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--scenario', str(scenario_path), '--output-dir', str(requested)] + small), 0)
            first = {path.name: path.read_bytes() for path in requested.iterdir()}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--resume', str(requested / 'checkpoint.pt'),
                                       '--output-dir', str(requested)] + small + ['--torch-threads', '2']), 0)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual({path.name: path.read_bytes() for path in requested.iterdir()}, first)
            resumed = self.paper.PaperPPOTrainer.load_checkpoint(root / 'run_2/checkpoint.pt')
            self.assertEqual(resumed.update_count, 2)
            self.assertEqual(resumed.episode_count, 2)
            self.assertEqual(resumed.environment_periods, 2)
            summary = json.loads((root / 'run_2/summary.json').read_text())
            self.assertTrue(summary['all_metrics_finite'])
            self.assertTrue(summary['checkpoint_reload_matches'])
            self.assertEqual(summary['invalid_evaluation_actions'], 0)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--resume', str(requested / 'checkpoint.pt'), '--dispatch-critic-target', 'pricing_advantage'])
            evaluation_path = root / 'evaluation'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--evaluate-only', str(root / 'run_2/checkpoint.pt'),
                                       '--output-dir', str(evaluation_path), '--eval-episodes', '1']), 0)
            self.assertTrue((evaluation_path / 'evaluation.json').is_file())

    def test_invalid_configuration_choices_and_empty_rollouts_fail_explicitly(self):
        for changes in ({'dispatch_critic_target': 'automatic'}, {'seed': True}, {'seed': -1},
                        {'episodes_per_update': 0}, {'epochs': 0}, {'hidden_size': 0},
                        {'actor_learning_rate': float('nan')}, {'clip_ratio': 0.}, {'gae_lambda': 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes).validate()
        trainer = self.trainer()
        rollout = trainer.collect_rollout()
        empty = deepcopy(rollout)
        for name, value in vars(empty).items():
            if isinstance(value, np.ndarray):
                setattr(empty, name, value[:0])
            elif isinstance(value, list):
                setattr(empty, name, [])
        with self.assertRaises(ValueError):
            trainer.update(empty)


if __name__ == '__main__':
    unittest.main()
