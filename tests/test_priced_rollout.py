"""Learned-price rollout information boundaries, physical replay, and provenance."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks.policies import dispatch_outcome, service_first_dispatch
from charging_env.config import Scenario
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PARAMETER_COUNT, pricing_observation
from charging_env.rl.fitted_value import sample_dispatch_observation
from charging_env.rl.priced_rollout import (
    PricedRolloutADPPolicy, main, sample_priced_dispatch_observation,
)
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy, evaluate_rollout_policy
from charging_env.simulator import ChargingSimulator


ALLOWED = {'period', 'locations', 'energy_kwh', 'arrival_rates', 'fcs_retail_prices',
           'fcs_discharge_prices', 'grid_buy_price', 'stage', 'pricing_mask'}


def scenario(vehicles=3, horizon=3, **changes):
    settings = dict(horizon=horizon, zone_xy_km=((0., 0.), (2., 0.), (4., 0.)),
        fcs_zones=(0,), initial_locations=tuple(i % 3 for i in range(vehicles)),
        initial_energy_kwh=tuple((20., 35., 55.)[i % 3] for i in range(vehicles)),
        mean_requests_by_zone=(2., 3., 2.), arrival_profile=(1.,)*horizon,
        grid_buy_per_kwh=(.15,)*horizon, grid_sell_per_kwh=(.1,)*horizon,
        fcs_retail_per_kwh=(.5,), fcs_ev_capacity_kwh=(25.,),
        fcs_fleet_exchange_capacity_kwh=(15.,), price_noise_distribution='normal',
        price_noise_scale=.15, charging_utility_intercept=3.)
    return replace(Scenario(), **(settings | changes)).validate()


class FixedActor:
    def __init__(self, config, index=4):
        self.scenario, self.index = config, index

    def prices(self, obs, rng=None):
        if set(obs) != ALLOWED or rng is not None:
            raise AssertionError('The actor must receive only permitted information and no RNG')
        return np.full(self.scenario.n_zones, self.index, dtype=np.int64)

    def provenance(self):
        return dict(schema='test.fixed_actor', index=self.index)


class VaryingActor(FixedActor):
    def prices(self, obs, rng=None):
        super().prices(obs, rng)
        return ((np.arange(self.scenario.n_zones)+int(obs['period'][0])
                 +int(np.mean(obs['energy_kwh']) < 30.)) % len(self.scenario.price_levels)).astype(np.int64)


class InvalidActor(FixedActor):
    def __init__(self, config, values):
        super().__init__(config)
        self.values = values

    def prices(self, obs, rng=None):
        return self.values


def constant_learned_actor(config, index=5):
    coefficients = np.zeros(PARAMETER_COUNT)
    coefficients[0] = config.price_levels[index]
    return LearnedPricingPolicy(config, coefficients)


def real_future(config, period, locations, energy, seed, actor):
    """Independent replay through original simulator quote AND dispatch methods."""
    simulator = ChargingSimulator(config)
    simulator.period, simulator.stage = period, 0
    simulator.locations, simulator.energy = np.asarray(locations).copy(), np.asarray(energy).copy()
    simulator.cumulative_profit = simulator.cumulative_discounted_profit = 0.
    simulator.rng = np.random.default_rng(seed)
    simulator._prepare_period()
    simulator._ready = True
    pre = pricing_observation(simulator.observe(), config)
    levels = actor.prices(pre)
    active = pre['pricing_mask'].any(axis=1)
    levels = np.where(active, levels, 0).astype(np.int64)
    post, reward, done, truncated, _ = simulator.quote_price_levels(levels)
    assert reward == 0. and not done and not truncated
    return simulator, post, levels, active


class PricedRolloutTests(unittest.TestCase):
    def test_public_sampler_matches_fixed_sampler_for_all_observation_fields(self):
        for m in (1, 3, 10):
            with self.subTest(vehicles=m):
                c = scenario(m, price_levels=(.2, .65, .9))
                actor = FixedActor(c, 1)
                for t in (0, 2):
                    old = sample_dispatch_observation(c, t, c.initial_locations, c.initial_energy_kwh, 140+t, 1)
                    new = sample_priced_dispatch_observation(c, t, c.initial_locations, c.initial_energy_kwh, 140+t, actor)
                    self.assertEqual(set(old), set(new))
                    for key in old:
                        np.testing.assert_array_equal(new[key], old[key], err_msg=key)

    def test_constant_actor_reproduces_full_legacy_episodes_and_query_scores(self):
        c, cfg = scenario(), RolloutADPConfig(seed=37, rollout_samples=2, price_index=5)
        old = RolloutADPPolicy(c, cfg)
        new = PricedRolloutADPPolicy(c, constant_learned_actor(c), cfg)
        a = evaluate_rollout_policy(old, [101, 103])
        b = evaluate_rollout_policy(new, [101, 103])
        self.assertEqual(a['action_counts'], b['action_counts'])
        self.assertEqual([e['discounted_profit'] for e in a['episodes']],
                         [e['discounted_profit'] for e in b['episodes']])
        for left, right in zip(old.decisions, new.decisions):
            for key in left:
                if key != 'seconds': self.assertEqual(left[key], right[key], key)
        for key in old.statistics: self.assertEqual(old.statistics[key], new.statistics[key], key)
        self.assertEqual(new.statistics['current_pricing_calls'], 6)
        self.assertEqual(new.statistics['future_pricing_calls'], new.statistics['future_model_queries'])
        self.assertEqual(len(new.pricing_decisions), 6)

    def test_current_and_future_actor_inputs_hide_latent_requests_and_masks(self):
        c = scenario(price_noise_distribution='none')
        actor = VaryingActor(c)
        policy = PricedRolloutADPPolicy(c, actor, RolloutADPConfig(rollout_samples=2))
        sim = ChargingSimulator(c)
        pre = sim.reset(seed=11)[0]
        changed = deepcopy(pre)
        for key in set(changed)-ALLOWED:
            if isinstance(changed[key], np.ndarray): changed[key] = np.full(changed[key].shape, 999.)
        changed['pricing_mask'] = np.zeros_like(changed['pricing_mask'])
        np.testing.assert_array_equal(policy.prices(pre), policy.prices(changed))
        post = sim.quote_price_levels(policy.prices(pre))[0]
        with patch.object(policy.pricing_policy, 'prices', wraps=policy.pricing_policy.prices) as calls:
            policy.dispatch(post)
        self.assertEqual(calls.call_count, policy.statistics['future_model_queries'])
        for call in calls.call_args_list:
            self.assertEqual(set(call.args[0]), ALLOWED)
            self.assertEqual(call.kwargs, {'rng': None})
            self.assertGreater(int(call.args[0]['period'][0]), 0)
        self.assertEqual(policy.statistics['current_pricing_calls'], 3)

    def test_nonuniform_future_prices_and_scores_replay_through_actual_simulator(self):
        c = scenario(3, 4, terminal_value_per_stored_kwh=.08)
        actor = VaryingActor(c)
        policy = PricedRolloutADPPolicy(c, actor, RolloutADPConfig(seed=19, rollout_samples=2))
        live = ChargingSimulator(c)
        pre = live.reset(seed=101)[0]
        post = live.quote_price_levels(policy.prices(pre))[0]
        names, candidates, _ = policy._candidate_plans(post)
        selected = policy.dispatch(post)
        record = policy.decisions[-1]
        seeds = np.asarray(record['future_draw_seeds'])
        values, digest, histogram = [], hashlib.sha256(), np.zeros(len(c.price_levels), dtype=np.int64)
        first = None
        for commands in candidates:
            current_sim = deepcopy(live)
            current_obs, current_reward, _, _, info = current_sim.dispatch(commands)
            self.assertEqual(info['invalid_actions'], [])
            current = dispatch_outcome(post, c, commands)
            self.assertAlmostEqual(current['reward'], current_reward, places=11)
            draws = []
            for k in range(2):
                total, discount = current_reward, c.discount_per_period
                locations, energy = current_obs['locations'], current_obs['energy_kwh']
                for offset, t in enumerate(range(1, c.horizon)):
                    simulated, observed, levels, active = real_future(c, t, locations,
                        np.clip(energy, c.reserve_kwh, c.battery_capacity_kwh), int(seeds[offset,k]), actor)
                    digest.update(levels.astype('<i8').tobytes())
                    histogram += np.bincount(levels[active], minlength=len(c.price_levels))
                    if first is None: first = levels.tolist()
                    action = service_first_dispatch(observed, c, .2)
                    observed, reward, terminal, _, info = simulated.dispatch(action)
                    self.assertEqual(info['invalid_actions'], [])
                    self.assertEqual(terminal, t == c.horizon-1)
                    total += discount*reward
                    discount *= c.discount_per_period
                    locations, energy = observed['locations'], observed['energy_kwh']
                draws.append(total)
            values.append(draws)
        np.testing.assert_allclose(record['sampled_total_scores'], values, rtol=0, atol=1e-11)
        self.assertEqual(record['future_price_levels_sha256'], digest.hexdigest())
        self.assertEqual(record['future_active_price_counts'], histogram.tolist())
        self.assertEqual(record['first_future_price_levels'], first)
        self.assertGreater(np.count_nonzero(histogram), 1)
        self.assertEqual(selected, candidates[int(np.argmax(np.mean(values, axis=1)))])
        self.assertEqual(names, record['candidates'])

    def test_terminal_uses_no_future_prices_and_keeps_salvage_and_negative_profit(self):
        c = scenario(2, 1, initial_locations=(1, 1), initial_energy_kwh=(5., 5.),
            mean_requests_by_zone=(0., 0., 0.), operating_cost_per_mcv_period=2.,
            terminal_value_per_stored_kwh=.1)
        policy = PricedRolloutADPPolicy(c, FixedActor(c), RolloutADPConfig(rollout_samples=2))
        sim = ChargingSimulator(c)
        pre = sim.reset(seed=5)[0]
        post = sim.quote_price_levels(policy.prices(pre))[0]
        with patch('charging_env.rl.priced_rollout._sample_priced_dispatch_observation',
                   side_effect=AssertionError('Terminal dispatch cannot sample future prices')):
            selected = policy.dispatch(post)
        _, reward, terminal, _, _ = sim.dispatch(selected)
        self.assertTrue(terminal)
        self.assertEqual(reward, -3.)
        row = policy.decisions[-1]
        self.assertEqual(row['sampled_total_scores'], [[-3., -3.]])
        self.assertEqual(row['future_pricing_calls'], 0)
        self.assertEqual(row['future_price_levels_sha256'], hashlib.sha256(b'').hexdigest())
        self.assertIsNone(row['first_future_price_levels'])
        self.assertEqual(row['future_active_price_counts'], [0]*len(c.price_levels))
        self.assertEqual(policy.pricing_decisions[0]['mobile_available'], [False]*c.n_zones)

    def test_external_actor_changes_do_not_change_the_frozen_controller(self):
        c = scenario()
        actor = FixedActor(c, 5)
        policy = PricedRolloutADPPolicy(c, actor)
        actor.index = 0
        pre = ChargingSimulator(c).reset(seed=3)[0]
        np.testing.assert_array_equal(policy.prices(pre), [5,5,5])
        self.assertEqual(policy.provenance()['pricing']['index'], 5)

    def test_live_rng_global_rng_and_observations_are_unchanged(self):
        c = scenario()
        live = ChargingSimulator(c)
        policy = PricedRolloutADPPolicy(c, VaryingActor(c), RolloutADPConfig(rollout_samples=2))
        pre = live.reset(seed=17)[0]
        external = np.random.default_rng(19)
        env_before = deepcopy(live.rng.bit_generator.state)
        external_before = deepcopy(external.bit_generator.state)
        pre_before = deepcopy(pre)
        numpy_before, python_before = np.random.get_state(), random.getstate()
        prices = policy.prices(pre, external)
        self.assertEqual(live.rng.bit_generator.state, env_before)
        post = live.quote_price_levels(prices)[0]
        env_before, post_before = deepcopy(live.rng.bit_generator.state), deepcopy(post)
        policy.dispatch(post, external)
        self.assertEqual(live.rng.bit_generator.state, env_before)
        self.assertEqual(external.bit_generator.state, external_before)
        for key in pre: np.testing.assert_array_equal(pre[key], pre_before[key])
        for key in post: np.testing.assert_array_equal(post[key], post_before[key])
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_reset_restores_query_and_price_replay_independent_of_episode_order(self):
        c = scenario(2, 2)
        cfg = RolloutADPConfig(seed=19, rollout_samples=2)
        left = PricedRolloutADPPolicy(c, VaryingActor(c), cfg)
        right = PricedRolloutADPPolicy(c, VaryingActor(c), cfg)
        a, b = evaluate_rollout_policy(left, [17,23]), evaluate_rollout_policy(right, [23,17])
        self.assertEqual({e['seed']:e['discounted_profit'] for e in a['episodes']},
                         {e['seed']:e['discounted_profit'] for e in b['episodes']})
        def records(policy):
            return {(r['episode_key'],r['period']): {k:v for k,v in r.items() if k!='seconds'}
                    for r in policy.decisions}
        self.assertEqual(records(left), records(right))
        self.assertEqual({(r['episode_key'],r['period']):r for r in left.pricing_decisions},
                         {(r['episode_key'],r['period']):r for r in right.pricing_decisions})

    def test_invalid_actor_actions_and_synthetic_states_are_rejected(self):
        c = scenario()
        pre = ChargingSimulator(c).reset(seed=7)[0]
        for values in ([0,1], [0.,1.,2.], [True,False,True], [-1,0,0], [99,0,0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PricedRolloutADPPolicy(c, InvalidActor(c, values)).prices(pre)
        with self.assertRaises(ValueError):
            PricedRolloutADPPolicy(c, FixedActor(replace(c, terminal_value_per_stored_kwh=.1)))
        actor = FixedActor(c)
        cases = [(True, c.initial_locations, c.initial_energy_kwh, 1),
                 (c.horizon, c.initial_locations, c.initial_energy_kwh, 1),
                 (0, c.initial_locations, c.initial_energy_kwh, -1),
                 (0, (0.,1.,2.), c.initial_energy_kwh, 1),
                 (0, (0,1,99), c.initial_energy_kwh, 1),
                 (0, c.initial_locations, (5.,5.,np.nan), 1),
                 (0, c.initial_locations, (4.,5.,5.), 1)]
        with patch('numpy.random.default_rng', side_effect=AssertionError('Reject before RNG construction')):
            for t,l,e,seed in cases:
                with self.subTest(state=(t,l,e,seed)), self.assertRaises(ValueError):
                    sample_priced_dispatch_observation(c,t,l,e,seed,actor)

    def test_controller_does_not_reconstruct_actor_scenario_during_future_queries(self):
        class CountedActor(FixedActor):
            def __init__(self, config): self._scenario, self.index = config, 4
            @property
            def scenario(self):
                if getattr(self, 'forbid_scenario_read', False):
                    raise AssertionError('Repeated actor scenario reconstruction')
                return self._scenario
            def prices(self, obs, rng=None):
                if set(obs) != ALLOWED: raise AssertionError('Information leak')
                return np.full(self._scenario.n_zones, self.index, dtype=np.int64)
        c = scenario(2,2)
        policy = PricedRolloutADPPolicy(c, CountedActor(c), RolloutADPConfig(rollout_samples=1))
        policy.pricing_policy.forbid_scenario_read = True
        live = ChargingSimulator(c)
        pre = live.reset(seed=7)[0]
        post = live.quote_price_levels(policy.prices(pre))[0]
        policy.dispatch(post)
        self.assertGreater(policy.statistics['future_pricing_calls'], 0)

    def test_cli_keeps_pricing_checkpoint_and_bounded_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = scenario(2,2)
            actor = constant_learned_actor(c)
            checkpoint = root/'pricing.npz'
            actor.save_checkpoint(checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            arguments = ['--pricing-checkpoint', str(checkpoint), '--episodes','1',
                '--eval-seed-base','831000001','--rollout-samples','1','--output-dir',str(root/'run')]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments),0)
                self.assertEqual(main(arguments),0)
            for name in ('run','run_2'):
                protocol = json.loads((root/name/'protocol.json').read_text())
                summary = json.loads((root/name/'summary.json').read_text())
                self.assertEqual(protocol['pricing_checkpoint_sha256'],digest)
                self.assertEqual(summary['invalid_actions'],0)
                self.assertEqual(len(json.loads((root/name/'pricing_trace.json').read_text())),2)
                for row in json.loads((root/name/'planning_trace.json').read_text()):
                    self.assertIn('future_price_levels_sha256',row)
                    self.assertNotIn('future_price_levels',row)
                self.assertIn('src/charging_env/rl/priced_rollout.py',protocol['source_sha256'])
                self.assertFalse(protocol['learned_value_bootstrap'])
                self.assertIsNone(protocol['fixed_price_index'])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)


if __name__ == '__main__':
    unittest.main()
