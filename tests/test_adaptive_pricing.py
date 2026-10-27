"""Request-blind adaptive pricing and reproducible original-profit search."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator
from charging_env.benchmarks.policies import service_first_dispatch
from charging_env.rl.adaptive_pricing import (LearnedPricingPolicy, PriceSearchConfig,
    PARAMETER_COUNT, fit_price_policy, pricing_observation, pricing_features)


def scenario(vehicles=2, horizon=2, **changes):
    values = dict(horizon=horizon, arrival_profile=(1.,)*horizon,
        grid_buy_per_kwh=(.15,)*horizon, grid_sell_per_kwh=(.1,)*horizon,
        initial_locations=tuple(i % 4 for i in range(vehicles)),
        initial_energy_kwh=(55.,)*vehicles, charging_utility_intercept=3.)
    return replace(Scenario(), **(values | changes)).validate()


def constant_policy(c, index=4):
    coefficients = np.zeros(PARAMETER_COUNT); coefficients[0] = c.price_levels[index]
    return LearnedPricingPolicy(c, coefficients)


class ServiceFirstBackend:
    def __init__(self, pricer, planning_seed):
        self.pricer, self.scenario = pricer, pricer.scenario
        self.statistics = dict(decisions=0)

    def reset_episode(self, episode_key, action_seed=None):
        pass

    def prices(self, obs, rng=None):
        return self.pricer.prices(obs)

    def dispatch(self, obs, rng=None):
        self.statistics['decisions'] += 1
        return service_first_dispatch(obs, self.scenario, .2)


class AdaptivePricingTests(unittest.TestCase):
    def test_projection_matches_paper_and_poisoned_hidden_inputs_do_not_change_prices(self):
        from charging_env.rl.paper_ppo import paper_pricing_observation
        c = scenario(3); sim = ChargingSimulator(c); obs, _ = sim.reset(seed=17)
        coefficients = np.linspace(-.07, .09, PARAMETER_COUNT); coefficients[0] = .6
        policy = LearnedPricingPolicy(c, coefficients)
        expected, actual = paper_pricing_observation(obs, c), pricing_observation(obs, c)
        self.assertEqual(set(expected), set(actual))
        for key in expected: np.testing.assert_array_equal(expected[key], actual[key])
        changed = deepcopy(obs)
        for key in ('requests_by_zone', 'potential_energy_kwh', 'request_details', 'mobile_demand',
                'fixed_demand', 'mobile_requested_kwh', 'fixed_requested_kwh', 'outside_requests',
                'mobile_wait_hours', 'mobile_available', 'eligible_mcvs', 'fcs_wait_hours',
                'mobile_prices', 'pricing_mask', 'action_mask'):
            changed[key] = np.full(np.shape(changed[key]), 999., dtype=float)
        np.testing.assert_array_equal(pricing_features(obs, c), pricing_features(changed, c))
        np.testing.assert_array_equal(policy.prices(obs), policy.prices(changed))
        np.testing.assert_array_equal(policy.prices(obs), policy.prices(actual))

    def test_features_are_bounded_shared_and_vehicle_permutation_invariant(self):
        for count in (1, 2, 3, 10):
            c = scenario(count); sim = ChargingSimulator(c); obs, _ = sim.reset(seed=29)
            features = pricing_features(obs, c)
            self.assertEqual(features.shape, (c.n_zones, PARAMETER_COUNT))
            self.assertTrue(np.isfinite(features).all() and (np.abs(features) <= 1).all())
            swapped = deepcopy(obs)
            swapped['locations'] = obs['locations'][::-1]
            swapped['energy_kwh'] = obs['energy_kwh'][::-1]
            np.testing.assert_allclose(features, pricing_features(swapped, c), atol=1e-15)

    def test_adaptive_prices_change_with_visible_state_and_respect_legal_levels(self):
        c = scenario(3, price_levels=(.2, .45, .9, 1.4), price_max=1.4)
        sim = ChargingSimulator(c); obs, _ = sim.reset(seed=17)
        coefficients = np.zeros(PARAMETER_COUNT); coefficients[0] = .8; coefficients[1] = .5
        policy = LearnedPricingPolicy(c, coefficients)
        early = policy.prices(obs)
        late = deepcopy(obs); late['period'][0] = 1
        self.assertTrue(np.any(early != policy.prices(late)))
        empty = deepcopy(obs); empty['energy_kwh'][:] = c.reserve_kwh
        np.testing.assert_array_equal(policy.prices(empty), np.zeros(c.n_zones, dtype=int))
        self.assertTrue(np.all((early >= 0) & (early < len(c.price_levels))))

    def test_readonly_coefficients_and_scenario_copies_protect_actor(self):
        c = scenario(); coefficients = np.zeros(PARAMETER_COUNT); coefficients[0] = .6
        policy = LearnedPricingPolicy(c, coefficients)
        coefficients[:] = 999
        self.assertEqual(policy.coefficients[0], .6)
        with self.assertRaises(ValueError): policy.coefficients[0] = 999
        extracted = policy.coefficients; extracted.setflags(write=True); extracted[:] = 999
        self.assertEqual(policy.coefficients[0], .6)
        copy = policy.scenario; copy.distances[:] = 999
        self.assertFalse(np.all(policy.scenario.distances == 999))
        metadata = policy.provenance(); metadata['scenario']['reserve_kwh'] = 999
        self.assertEqual(policy.scenario, c)
        self.assertEqual(policy.provenance()['scenario']['reserve_kwh'], c.reserve_kwh)

    def test_checkpoint_roundtrip_hashes_and_no_overwrite(self):
        c = scenario(); policy = constant_policy(c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'; policy.save_checkpoint(path)
            loaded = LearnedPricingPolicy.load_checkpoint(path)
            self.assertEqual(policy.provenance()['policy_sha256'], loaded.provenance()['policy_sha256'])
            self.assertIsNone(policy.provenance()['checkpoint_sha256'])
            self.assertIsNotNone(loaded.provenance()['checkpoint_sha256'])
            self.assertEqual(loaded.scenario, c)
            np.testing.assert_array_equal(loaded.coefficients, policy.coefficients)
            with self.assertRaises(FileExistsError): policy.save_checkpoint(path)
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(str(saved['metadata'])); meta['coefficients'][0] += 1
            altered = Path(directory)/'altered.npz'
            np.savez(altered, coefficients=policy.coefficients, metadata=np.asarray(json.dumps(meta)))
            with self.assertRaisesRegex(ValueError, 'identity'): LearnedPricingPolicy.load_checkpoint(altered)

    def test_checkpoint_rejects_large_headers_before_array_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'hostile.npz'
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header,
                dict(descr='<f8', fortran_order=False, shape=(10**12,)))
            metadata = io.BytesIO(); np.save(metadata, np.asarray('{}'), allow_pickle=False)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('coefficients.npy', header.getvalue())
                archive.writestr('metadata.npy', metadata.getvalue())
            with self.assertRaisesRegex(ValueError, 'header'): LearnedPricingPolicy.load_checkpoint(path)

    def test_search_is_deterministic_paired_and_scores_original_business_profit(self):
        c = scenario(1, 1)
        config = PriceSearchConfig(seed=19, generations=2, population=5, elite_count=2,
            episodes_per_generation=2, training_seed_base=800000000, fixed_anchor_indices=(3,))
        global_before = np.random.get_state()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            policy, summary = fit_price_policy(c, config, ServiceFirstBackend, Path(directory)/'first')
            other, second = fit_price_policy(c, config, ServiceFirstBackend, Path(directory)/'second')
            np.testing.assert_array_equal(policy.coefficients, other.coefficients)
            self.assertEqual(summary['generation_seeds'], [[800000000, 800000002], [800000004, 800000006]])
            self.assertEqual(summary['candidate_evaluations'], 10)
            self.assertEqual(summary['environment_episodes'], 20)
            self.assertEqual(summary['environment_periods'], 20)
            self.assertEqual(summary['backend_counters']['decisions'], 20)
            self.assertEqual(summary['final_mean_discounted_profit'], second['final_mean_discounted_profit'])
            history = json.loads((Path(directory)/'first/training_history.json').read_text())
            for generation in history:
                self.assertEqual(generation['candidates'][0]['kind'], 'incumbent')
                for row in generation['candidates']:
                    actor = LearnedPricingPolicy(c, row['coefficients'])
                    returns = []
                    for seed in generation['environment_seeds']:
                        sim = ChargingSimulator(c); obs, _ = sim.reset(seed=seed)
                        post, _, _, _, _ = sim.quote_price_levels(actor.prices(obs))
                        _, reward, _, _, _ = sim.dispatch(service_first_dispatch(post, c, .2))
                        returns.append(reward)
                    np.testing.assert_array_equal(returns, row['episode_discounted_profits'])
                    self.assertEqual(row['mean_discounted_profit'], np.mean(returns))
                winner = max(range(len(generation['candidates'])),
                    key=lambda i: generation['candidates'][i]['mean_discounted_profit'])
                self.assertEqual(generation['winner_index'], winner)
            with self.assertRaises(FileExistsError):
                fit_price_policy(c, config, ServiceFirstBackend, Path(directory)/'first')
        after = np.random.get_state()
        for before_item, after_item in zip(global_before, after):
            np.testing.assert_equal(before_item, after_item)

    def test_config_guards_budgets_and_refinement_keeps_random_proposals(self):
        c = scenario()
        config = PriceSearchConfig(generations=1, population=8, elite_count=3,
            episodes_per_generation=1, fixed_anchor_indices=())
        self.assertEqual(config.validate(c), config)
        for changed in (replace(config, training_seed_base=1), replace(config, seed=True),
                replace(config, minimum_std=.3), replace(config, population=1),
                replace(config, generations=100000), replace(config, fixed_anchor_indices=(0, 0)),
                replace(config, fixed_anchor_indices=(999,))):
            with self.assertRaises(ValueError): changed.validate(c)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            _, _ = fit_price_policy(c, config, ServiceFirstBackend, Path(directory)/'refine', constant_policy(c))
            history = json.loads((Path(directory)/'refine/training_history.json').read_text())
            self.assertEqual(len(history[0]['candidates']), 8)
            self.assertTrue(any('sample' in row['kind'] for row in history[0]['candidates']))

    def test_nonfinite_or_invalid_actions_fail_search(self):
        c = scenario(); config = PriceSearchConfig(generations=1, population=2, elite_count=1,
            episodes_per_generation=1, fixed_anchor_indices=())
        fake = dict(episodes=[dict(discounted_profit=1.)], invalid_actions=1)
        with tempfile.TemporaryDirectory() as directory, patch(
                'charging_env.rl.adaptive_pricing.evaluate_policy', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'invalid actions'):
                fit_price_policy(c, config, ServiceFirstBackend, Path(directory)/'invalid')
        bad = np.zeros(PARAMETER_COUNT); bad[0] = np.nan
        with self.assertRaises(ValueError): LearnedPricingPolicy(c, bad)


if __name__ == '__main__': unittest.main()
