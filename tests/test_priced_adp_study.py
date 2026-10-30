"""Study-selection, provenance, accounting and safe-resume checks on tiny banks."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator
from charging_env.benchmarks.policies import coordinated_dispatch, dispatch_outcome
from charging_env.rl.adaptive_pricing import PriceSearchConfig, fit_price_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('priced_study_tests', ROOT/'experiments/run_priced_adp_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def small_scenario():
    return replace(Scenario(), horizon=2, arrival_profile=(1., 1.), grid_buy_per_kwh=(.15, .15),
        fcs_pricing_mode='legacy', mcv_pricing_mode='absolute',
                   grid_sell_per_kwh=(.1, .1), charging_utility_intercept=3.).validate()


def selection_fixture():
    scenario = Scenario.from_json(ROOT/study.CASES['small'])
    prices = {str(price): {str(seed): 0. for seed in study.SEEDS}
              for price in study.price_indices(scenario)}
    prices['0'] = {'7': 100., '19': -100., '37': -100.}
    prices['1'] = {str(seed): 10. for seed in study.SEEDS}
    learned = {backend: {str(seed): dict(warm=10., refined=9.) for seed in study.SEEDS}
               for backend in study.BACKENDS}
    inventory = dict(mean_discounted_profit=8., specification=dict(id='inventory',
        price_index=study.EXACT_PRICE_INDEX(scenario),
        early_shadow_price=.2, late_shadow_price=.1, family='inventory_rule', switch_period=6))
    return study.selection_from_scores(prices, learned, inventory)


class PricedADPStudyTests(unittest.TestCase):
    def test_equal_refinement_budgets_distinct_banks_and_warm_anchors(self):
        banks = []
        for case in study.CASES:
            scenario = Scenario.from_json(ROOT/study.CASES[case])
            self.assertEqual(len(study.rule_grid(scenario)),
                             scenario.n_price_levels*len(study.previous.SCHEDULES))
            self.assertEqual(study.price_indices(scenario), tuple(range(scenario.n_price_levels)))
            for seed in study.SEEDS:
                warm = study.training_config(case, 'warm', seed)
                self.assertEqual(warm.generations*warm.population*warm.episodes_per_generation, 4608)
                self.assertEqual(warm.anchors(scenario), tuple(range(scenario.n_price_levels)))
                for phase in study.PHASES:
                    config = study.training_config(case, phase, seed)
                    bank = sum(config.generation_seeds(), [])
                    self.assertTrue(all(s % 2 == 0 for s in bank))
                    self.assertEqual(bank[0], config.training_seed_base)
                    banks.append(set(bank))
                    if phase != 'warm':
                        self.assertEqual(config.generations*config.population*config.episodes_per_generation, 128)
                        self.assertEqual(config.elite_count, 3)
                        self.assertEqual(config.anchors(scenario), ())
        for i, bank in enumerate(banks):
            self.assertTrue(all(bank.isdisjoint(other) for other in banks[:i]))

    def test_fixed_price_selection_averages_every_seed_and_ties_preserve_fallback(self):
        selection = selection_fixture()
        self.assertEqual(selection['fixed_rollout']['price_index'], 1)
        for seed in study.SEEDS:
            self.assertEqual(selection['selected']['rollout'][str(seed)]['candidate'], 'fixed')
            self.assertEqual(selection['selected']['inventory'][str(seed)]['candidate'], 'warm')

    def test_final_rows_preserve_swap_actor_and_strong_small_reference(self):
        selection = selection_fixture()
        selection['selected']['rollout']['19']['candidate'] = 'refined'
        small = study.final_specs('small', selection); city = study.final_specs('city', selection)
        self.assertEqual((len(small), len(city)), (14, 13))
        self.assertEqual((len(small)+len(city))*256, 6912)
        swapped = city['rollout_price_inventory_19']
        self.assertEqual(swapped['backend'], 'inventory')
        self.assertEqual(swapped['phase'], 'rollout')
        self.assertEqual(swapped['seed'], 19)
        for name, row in city.items():
            if row['backend'] == 'inventory':
                self.assertEqual(row['setting'], selection['fixed_inventory']['specification'])
        self.assertEqual(small['exact_inventory']['setting']['energy_fractions'], [.5, 1.])
        self.assertEqual(small['exact_inventory']['price_index'],
                         study.EXACT_PRICE_INDEX(Scenario.from_json(ROOT/study.CASES['small'])))

    def test_splits_reject_historical_and_cross_case_reuse(self):
        p = dict(cases={case: dict(validation_seeds=study.VALIDATION[case], test_seeds=study.TEST[case])
                        for case in study.CASES})
        study.validate_splits(p)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            study.validate_splits(p, [study.TEST['small'][0]])
        changed = deepcopy(p); changed['cases']['city']['validation_seeds'] = study.TEST['small']
        with self.assertRaisesRegex(ValueError, 'disjoint'): study.validate_splits(changed)

    def test_historical_registry_reconstructs_implicit_training_and_query_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.write(root/'outputs/config.json', dict(seed=7, training_seed_base=100,
                updates=2, episodes_per_update=3, warmstart_seed_base=200, warmstart_episodes=2,
                eval_seed_base=1001, eval_episodes=2,
                plan=dict(query_seed_start=3000, query_seed_count=3, query_seed_stride=2),
                future_draw_seeds=[[10000, 10002]]))
            seeds, hashes = study.historical_seeds(root)
            self.assertTrue(set(range(214, 226, 2)) <= seeds)
            self.assertTrue({414, 416, 1001, 1003, 3000, 3002, 3004, 10000, 10002} <= seeds)
            self.assertEqual(set(hashes), {'outputs/config.json'})

    def test_exact_reference_matches_existing_solver_and_shared_physics(self):
        scenario = small_scenario(); sim = ChargingSimulator(scenario)
        pre, _ = sim.reset(seed=17)
        pricer = study.constant_pricer(scenario, 4)
        post, *_ = sim.quote_price_levels(pricer.prices(pre))
        backend = study.make_backend(scenario, 'exact', pricer, 7, study.EXACT)
        actual = backend.dispatch(post)
        expected = coordinated_dispatch(post, scenario, .3, energy_fractions=(.5, 1.))
        self.assertEqual(actual, expected)
        predicted = dispatch_outcome(post, scenario, actual)
        _, reward, _, _, info = sim.dispatch(actual)
        self.assertAlmostEqual(predicted['reward'], reward)
        self.assertFalse(info['invalid_actions'])
        self.assertGreater(backend.statistics['exhaustive_candidate_outcomes'], 0)

    def test_fixed_and_adaptive_backend_routing_is_explicit(self):
        c = small_scenario(); pricer = study.constant_pricer(c, 4)
        fixed = study.make_backend(c, 'rollout', pricer, 19, fixed_price=4)
        adaptive = study.make_backend(c, 'rollout', pricer, 19)
        self.assertIs(type(fixed), study.RolloutADPPolicy)
        self.assertIs(type(adaptive), study.PricedRolloutADPPolicy)
        self.assertEqual(fixed.config.rollout_samples, adaptive.config.rollout_samples)

    def test_saved_rollout_can_resume_without_new_scenes_and_detects_tampering(self):
        c = small_scenario()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'provenance').mkdir()
            history = root/'provenance/historical_seeds.npz'
            np.savez_compressed(history, seeds=np.asarray([], dtype=np.int64))
            p = dict(cases={'small': dict(scenario=study.normalized(asdict(c)),
                validation_seeds=[101], test_seeds=[103])}, historical_seed_sha256=study.digest(history))
            specification = dict(label='fixed', backend='rollout', pricing='fixed', price_index=4, seed=7)
            path = root/'result.json'
            with redirect_stdout(io.StringIO()):
                result = study.evaluate_saved(root, p, 'small', [17], path, specification)
            self.assertEqual(result['n_episodes'], 1)
            self.assertEqual(result['backend_statistics']['decisions'], c.horizon)
            with patch.object(study, 'evaluate_policy', side_effect=AssertionError('No second rollout')):
                self.assertEqual(study.evaluate_saved(root, p, 'small', [17], path, specification), result)
            provenance_path = path.with_suffix('.provenance.json')
            original = study.read(provenance_path); altered = deepcopy(original)
            altered['base_policy'] = 'unexecuted_backend'
            study.write(provenance_path, altered)
            altered_result = deepcopy(result)
            altered_result['evidence_sha256'][provenance_path.name] = study.digest(provenance_path)
            study.seal(path, altered_result)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                study.inspect_evaluation(root, p, 'small', [17], path, specification)
            study.write(provenance_path, original)
            changed = deepcopy(result); changed['mean_discounted_profit'] += 1.
            study.seal(path, changed)
            with self.assertRaisesRegex(ValueError, 'mean profit'):
                study.inspect_evaluation(root, p, 'small', [17], path, specification)

    def test_frozen_trace_detects_argmax_and_budget_tampering(self):
        from test_general_adp_study import trace_fixture
        c = small_scenario(); config = study.RolloutADPConfig(seed=7, rollout_samples=8)
        trace, counters = trace_fixture(c, config, [17])
        study.previous.check_planning_trace(trace, counters, c, config, [17])
        changed = deepcopy(trace); changed[0]['future_model_queries'] += 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            study.previous.check_planning_trace(changed, counters, c, config, [17])

    def test_training_inspection_checks_history_means_and_final_actor(self):
        c = small_scenario()
        config = PriceSearchConfig(seed=7, generations=2, population=3, elite_count=2,
            episodes_per_generation=1, training_seed_base=100, fixed_anchor_indices=())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); directory = study.train_path(root, 'small', 'warm', 7)
            def factory(pricer, seed): return study.RuleBackend(c, pricer, 'warm', {})
            with redirect_stdout(io.StringIO()):
                _, summary = fit_price_policy(c, config, factory, directory)
            p = dict(cases={'small': dict(scenario=study.normalized(asdict(c)))}, source_sha256=dict(summary['source_sha256']))
            p['source_sha256']['experiments/run_priced_adp_study.py'] = summary['backend_factory']['source_sha256']
            study.seal(directory.with_suffix('.binding.json'), {'bound': True})
            with patch.object(study, 'training_config', return_value=config), patch.object(study, 'training_binding', return_value={'bound': True}):
                model = study.inspect_training(root, p, 'small', 'warm', 7)
                self.assertEqual(model.scenario, c)
                history = study.read(directory/'training_history.json')
                history[0]['candidates'][0]['mean_discounted_profit'] += 1.
                study.write(directory/'training_history.json', history)
                summary['training_history_sha256'] = study.digest(directory/'training_history.json')
                study.write(directory/'summary.json', summary)
                with self.assertRaisesRegex(ValueError, 'accounting'):
                    study.inspect_training(root, p, 'small', 'warm', 7)

    def test_final_preflight_precedes_creating_final_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(study, 'verify', return_value={}), patch.object(study, 'prepare_final', side_effect=ValueError('missing training')):
                with self.assertRaisesRegex(ValueError, 'missing training'): study.evaluate(root)
            self.assertFalse((root/'test').exists())

    def test_output_guards_and_worker_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError): study.init(Path(folder), ROOT)
        for workers in (0, 4, True):
            with self.assertRaises(ValueError): study.run_jobs([], workers)
        study.run_jobs([], 1)

    def test_replay_copies_only_frozen_init_evidence_and_marks_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, target = root/'source', root/'target'
            study.seal(source/'protocol.json', dict(schema='fixture'))
            study.write(source/'source_snapshot/source.json', dict(code='fixed'))
            for name in ('historical_json_sha256.json', 'exact_inventory_origin.json'):
                study.write(source/'provenance'/name, {})
            np.savez_compressed(source/'provenance/historical_seeds.npz', seeds=np.array([17]))
            study.write(source/'test/result.json', {'profit': 99.})
            study.write(source/'training/result.json', {'weights': 1.})
            with patch.object(study, 'verify', return_value={}): study.replay_init(target, source)
            self.assertEqual(study.digest(target/'protocol.json'), study.digest(source/'protocol.json'))
            self.assertFalse((target/'test').exists())
            self.assertFalse((target/'training').exists())
            self.assertIs(study.sealed(target/'provenance/replay.json')['fresh_independent_study'], False)
            with patch.object(study, 'verify', side_effect=ValueError('source changed')):
                with self.assertRaisesRegex(ValueError, 'source changed'): study.replay_init(root/'bad', source)
            self.assertFalse((root/'bad').exists())

    def test_seed_roles_and_active_price_pooling_are_explicit(self):
        items = [dict(specification=dict(seed=seed), mean_mobile_price=.5,
            active_pricing=dict(counts=[0, 0, 0, 0, 2, 0, 0], by_period=[[0, 0, 0, 0, 2, 0, 0]], mean_active_price=.6))
            for seed in study.SEEDS]
        with patch.object(study.previous, 'aggregate_models', return_value={}):
            fixed = study.aggregate(items, 'fixed_rollout')
            self.assertEqual(fixed['seed_role'], 'planning_seed')
            self.assertEqual(fixed['active_pricing']['counts'][4], 6)
            self.assertAlmostEqual(fixed['active_pricing']['mean_active_price'], .6)
            self.assertEqual(study.aggregate(items, 'rollout_price_inventory')['seed_role'], 'source_pricing_training_seed')

    def test_reconstructed_training_queries_match_recorded_rng_protocol(self):
        from test_general_adp_study import trace_fixture
        c = small_scenario(); config = study.RolloutADPConfig(seed=19, rollout_samples=8)
        trace, stats = trace_fixture(c, config, [17, 19])
        expected = study.previous.check_planning_trace(trace, stats, c, config, [17, 19])
        self.assertEqual(set(study.planning_query_seeds(c.horizon, 19, (17, 19))), expected)


if __name__ == '__main__': unittest.main()
