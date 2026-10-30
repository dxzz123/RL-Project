"""Declared history-study splits, selection, provenance and safe output checks.

These tests do not collect historical data or run the reserved private scenes.
Selection tests use small, explicit synthetic result records.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.config import Scenario
from charging_env.rl.history_control import model_scenario_sha256


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('history_study_tests',
    ROOT / 'experiments/run_history_adp_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def city():
    return Scenario.from_json(ROOT / study.SCENARIO_FILE)


def rule():
    return dict(id='inventory_p5_e0.2_l0.1', family='inventory_rule', price_index=5,
        early_shadow_price=.2, late_shadow_price=.1, switch_period=6)


def model_stub(root):
    path = study.model_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'explicit unit-test model identity')
    return SimpleNamespace(scenario=city())


def result_fixture(seeds, value=10.):
    terms = dict(mobile_service_revenue=value, fixed_service_revenue=0.,
        discharge_revenue=0., terminal_inventory_value=0., procurement_cost=0.,
        travel_cost=0., operating_cost=0., unmet_demand_cost=0.)
    return dict(evaluation_seeds=list(seeds), n_episodes=len(seeds), invalid_actions=0,
        action_seed=study.ACTION_SEED, mean_discounted_profit=value,
        episodes=[dict(seed=s, discounted_profit=value, discounted_reward_terms=terms.copy())
            for s in seeds])


class HistoryStudyTests(unittest.TestCase):
    def test_declared_banks_are_disjoint_and_ppo_uses_doubled_seed_mapping(self):
        banks = study.environment_banks()
        self.assertEqual(sum(map(len, banks.values())), 26056)
        self.assertEqual(len(set.union(*banks.values())), 26056)
        self.assertEqual(len(banks['history_fit']), 120)
        self.assertEqual(len(banks['history_holdout']), 32)
        for ordinal, seed in enumerate(study.SEEDS):
            self.assertEqual(min(banks[f'ppo/{seed}']), 2*(1_700_000_000+ordinal*100_000+seed))
            self.assertEqual(len(banks[f'ppo/{seed}']), 8000)
            self.assertEqual(len(banks[f'demonstration/{seed}']), 256)
            self.assertTrue(all(s % 2 == 0 for s in banks[f'ppo/{seed}']))
        self.assertEqual(sum(map(len, study.TEST.values())) * 10, 3840)
        with patch.object(study, 'MONITOR', (study.FIT_SEEDS[0],)):
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                study.environment_banks()

    def test_equal_price_refinement_budgets_and_strong_ppo_setup(self):
        scenario = city()
        for seed in study.SEEDS:
            warm = study.price_config('warm', seed)
            self.assertEqual(warm.generations*warm.population*warm.episodes_per_generation, 4608)
            self.assertEqual(warm.anchors(scenario), tuple(range(scenario.n_price_levels)))
            for backend in ('rollout', 'inventory'):
                c = study.price_config(backend, seed)
                self.assertEqual(c.generations*c.population*c.episodes_per_generation, 128)
                self.assertEqual(c.anchors(scenario), ())
                self.assertEqual(c.planning_seed, seed)
                self.assertEqual(c.generation_seeds()[0][0], c.training_seed_base)
            ppo = study.ppo_config(seed)
            self.assertEqual((ppo.updates, ppo.episodes_per_update), (500, 16))
            self.assertEqual((ppo.warmstart_episodes, ppo.warmstart_epochs), (256, 100))
            self.assertEqual(ppo.warmstart_dispatch_solver, 'coordinate')
            self.assertEqual(ppo.eval_policy, 'stochastic')
            self.assertEqual((ppo.eval_interval, ppo.eval_episodes), (100, 32))
            self.assertTrue(ppo.normalize_advantages)
            self.assertEqual(ppo.reward_scale, 20.)
        for phase, seed in [('unknown', 7), ('warm', 42)]:
            with self.assertRaises(ValueError): study.price_config(phase, seed)
        with self.assertRaises(ValueError): study.ppo_config(42)

    def test_private_shifts_change_only_arrivals_and_preserve_declared_totals(self):
        base = city(); original = deepcopy(asdict(base))
        shifts = study.shifted_scenarios(base)
        self.assertEqual(tuple(shifts), study.REGIMES)
        self.assertIs(shifts['baseline'], base)
        np.testing.assert_allclose(shifts['volume_up'].resolved_arrival_rates,
            1.4*base.resolved_arrival_rates, rtol=0., atol=0.)
        expected = np.roll(base.resolved_arrival_rates, 3, axis=0)[:, ::-1]
        np.testing.assert_array_equal(shifts['spatiotemporal'].resolved_arrival_rates, expected)
        self.assertAlmostEqual(expected.sum(), base.resolved_arrival_rates.sum())
        for shifted in shifts.values():
            actual = asdict(shifted)
            self.assertEqual({k:v for k,v in actual.items() if k != 'arrival_rates'},
                {k:v for k,v in original.items() if k != 'arrival_rates'})
        self.assertEqual(asdict(base), original)
        bad_points = list(base.zone_xy_km); bad_points[0] = (-.17, -.29)
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            study.shifted_scenarios(replace(base, zone_xy_km=tuple(bad_points)))

    def _selection(self, root, backend, seed, scores):
        model = model_stub(root)
        seen = []
        def evaluate(directory, truth, fitted, seeds, path, specification):
            self.assertEqual(truth, model.scenario)
            self.assertEqual(fitted, model.scenario)
            self.assertEqual(tuple(seeds), study.POLICY_VALIDATION)
            seen.append(deepcopy(specification))
            row = result_fixture(seeds, scores[specification['kind']])
            study.seal(path, row)
            return row
        with ExitStack() as stack:
            stack.enter_context(patch.object(study, 'verify', return_value={}))
            stack.enter_context(patch.object(study, 'fitted_model', return_value=model))
            stack.enter_context(patch.object(study, 'rule_setting', return_value=rule()))
            stack.enter_context(patch.object(study, 'check_training', return_value={}))
            stack.enter_context(patch.object(study, 'evaluation', side_effect=evaluate))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = study.selection_job((str(root), backend, seed))
        return result, seen

    def test_checkpoint_selection_ties_favor_final_and_never_private_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chosen, seen = self._selection(root, 'ppo', 7, dict(final=10., best=10., warmstart=10.))
            self.assertEqual([row['kind'] for row in seen], ['final', 'best', 'warmstart'])
            self.assertEqual(chosen['winner']['specification']['kind'], 'final')
            self.assertEqual(len(chosen['candidates']), 3)
            self.assertEqual(chosen, study.sealed(root/'selection/ppo_7.json'))

    def test_price_selection_has_fixed_fallback_and_uses_each_seed_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, _ = self._selection(root, 'rollout', 7, dict(fixed=10., warm=10., refined=9.))
            second, seen = self._selection(root, 'rollout', 19, dict(fixed=10., warm=11., refined=12.))
            self.assertEqual(first['winner']['specification']['kind'], 'fixed')
            self.assertEqual(second['winner']['specification']['kind'], 'refined')
            self.assertTrue(all(row['seed'] == 19 for row in seen))
            self.assertTrue(all(row['setting'] == rule() for row in seen))
            with self.assertRaisesRegex(ValueError, 'selection changed'):
                self._selection(root, 'rollout', 7, dict(fixed=10., warm=11., refined=12.))

    def test_final_contains_all_three_seeds_and_one_fixed_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend in ('rollout', 'ppo', 'inventory'):
                for seed in study.SEEDS:
                    kinds = study.PPO_CANDIDATES if backend == 'ppo' else study.PRICE_CANDIDATES
                    self._selection(root, backend, seed, dict(zip(kinds, (10., 9., 8.))))
            with patch.object(study, 'rule_setting', return_value=rule()):
                specs = study.final_specs(root)
            self.assertEqual(len(specs), 10)
            for backend in ('rollout', 'ppo', 'inventory'):
                self.assertEqual({specs[f'{backend}_{s}']['seed'] for s in study.SEEDS}, {7, 19, 37})
            self.assertEqual(specs['fixed_inventory']['setting'], rule())
            self.assertEqual(specs['fixed_inventory']['kind'], 'fixed')

    def test_incomplete_preflight_cannot_start_private_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(study, 'verify', return_value={}), \
                    patch.object(study, 'check_training', side_effect=ValueError('incomplete training')), \
                    patch.object(study, 'run_jobs', side_effect=AssertionError('Private tests must not start')):
                with self.assertRaisesRegex(ValueError, 'incomplete training'):
                    study.evaluate(root, 1)
            self.assertFalse((root/'test').exists())

    def test_result_checks_reject_bad_mean_ledger_or_episode_bank(self):
        result = result_fixture([90001, 90003], 10.)
        study.check_result(result, [90001, 90003])
        bad = deepcopy(result); bad['mean_discounted_profit'] += 1.
        with self.assertRaisesRegex(ValueError, 'mean profit'): study.check_result(bad, [90001, 90003])
        bad = deepcopy(result); bad['episodes'][0]['discounted_reward_terms']['travel_cost'] += 1.
        with self.assertRaisesRegex(ValueError, 'ledger'): study.check_result(bad, [90001, 90003])
        with self.assertRaises(ValueError): study.check_result(result, [90003, 90001])

    def test_initialization_refuses_existing_directory_without_reading_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(study.priced, 'historical_seeds', side_effect=AssertionError('No read needed')):
                with self.assertRaises(FileExistsError): study.init(root, root)

    def test_actual_evaluation_resume_keeps_private_truth_and_fitted_controller_separate(self):
        # Deliberate tiny development fixture, never a declared study scene.
        truth = replace(Scenario(), horizon=2, arrival_profile=(1., 1.),
            fcs_pricing_mode='legacy', mcv_pricing_mode='absolute',
            grid_buy_per_kwh=(.15, .16), grid_sell_per_kwh=(.1, .11),
            charging_utility_intercept=3.).validate()
        fitted = replace(truth, arrival_rates=((2., 2.5, 2., 2.5),)*2,
            charging_utility_intercept=2.7, price_sensitivity_per_dollar=.8).validate()
        seeds = [4_210_000_001, 4_210_000_003]
        for backend in ('inventory', 'rollout'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); model_stub(root)
                (root/'provenance').mkdir()
                history = root/'provenance/historical_seeds.npz'
                np.savez_compressed(history, seeds=np.array([17], dtype=np.int64))
                study.seal(root/'protocol.json', {'historical_seed_sha256': study.digest(history)})
                setting = dict(rule(), switch_period=1)
                specification = dict(backend=backend, kind='fixed', seed=7,
                    setting=setting, label='tiny_private_fixture')
                path = root/'result.json'
                with patch.object(study, 'fitted_model', return_value=SimpleNamespace(scenario=fitted)):
                    result = study.evaluation(root, truth, fitted, seeds, path, specification)
                    self.assertEqual(result['invalid_actions'], 0)
                    self.assertEqual(result['evaluation_environment_sha256'], model_scenario_sha256(truth))
                    self.assertEqual(result['history_boundary']['model_scenario_sha256'],
                        model_scenario_sha256(fitted))
                    self.assertNotEqual(result['evaluation_environment_sha256'],
                        result['history_boundary']['model_scenario_sha256'])
                    with patch.object(study, 'evaluate_history_policy',
                                      side_effect=AssertionError('Completed rows must not rerun')):
                        self.assertEqual(study.evaluation(root, truth, fitted, seeds, path, specification), result)
                    with self.assertRaisesRegex(ValueError, 'model or settings'):
                        study.inspect_evaluation(root, seeds, path, specification,
                            truth=fitted, scenario=fitted)
                    altered = deepcopy(result)
                    altered['history_boundary']['true_model_passed_to_controller'] = True
                    study.seal(path, altered)
                    with self.assertRaisesRegex(ValueError, 'boundary'):
                        study.inspect_evaluation(root, seeds, path, specification,
                            truth=truth, scenario=fitted)

    def test_incomplete_evaluation_trace_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root/'partial.json'
            trace = path.with_suffix('.pricing.json'); trace.write_text('[]')
            with patch.object(study, 'evaluate_history_policy', side_effect=AssertionError('No overwrite')):
                with self.assertRaisesRegex(FileExistsError, 'Incomplete'):
                    study.evaluation(root, city(), city(), [4_210_000_001], path,
                        dict(backend='inventory', kind='fixed', seed=7, setting=rule(), label='partial'))
            self.assertEqual(trace.read_text(), '[]')

    def test_source_protocol_and_frozen_snapshot_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder); root = directory/'study'; controls = directory/'controls'
            controls.mkdir()
            paths = [ROOT/study.SCENARIO_FILE, ROOT/'experiments/run_history_adp_study.py']
            with patch.object(study.priced, 'historical_seeds', return_value=({17}, {})), \
                    patch.object(study, 'source_files', return_value=paths), redirect_stdout(io.StringIO()):
                study.init(root, controls)
                original = study.sealed(root/'protocol.json')
                self.assertEqual(study.verify(root), original)
                for key, value in [('action_seed', study.ACTION_SEED+1),
                                   ('logging_seed', study.LOGGING_SEED+1),
                                   ('final_policy_episodes', 3839)]:
                    bad = deepcopy(original); bad[key] = value; study.seal(root/'protocol.json', bad)
                    with self.assertRaises(ValueError, msg=key): study.verify(root)
                study.seal(root/'protocol.json', original)
                snapshot = root/'source_snapshot'/study.SCENARIO_FILE
                snapshot.write_text('changed frozen scenario')
                with self.assertRaises(ValueError): study.verify(root)


if __name__ == '__main__':
    unittest.main()
