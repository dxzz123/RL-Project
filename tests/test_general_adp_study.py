"""Rollout study guards use saved synthetic fixtures, never reserved rollouts."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('general_adp_study_tests', ROOT/'experiments/run_general_adp_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def result(seeds, profit=10.):
    terms = dict(mobile_service_revenue=profit, fixed_service_revenue=0., discharge_revenue=0.,
        procurement_cost=0., travel_cost=0., operating_cost=0., unmet_demand_cost=0., terminal_inventory_value=0.)
    return dict(evaluation_seeds=list(seeds), n_episodes=len(seeds), invalid_actions=0,
        mean_discounted_profit=profit, episodes=[dict(seed=s, discounted_profit=profit,
        discounted_reward_terms=terms.copy()) for s in seeds])


def trace_fixture(scenario, config, seeds):
    trace = []
    stats = dict(decisions=0, candidate_plans=0, current_plan_outcomes=0, future_model_queries=0,
        exogenous_draws=0, coordinate_candidate_outcomes=0, boundary_roundoff_corrections=0,
        max_boundary_roundoff_kwh=0.)
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed, 0x524F4C4C]))
        for t in range(scenario.horizon):
            remaining = scenario.horizon-t-1
            draws = 2*rng.integers(0, np.iinfo(np.int64).max//2, size=(remaining, config.rollout_samples), dtype=np.int64)
            budget = remaining*config.rollout_samples
            trace.append(dict(episode_key=seed, episode_decision=t, period=t, candidates=['service_first_base'],
                selected='service_first_base', current_profit=[10.], mean_total_scores=[10.],
                sampled_total_scores=[[10.]*config.rollout_samples], future_periods=list(range(t+1, scenario.horizon)),
                future_draw_seeds=draws.tolist(), future_model_queries=budget, query_budget=budget,
                coordinate_candidate_outcomes=2, seconds=.01))
            for key in ('decisions', 'candidate_plans', 'current_plan_outcomes'): stats[key] += 1
            stats['future_model_queries'] += budget; stats['exogenous_draws'] += draws.size
            stats['coordinate_candidate_outcomes'] += 2
    return trace, stats


class GeneralADPStudyTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(cases={}, controls={}, source_sha256={})
        for case, file in study.CASES.items():
            scenario = study.Scenario.from_json(ROOT/file)
            self.protocol['cases'][case] = dict(scenario=study.normalized(asdict(scenario)),
                configurations=study.configuration(case, scenario), validation_seeds=study.VALIDATION_SEEDS[case],
                test_seeds=study.TEST_SEEDS[case], rule_candidates=study.rule_candidates(scenario))

    def write_rollout(self, directory, case, price, seed, profit=10.):
        data = self.protocol['cases'][case]
        scenario = study._scenario_data(data['scenario'])
        config = study.config_from_data(data['configurations'][str(price)][str(seed)])
        trace, stats = trace_fixture(scenario, config, data['validation_seeds'])
        provenance = study.normalized(study.RolloutADPPolicy(scenario, config).provenance())
        provenance['statistics'] = stats
        self.protocol['source_sha256'].update(provenance['source_sha256'])
        path = study.validation_path(directory, case, price, seed)
        trace_path, provenance_path = study.rollout_paths(path)
        study.write(trace_path, trace); study.write(provenance_path, provenance)
        output = result(data['validation_seeds'], profit)
        output.update(planning_seed=seed, price_index=price, planning_statistics=stats,
            planning_trace_sha256=study.digest(trace_path), planning_provenance_sha256=study.digest(provenance_path))
        study.write(path, output)
        return path, trace, stats

    def test_protocol_has_online_planning_no_offline_training_and_all_rules(self):
        for case, data in self.protocol['cases'].items():
            scenario = study.Scenario.from_json(ROOT/study.CASES[case])
            expected = scenario.n_price_levels*(3+len(study.SCHEDULES))
            self.assertEqual(len(data['rule_candidates']), expected)
            self.assertEqual(len({r['id'] for r in data['rule_candidates']}), expected)
            for price in study.PRICE_INDICES:
                for seed in study.SEEDS:
                    cfg = study.config_for(case, seed, price)
                    self.assertEqual(cfg.rollout_samples, 8)
                    self.assertEqual(cfg.base_recharge_threshold, .2)
                    self.assertNotIn('training_seed_base', asdict(cfg))
                    self.assertEqual(study.config_from_data(study.normalized(asdict(cfg))), cfg)

    def test_holdouts_are_disjoint_from_control_training_and_monitoring(self):
        study.validate_splits(self.protocol)
        changed = deepcopy(self.protocol)
        changed['cases']['city']['test_seeds'] = changed['cases']['small']['validation_seeds']
        with self.assertRaisesRegex(ValueError, 'disjoint'): study.validate_splits(changed)
        changed = deepcopy(self.protocol)
        changed['controls']['fake'] = dict(family='ppo', episode_count=0, training_config=dict(
            seed=0, training_seed_base=0, warmstart_seed_base=0, warmstart_episodes=0,
            eval_seed_base=changed['cases']['small']['test_seeds'][0], eval_episodes=1))
        with self.assertRaisesRegex(ValueError, 'disjoint'): study.validate_splits(changed)

    def test_trace_reconstructs_every_local_seed_and_query_counter(self):
        data = self.protocol['cases']['city']; scenario = study._scenario_data(data['scenario'])
        config = study.config_for('city', 19, study.PRICE_INDICES[-1])
        trace, stats = trace_fixture(scenario, config, [17, 19])
        used = study.check_planning_trace(trace, stats, scenario, config, [17, 19])
        self.assertTrue(all(s % 2 == 0 for s in used))
        self.assertEqual(len(used), 2*config.rollout_samples*sum(range(scenario.horizon)))
        changed = deepcopy(trace); changed[0]['future_draw_seeds'][0][0] += 2
        with self.assertRaisesRegex(ValueError, 'seed protocol'):
            study.check_planning_trace(changed, stats, scenario, config, [17, 19])
        changed = dict(stats, future_model_queries=stats['future_model_queries']+1)
        with self.assertRaisesRegex(ValueError, 'statistics'):
            study.check_planning_trace(trace, changed, scenario, config, [17, 19])

    def test_trace_checks_selected_argmax_and_terminal_current_profit(self):
        scenario = study._scenario_data(self.protocol['cases']['small']['scenario'])
        config = study.config_for('small', 7, study.PRICE_INDICES[0])
        trace, stats = trace_fixture(scenario, config, [17])
        changed = deepcopy(trace); changed[0]['selected'] = 'inventory'
        with self.assertRaisesRegex(ValueError, 'selection'):
            study.check_planning_trace(changed, stats, scenario, config, [17])
        changed = deepcopy(trace); changed[-1]['current_profit'] = [11.]
        with self.assertRaisesRegex(ValueError, 'Terminal'):
            study.check_planning_trace(changed, stats, scenario, config, [17])

    def test_rollout_provenance_hash_and_exact_configuration_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); self.protocol['cases']['small']['validation_seeds'] = [17, 19]
            path, _, _ = self.write_rollout(directory, 'small', study.PRICE_INDICES[0], 7)
            actual = study.inspect_rollout(path, self.protocol, 'small', study.PRICE_INDICES[0], 7, [17, 19])
            self.assertEqual(actual['planning_seed'], 7)
            changed = study.read(path); changed['price_index'] = study.PRICE_INDICES[-1]; study.write(path, changed)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                study.inspect_rollout(path, self.protocol, 'small', study.PRICE_INDICES[0], 7, [17, 19])

    def test_rule_selection_uses_all_validation_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); study.write(directory/'protocol.json', self.protocol)
            def fake(scenario, seeds, settings, **kwargs):
                output = result(seeds, float(self.protocol['cases']['small']['rule_candidates'].index(settings)))
                output['policy_specification'] = settings
                return output
            with patch.object(study, 'verify', return_value=self.protocol), patch.object(study, 'evaluate_rule', side_effect=fake), redirect_stdout(io.StringIO()):
                study.rules(directory, 'small')
            selection = study.verify_rules(directory, self.protocol, 'small')
            selection['winners']['inventory_rule'] = selection['rows'][21]
            study.seal(directory/'rule_selection/small/selection.json', selection)
            with self.assertRaisesRegex(ValueError, 'maximize'): study.verify_rules(directory, self.protocol, 'small')

    def test_price_selection_averages_three_planning_seeds_not_best_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); self.protocol['cases']['small']['validation_seeds'] = [17, 19]
            study.write(directory/'protocol.json', self.protocol)
            for price in study.PRICE_INDICES:
                for seed in study.SEEDS:
                    low = study.PRICE_INDICES[0]
                    profit = 1000. if price == low and seed == 7 else (0. if price == low else 400.)
                    self.write_rollout(directory, 'small', price, seed, profit)
            with patch.object(study, 'verify', return_value=self.protocol): study.select(directory, 'small')
            selection = study.verify_prices(directory, self.protocol, 'small')
            self.assertEqual(selection['winner']['price_index'], study.PRICE_INDICES[-1])
            self.assertEqual(selection['winner']['mean_discounted_profit'], 400.)
            self.assertFalse(selection['offline_training'])
            self.assertNotIn('checkpoint_sha256', selection)

    def test_missing_selection_fails_before_creating_final_test_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(study, 'verify', return_value=self.protocol), patch.object(study, 'run_jobs') as jobs:
                with self.assertRaises(FileNotFoundError): study.evaluate(directory)
                self.assertFalse((directory/'test').exists()); jobs.assert_not_called()

    def test_partial_validation_jobs_are_explicit_and_workers_bounded(self):
        with patch.object(study, 'verify', return_value=self.protocol), patch.object(study, 'run_jobs') as jobs:
            study.validate(Path('/unused'), 'city', 19, study.PRICE_INDICES[-1], 3)
            jobs.assert_called_once_with([('/unused', 'validation', 'city', 'rollout_adp', 19, study.PRICE_INDICES[-1])], 3)
        for workers in (0, 4, True):
            with self.assertRaises(ValueError): study.run_jobs([], workers)

    def test_bad_actions_nonfinite_profit_and_inconsistent_ledgers_fail(self):
        good = result([17, 19]); study.check_result(good, [17, 19])
        for bad in (dict(good, invalid_actions=1), dict(good, mean_discounted_profit=float('nan')),
                    dict(good, mean_discounted_profit=11.)):
            with self.assertRaises(ValueError): study.check_result(bad, [17, 19])
        wrong = deepcopy(good); wrong['episodes'][0]['discounted_profit'] += 1.
        with self.assertRaisesRegex(ValueError, 'reward ledger'): study.check_result(wrong, [17, 19])

    def test_zero_continuation_uses_selected_price_and_same_coordinate_budget(self):
        scenario = study._scenario_data(self.protocol['cases']['small']['scenario'])
        calls = []
        def coordinate(obs, config, shadow, **kwargs):
            calls.append((shadow, kwargs['sweeps'], kwargs['energy_fractions']))
            kwargs['diagnostics'].update(candidate_outcomes=1, last_call=dict(candidate_budget=10))
            return []
        def evaluate(config, seeds, pricing, dispatch, **kwargs):
            obs = {'stage': 0, 'mobile_available': np.ones(config.n_zones, dtype=bool)}
            np.testing.assert_array_equal(pricing(obs, None), np.full(config.n_zones, study.PRICE_INDICES[-1]))
            dispatch({}, None)
            return result(seeds)
        with patch.object(study, 'coordinate_inventory_dispatch', side_effect=coordinate), \
             patch.object(study, 'evaluate_policy', side_effect=evaluate):
            output = study.evaluate_zero(scenario, [17, 19], study.PRICE_INDICES[-1])
        self.assertEqual(calls, [(0., 2, (1.,))]); self.assertEqual(output['price_index'], study.PRICE_INDICES[-1])

    def test_existing_output_and_changed_sealed_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(FileExistsError): study.init(directory, directory)
            path = directory/'selection.json'; study.seal(path, dict(a=1))
            path.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError, 'Frozen evidence'): study.sealed(path)


if __name__ == '__main__': unittest.main()
