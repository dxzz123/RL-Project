"""Check compact benchmark meaning, frozen selection, and holdout boundaries."""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks import compact
from charging_env.config import Scenario


def study_module():
    path = Path(__file__).resolve().parents[1] / 'experiments/run_fitted_value_study.py'
    spec = importlib.util.spec_from_file_location('fitted_value_benchmark_guards', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompactBenchmarkTests(unittest.TestCase):
    def test_frozen_grid_has_three_meaningful_families_and_unique_settings(self):
        scenario = replace(Scenario(), charging_utility_intercept=3.,
                           fcs_pricing_mode='legacy', mcv_pricing_mode='absolute').validate()
        specs = compact.candidates(scenario)
        self.assertEqual(len(specs), 82)
        self.assertEqual(Counter(item['family'] for item in specs),
                         {'service_first': 21, 'constant_inventory': 45, 'time_inventory': 16})
        premium = compact.candidates(replace(scenario, fcs_pricing_mode='base_price',
                                             mcv_pricing_mode='premium'))
        self.assertEqual(len(premium), 52)
        self.assertEqual(Counter(item['family'] for item in premium),
                         {'service_first': 9, 'constant_inventory': 27, 'time_inventory': 16})
        self.assertEqual(len({item['id'] for item in premium}), len(premium))
        wide = compact.candidates(replace(scenario, fcs_pricing_mode='base_price',
                                          mcv_pricing_mode='premium',
                                          mcv_premium_levels=(1.1, 1.4, 1.7, 2.0, 2.2)))
        self.assertEqual(len(wide), 76)
        self.assertEqual(Counter(item['family'] for item in wide),
                         {'service_first': 15, 'constant_inventory': 45, 'time_inventory': 16})
        # Time inventory pins the middle premium, which moves with the menu size.
        self.assertEqual({item['price_index'] for item in wide
                          if item['family'] == 'time_inventory'}, {2})
        self.assertEqual(len({item['id'] for item in specs}), len(specs))
        self.assertEqual(specs, compact.candidates(scenario))
        constants = [spec for spec in specs if spec['family'] == 'constant_inventory']
        self.assertEqual({scenario.price_levels[spec['price_index']] for spec in constants}, {.4, .5, .6, .8, 1.})
        self.assertEqual({spec['shadow_price'] for spec in constants}, {.05, .10, .15, .20, .25, .30, .35, .40, .45})
        self.assertEqual(compact.ENERGY_FRACTIONS, (.5, 1.))
        for spec in specs:
            self.assertIn(spec['family'], compact.FAMILIES)
            self.assertTrue(0 <= spec['price_index'] < len(scenario.price_levels))

    def test_time_schedule_switch_and_terminal_value_have_unambiguous_periods(self):
        scenario = Scenario().validate()
        spec = next(item for item in compact.candidates(scenario)
                    if item['family'] == 'time_inventory' and item['early_shadow_price'] == .4
                    and item['late_shadow_price'] == .1)
        for period, expected in ((0, .4), (5, .4), (6, .1), (10, .1), (11, 0.)):
            self.assertEqual(compact.inventory_shadow({'period': np.array([period])}, scenario, spec), expected)
        constant = next(item for item in compact.candidates(scenario) if item['family'] == 'constant_inventory')
        self.assertEqual(compact.inventory_shadow({'period': np.array([10])}, scenario, constant), .05)
        self.assertEqual(compact.inventory_shadow({'period': np.array([11])}, scenario, constant), 0.)

    def test_three_rules_execute_legal_actions_and_validate_selected_settings(self):
        scenario = replace(Scenario(), horizon=2, arrival_profile=(1., 1.),
                           grid_buy_per_kwh=(.2, .3), grid_sell_per_kwh=(.1, .2),
                           charging_utility_intercept=3.).validate()
        for family in compact.FAMILIES:
            spec = next(item for item in compact.candidates(scenario) if item['family'] == family)
            result = compact.evaluate_rule(scenario, [121, 123], dict(spec, mean_discounted_profit=999.))
            self.assertEqual(result['invalid_actions'], 0)
            self.assertEqual(result['evaluation_seeds'], [121, 123])
            self.assertEqual(result['policy_specification'], spec)
            self.assertTrue(np.isfinite(result['mean_discounted_profit']))
            with self.assertRaisesRegex(ValueError, 'frozen'):
                compact.evaluate_rule(scenario, [121], dict(spec, hidden_test_tuning=True))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            compact.evaluate_rule(scenario, [121], {'family': 'random'})


class CompactStudyGuardTests(unittest.TestCase):
    def setUp(self):
        self.module = study_module()
        self.scenario = replace(Scenario(), charging_utility_intercept=3.).validate()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name)
        self.root_protocol = {'test': 'Selection fixtures do not open test scenarios.'}
        self.module.write_json(self.study / 'protocol.json', self.root_protocol)
        self.root_hash = self.module.digest(self.study / 'protocol.json')

    def select_fixture(self):
        specs = compact.candidates(self.scenario)
        scores = {spec['id']: float(index) for index, spec in enumerate(specs)}
        seen = []
        def validation_only(config, seeds, spec, **kwargs):
            self.assertEqual(config, self.scenario)
            self.assertEqual(seeds, self.module.VALIDATION_SEEDS)
            self.assertTrue(set(seeds).isdisjoint(self.module.TEST_SEEDS))
            seen.append(spec['id'])
            return {'mean_discounted_profit': scores[spec['id']],
                    'evaluation_seeds': list(seeds), 'policy_specification': spec,
                    'invalid_actions': 0}
        with patch.object(self.module, 'evaluate_rule', side_effect=validation_only), redirect_stdout(io.StringIO()):
            self.module.select_rules(self.study, self.scenario, self.root_hash)
        self.assertEqual(seen, [spec['id'] for spec in specs])
        return json.loads((self.study / 'rule_selection/selection.json').read_text())

    def test_holdouts_are_unique_disjoint_odd_blocks_and_overlap_is_rejected(self):
        blocks = self.module.validate_splits()
        self.assertEqual([len(blocks[key]) for key in ('validation', 'base', 'high_demand', 'low_inventory')],
                         [128, 512, 256, 256])
        combined = sum(blocks.values(), [])
        self.assertEqual(len(combined), len(set(combined)))
        with patch.object(self.module, 'TEST_SEEDS', list(self.module.VALIDATION_SEEDS)), \
             self.assertRaisesRegex(ValueError, 'disjoint'):
            self.module.validate_splits()
        with patch.object(self.module, 'VALIDATION_SEEDS', [1, 1]), \
             self.assertRaisesRegex(ValueError, 'unique'):
            self.module.validate_splits()
        with patch.object(self.module, 'VALIDATION_SEEDS', [2]), \
             self.assertRaisesRegex(ValueError, 'odd'):
            self.module.validate_splits()

    def test_selection_uses_only_validation_and_selects_each_family_independently(self):
        selected = self.select_fixture()
        specs = compact.candidates(self.scenario)
        for family in compact.FAMILIES:
            expected = [spec for spec in specs if spec['family'] == family][-1]
            winner = selected['family_winners'][family]
            self.assertEqual({key: value for key, value in winner.items() if key != 'mean_discounted_profit'}, expected)
        self.assertEqual(selected['strongest_validation_family'], 'time_inventory')
        self.assertFalse(selected['test_scenarios_accessed'])
        self.assertFalse((self.study / 'benchmarks').exists())
        with self.assertRaises(FileExistsError):
            self.module.select_rules(self.study, self.scenario, self.root_hash)

    def test_evaluation_refuses_missing_models_before_creating_any_test_output(self):
        self.select_fixture()
        with patch.object(self.module, 'evaluate_policy') as evaluation, \
             self.assertRaisesRegex(ValueError, 'All six complete frozen checkpoints'):
            self.module.prepare_evaluation(self.study, self.scenario, self.root_protocol, self.root_hash)
        evaluation.assert_not_called()
        self.assertFalse((self.study / 'benchmarks').exists())

    def test_evaluation_recomputes_winners_and_rejects_hand_changed_selection(self):
        selected = self.select_fixture()
        selected['family_winners']['time_inventory']['early_shadow_price'] = .123
        self.module.write_json(self.study / 'rule_selection/selection.json', selected)
        with patch.object(self.module, 'evaluate_policy') as evaluation, \
             self.assertRaisesRegex(ValueError, 'maximize validation'):
            self.module.prepare_evaluation(self.study, self.scenario, self.root_protocol, self.root_hash)
        evaluation.assert_not_called()
        self.assertFalse((self.study / 'benchmarks').exists())

    def test_evaluation_rejects_validation_record_with_test_scenarios(self):
        self.select_fixture()
        first = compact.candidates(self.scenario)[0]
        path = self.study / 'rule_selection/validation' / f'{first["id"]}.json'
        record = json.loads(path.read_text())
        record['evaluation_seeds'] = self.module.TEST_SEEDS
        self.module.write_json(path, record)
        with patch.object(self.module, 'evaluate_policy') as evaluation, \
             self.assertRaisesRegex(ValueError, 'candidate or split'):
            self.module.prepare_evaluation(self.study, self.scenario, self.root_protocol, self.root_hash)
        evaluation.assert_not_called()
        self.assertFalse((self.study / 'benchmarks').exists())

    def test_ppo_guard_rejects_nonfinite_tensors_nested_in_optimizer_lists(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Install the train extra for PPO checkpoint metadata checks')
        payload = {'scenario': asdict(self.scenario),
                   'training_config': asdict(self.module.ppo_expected_config(7)),
                   'update_count': 500, 'environment_periods': 96000, 'episode_count': 8000,
                   'optimizer': {'groups': [{'state': (torch.tensor([float('nan')]),)}]}}
        with patch.object(torch, 'load', return_value=payload), \
             self.assertRaisesRegex(ValueError, 'nonfinite PPO tensor'):
            self.module.ppo_metadata(self.study / 'fake.pt', self.scenario, 7)

    def test_averages_use_actual_model_risk_and_preserve_scene_pairing(self):
        keys = ('mean_discounted_profit', 'mean_profit', 'profit_p10', 'lower_tail_mean_10pct',
                'loss_probability', 'mean_served_request_share', 'mean_outside_option_share',
                'pooled_selected_service_fulfillment', 'mean_mobile_delivered_kwh', 'mean_fixed_delivered_kwh',
                'mean_final_inventory_kwh', 'mean_export_grid_kwh', 'mean_charge_grid_kwh', 'mean_mobile_price')
        results = []
        for seed, profits in zip((7, 19, 37), ((10., -10.), (-10., 10.), (10., 10.))):
            result = {key: 1. for key in keys}
            result.update(training_seed=seed, evaluation_seeds=[101, 103],
                          episodes=[{'seed': scene, 'discounted_profit': profit}
                                    for scene, profit in zip((101, 103), profits)],
                          mean_discounted_profit=float(np.mean(profits)),
                          mean_profit=float(np.mean(profits)),
                          loss_probability=float(np.mean(np.asarray(profits) < 0)),
                          lower_tail_mean_10pct=float(min(profits)), invalid_actions=0,
                          mean_reward_terms={'test': 1.}, mean_discounted_reward_terms={'test': 1.},
                          mean_policy_decision_ms={'pricing': 1., 'dispatch': 2.})
            results.append(result)
        actual = self.module.average_models(results, 'three separate policies')
        self.assertAlmostEqual(actual['loss_probability'], 1/3)
        self.assertAlmostEqual(actual['lower_tail_mean_10pct'], -10/3)
        self.assertEqual(actual['episodes'], [{'seed': 101, 'discounted_profit': 10/3},
                                             {'seed': 103, 'discounted_profit': 10/3}])
        bad_order = deepcopy(results)
        bad_order[-1]['evaluation_seeds'].reverse()
        with self.assertRaisesRegex(ValueError, 'identical ordered'):
            self.module.average_models(bad_order, 'wrongly paired')


if __name__ == '__main__':
    unittest.main()
