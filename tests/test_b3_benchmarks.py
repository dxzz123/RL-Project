"""Small guard tests; no training, rule simulation, or test-scene evaluation."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from charging_env.config import Scenario
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'Install training dependencies for checkpoint metadata tests')
class B3BenchmarkGuardTests(unittest.TestCase):
    def setUp(self):
        import sys
        self.project = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('run_b3_benchmarks',self.project/'experiments/run_b3_benchmarks.py')
        with patch.object(sys,'path',[str(self.project/'experiments')]+sys.path):
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)/'study'
        self.study.mkdir()
        self.scenario = replace(Scenario(),charging_utility_intercept=3.).validate()
        self.scenario_path = self.study/'scenario.json'
        self.scenario.to_json(self.scenario_path)
        self.protocol_path = self.study/'protocol.json'
        self.protocol_path.write_text('{}\n')
        _,self.provenance = self.module.load_study(self.study,self.scenario_path,self.protocol_path)

    def metadata_file(self,*,updates=500,activation='silu',intercept=3.):
        settings = replace(self.module.expected_training_config(7),dispatch_final_activation=activation)
        path = self.study/'metadata.pt'
        torch.save({'training_config':asdict(settings),
                    'scenario':asdict(replace(self.scenario,charging_utility_intercept=intercept)),
                    'update_count':updates,'episode_count':16*updates,'environment_periods':192*updates},path)
        return path

    def test_metadata_guard_accepts_only_matching_budget_activation_and_scenario(self):
        valid = self.module.checkpoint_metadata(self.metadata_file(),self.scenario,7,'final')
        self.assertEqual(valid['training_physical_periods'],96000)
        for change in ({'updates':499},{'activation':'tanh'},{'intercept':0.}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.module.checkpoint_metadata(self.metadata_file(**change),self.scenario,7,'final')
        self.module.checkpoint_metadata(self.metadata_file(updates=0),self.scenario,7,'initial')
        self.module.checkpoint_metadata(self.metadata_file(updates=425),self.scenario,7,'validation_selected')

    def make_selection(self):
        def evaluate(config,seeds,specification,*,action_seed):
            self.assertEqual(seeds,self.module.VALIDATION_SEEDS)
            self.assertFalse(set(seeds)&set(self.module.TEST_SEEDS+self.module.STRESS_SEEDS))
            self.assertEqual(action_seed,271991)
            return {'mean_discounted_profit':1.,'invalid_actions':0}
        with patch.object(self.module,'evaluate_rule',side_effect=evaluate) as rules, \
             patch.object(self.module,'evaluate_checkpoint') as learned,redirect_stdout(io.StringIO()):
            self.module.select_rules(self.study,self.scenario,self.provenance)
        self.assertEqual(rules.call_count,61)
        learned.assert_not_called()

    def test_selection_uses_only_validation_and_can_run_without_checkpoints(self):
        self.make_selection()
        selection=json.loads((self.study/'rule_selection/selection.json').read_text())
        self.assertEqual(set(selection['family_winners']),self.module.RULE_FAMILIES)
        self.assertFalse(selection['test_scenarios_accessed'])
        self.assertFalse((self.study/'benchmarks').exists())

    def test_evaluation_requires_all_frozen_models_before_creating_test_output(self):
        self.make_selection()
        with patch.object(self.module,'evaluate_rule') as rules, \
             patch.object(self.module,'evaluate_checkpoint') as learned,self.assertRaises(ValueError):
            self.module.evaluate_models(self.study,self.scenario,self.provenance)
        rules.assert_not_called();learned.assert_not_called()
        self.assertFalse((self.study/'benchmarks').exists())

    def test_modified_selection_protocol_is_rejected_before_tests(self):
        self.make_selection()
        path=self.study/'rule_selection/protocol.json'
        path.write_text(path.read_text()+' ')
        with patch.object(self.module,'evaluate_checkpoint') as learned,self.assertRaises(ValueError):
            self.module.evaluate_models(self.study,self.scenario,self.provenance)
        learned.assert_not_called()
        self.assertFalse((self.study/'benchmarks').exists())


if __name__=='__main__':
    unittest.main()
