"""Scenario routing and preflight isolation for experiment command-line entrypoints."""
from contextlib import redirect_stderr, redirect_stdout
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

ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name):
    specification = importlib.util.spec_from_file_location(name, ROOT/'experiments'/f'{name}.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'Install the train extra to test experiment entrypoints')
class ScenarioEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.convergence = load_experiment('run_convergence')
        self.benchmarks = load_experiment('run_policy_benchmarks')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scenario = replace(Scenario(), charging_utility_intercept=3.).validate()
        self.scenario_path = self.root/'scenario.json'
        self.scenario.to_json(self.scenario_path)
        self.checkpoints = self.root/'checkpoints'
        self.output = self.root/'benchmark'

    def save_checkpoint_metadata(self, scenario, *, legacy=False):
        metadata = asdict(scenario)
        if legacy:
            metadata.pop('charging_utility_intercept')
        for seed in (7,19,37):
            folder = self.checkpoints/f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            for name in ('checkpoint.pt','best_checkpoint.pt'):
                torch.save({'scenario':metadata},folder/name)

    def arguments(self):
        return ['--scenario',str(self.scenario_path),'--checkpoint-root',str(self.checkpoints),
                '--output-dir',str(self.output),'--validation-episodes','2',
                '--test-episodes','2','--stress-episodes','2']

    def test_custom_scenario_reaches_training_constructor(self):
        with patch.object(self.convergence,'BudgetTrainer') as constructor:
            self.convergence.main(['--seed','7','--scenario',str(self.scenario_path),
                                   '--output-root',str(self.root/'training')])
        scenario,settings = constructor.call_args.args
        self.assertEqual(scenario.charging_utility_intercept,3.)
        self.assertEqual(settings.updates,500)
        constructor.return_value.train.assert_called_once_with(self.root/'training/seed_7')

    def test_legacy_checkpoint_defaults_to_zero_and_is_rejected_before_search(self):
        self.save_checkpoint_metadata(Scenario(),legacy=True)
        path = self.checkpoints/'seed_7/checkpoint.pt'
        self.assertEqual(self.benchmarks.checkpoint_scenario(path).charging_utility_intercept,0.)
        errors = io.StringIO()
        with patch.object(self.benchmarks,'candidates') as candidates, \
             patch.object(self.benchmarks,'evaluate_rule') as evaluate, \
             redirect_stderr(errors),self.assertRaises(SystemExit) as raised:
            self.benchmarks.main(self.arguments())
        self.assertEqual(raised.exception.code,2)
        self.assertIn('charging_utility_intercept',errors.getvalue())
        self.assertIn('checkpoint=0.0, selected=3.0',errors.getvalue())
        candidates.assert_not_called();evaluate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_all_six_checkpoints_are_checked_before_search(self):
        self.save_checkpoint_metadata(self.scenario)
        torch.save({'scenario':asdict(Scenario())},self.checkpoints/'seed_37/best_checkpoint.pt')
        with patch.object(self.benchmarks,'candidates') as candidates, \
             patch.object(self.benchmarks,'checkpoint_scenario',wraps=self.benchmarks.checkpoint_scenario) as read, \
             redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            self.benchmarks.main(self.arguments())
        self.assertEqual(read.call_count,6)
        candidates.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_custom_scenario_and_rng_reach_every_mocked_evaluation(self):
        self.save_checkpoint_metadata(self.scenario)
        spec={'id':'frozen_test_rule','family':'fixed_inventory','price_index':3,'shadow_price':.1}
        calls=[]
        def result(scenario,seeds,*,action_seed,label):
            calls.append((scenario,seeds,action_seed,label))
            return {'label':label,'invalid_actions':0,'mean_discounted_profit':1.,
                    'episodes':[{'seed':seed,'discounted_profit':1.} for seed in seeds],
                    'evaluation_seeds':seeds}
        def rule(scenario,seeds,spec,label=None,*,action_seed):
            return result(scenario,seeds,action_seed=action_seed,label=label or spec['id'])
        def learned(scenario,seeds,path,*,deterministic,action_seed,label):
            return result(scenario,seeds,action_seed=action_seed,label=label)
        def average(models,label):
            return dict(models[0],label=label)
        arguments=self.arguments()+['--validation-seed-base','3100001','--test-seed-base','3300001',
                                    '--stress-seed-base','3500001','--action-seed','7163']
        with patch.object(self.benchmarks,'candidates',return_value=[spec]), \
             patch.object(self.benchmarks,'evaluate_rule',side_effect=rule), \
             patch.object(self.benchmarks,'evaluate_checkpoint',side_effect=learned), \
             patch.object(self.benchmarks,'average_trained_models',side_effect=average), \
             patch.object(self.benchmarks,'table_row',side_effect=lambda name,evaluation:{'policy':name,'profit':1.}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(self.benchmarks.main(arguments),0)
        self.assertTrue(calls)
        self.assertTrue(all(config.charging_utility_intercept==3. for config,_,_,_ in calls))
        self.assertTrue(all(seed==7163 for _,_,seed,_ in calls))
        self.assertEqual({tuple(seeds) for _,seeds,_,_ in calls},
                         {(3100001,3100003),(3300001,3300003),(3500001,3500003)})
        protocol=json.loads((self.output/'protocol.json').read_text())
        self.assertEqual(protocol['scenario']['charging_utility_intercept'],3.)
        self.assertEqual(protocol['action_seed'],7163)

    def test_validation_stress_overlap_is_rejected_before_search(self):
        self.save_checkpoint_metadata(self.scenario)
        with patch.object(self.benchmarks,'candidates') as candidates, \
             redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            self.benchmarks.main(self.arguments()+['--validation-seed-base','1100001'])
        candidates.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_seed_bases_and_negative_action_seed_are_rejected(self):
        for flag,value in [('--validation-seed-base','4'),('--test-seed-base','-1'),
                           ('--stress-seed-base','0'),('--action-seed','-2')]:
            with self.subTest(flag=flag),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.benchmarks.main(self.arguments()+[flag,value])
        self.assertFalse(self.output.exists())

    def test_evaluate_rule_forwards_the_optional_action_seed(self):
        spec={'id':'random','family':'random'}
        with patch.object(self.benchmarks,'evaluate_policy',return_value={'invalid_actions':0}) as evaluate:
            self.benchmarks.evaluate_rule(self.scenario,[101,103],spec,action_seed=71)
        self.assertEqual(evaluate.call_args.kwargs['action_seed'],71)
        with patch.object(self.benchmarks,'evaluate_policy',return_value={'invalid_actions':0}) as evaluate:
            self.benchmarks.evaluate_rule(self.scenario,[101,103],spec)
        self.assertEqual(evaluate.call_args.kwargs['action_seed'],54123)


if __name__=='__main__':
    unittest.main()
