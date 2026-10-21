"""Guard tests for the new study; never train or simulate held-out scenarios."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from charging_env.config import Scenario
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install training dependencies for checkpoint guards')
class ImprovedB3BenchmarkGuardTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('improved_bench', project / 'experiments/run_b3_improved_benchmarks.py')
        with patch.object(sys, 'path', [str(project / 'experiments')] + sys.path):
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name)
        self.scenario = replace(Scenario(), charging_utility_intercept=3.).validate()
        self.expected = replace(self.module.old_training_config(7), dispatch_agent_id=True,
            encoder_layer_norm=True, inventory_potential=.3, gae_lambda=1.,
            training_seed_base=10000000, eval_seed_base=4300001, eval_episodes=64)
        self.protocol = {'scenario': asdict(self.scenario), 'training_seeds': [7, 19, 37],
            'configuration_by_seed': {str(seed): asdict(replace(self.expected, seed=seed,
                training_seed_base=10000000 + index*20000)) for index, seed in enumerate((7, 19, 37))}}
        self.scenario_path = self.study / 'scenario.json'
        self.scenario.to_json(self.scenario_path)
        self.protocol_path = self.study / 'protocol.json'
        self.module.write_json(self.protocol_path, self.protocol)
        _, _, self.provenance = self.module.load_study(self.study, self.scenario_path, self.protocol_path)

    def checkpoint(self, *, updates=500, settings=None, periods=None):
        path = self.study / 'metadata.pt'
        torch.save({'training_config': asdict(settings or self.expected), 'scenario': asdict(self.scenario),
            'update_count': updates, 'episode_count': updates*16,
            'environment_periods': updates*192 if periods is None else periods}, path)
        return path

    def selection_fixture(self):
        directory = self.study / 'rule_selection'
        candidates = self.module.candidates(self.scenario)
        protocol = {'scenario': asdict(self.scenario), 'validation_seeds': self.module.VALIDATION_SEEDS,
            'action_seed': self.module.ACTION_SEED, 'rule_candidates': candidates,
            'source_sha256': self.module.source_hashes(selection=True)}
        self.module.write_json(directory / 'protocol.json', protocol)
        winners = {candidate['family']: candidate for candidate in candidates}
        self.module.write_json(directory / 'selection.json', {'family_winners': winners,
            'strongest_validation_family': 'fixed_inventory',
            'selection_protocol_sha256': self.module.digest(directory / 'protocol.json')})

    def test_new_config_and_final_budget_are_enforced(self):
        good = self.module.checkpoint_metadata(self.checkpoint(), self.scenario, 7, 'final', self.expected)
        self.assertEqual(good['training_physical_periods'], 96000)
        # Configurations must match the selected bundle, even when the old
        # checkpoint has the same scenario, architecture width, seed and budget.
        for kwargs in ({'settings': replace(self.expected, inventory_potential=0.)},
                       {'settings': replace(self.expected, dispatch_agent_id=False)},
                       {'settings': replace(self.expected, training_seed_base=10020000)},
                       {'updates': 499}, {'periods': 96001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.module.checkpoint_metadata(self.checkpoint(**kwargs), self.scenario, 7, 'final', self.expected)
        self.module.checkpoint_metadata(self.checkpoint(updates=325), self.scenario, 7,
                                        'validation_selected', self.expected)
        self.module.checkpoint_metadata(self.checkpoint(updates=0), self.scenario, 7,
                                        'warmstart', self.expected)

    def test_checkpoint_selection_must_not_use_any_held_out_split(self):
        splits = (self.module.VALIDATION_SEEDS, self.module.TEST_SEEDS,
                  self.module.HIGH_DEMAND_SEEDS, self.module.LOW_INVENTORY_SEEDS)
        self.assertEqual(sum(len(split) for split in splits), len(set().union(*map(set, splits))))
        for split in splits:
            conflicting = replace(self.expected, eval_seed_base=split[0])
            with self.subTest(seed=split[0]), self.assertRaisesRegex(ValueError, 'overlap'):
                self.module.checkpoint_metadata(self.checkpoint(settings=conflicting), self.scenario,
                                                7, 'final', conflicting)

    def test_partial_warmstart_set_rejected_before_any_test_execution(self):
        self.selection_fixture()
        warm = self.study / 'training/seed_7/warmstart_checkpoint.pt'
        warm.parent.mkdir(parents=True)
        warm.touch()
        with patch.object(self.module, 'evaluate_rule') as rules, \
             patch.object(self.module, 'evaluate_checkpoint') as learned, \
             self.assertRaisesRegex(ValueError, 'all three checkpoints or none'):
            self.module.evaluate_models(self.study, self.scenario, self.protocol, self.provenance)
        rules.assert_not_called()
        learned.assert_not_called()
        self.assertFalse((self.study / 'benchmarks').exists())

    def test_per_seed_protocol_must_contain_all_three_configs(self):
        self.selection_fixture()
        del self.protocol['configuration_by_seed']['37']
        with patch.object(self.module, 'evaluate_rule') as rules, \
             patch.object(self.module, 'evaluate_checkpoint') as learned, \
             self.assertRaisesRegex(ValueError, 'full configuration'):
            self.module.evaluate_models(self.study, self.scenario, self.protocol, self.provenance)
        rules.assert_not_called()
        learned.assert_not_called()
        self.assertFalse((self.study / 'benchmarks').exists())


if __name__ == '__main__':
    unittest.main()
