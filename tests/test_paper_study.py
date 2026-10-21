"""Guards against accidental selection leakage and overwritten study evidence."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load('paper_study_guard_target', 'run_paper_ppo_study.py')
summary = load('paper_summary_guard_target', 'summarize_paper_ppo_study.py')


class PaperStudyTests(unittest.TestCase):
    def test_five_holdouts_are_disjoint_and_overlap_is_rejected(self):
        blocks = list(study.validate_splits().values())
        self.assertEqual([len(block) for block in blocks], [64, 128, 512, 256, 256])
        self.assertTrue(all(value % 2 == 1 for block in blocks for value in block))
        self.assertEqual(sum(len(block) for block in blocks), len(set(sum(blocks, []))))
        with patch.object(study, 'MONITOR_SEEDS', [study.TEST_SEEDS[0]]):
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                study.validate_splits()

    def test_source_snapshot_matches_frozen_bytes_and_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source.py').write_text('original = True\n')
            output = root / 'study'
            output.mkdir()
            protocol = {'training_source_sha256': {'source.py': study.digest(root / 'source.py')}}
            study.write_json(output / 'protocol.json', protocol)
            with patch.object(study, 'ROOT', root):
                study.snapshot_training_sources(output, protocol)
                copied = output / 'provenance/executed_training_source/source.py'
                self.assertEqual(copied.read_bytes(), (root / 'source.py').read_bytes())
                (root / 'source.py').write_text('original = False\n')
                with self.assertRaisesRegex(ValueError, 'sources changed'):
                    study.verify_sources(protocol['training_source_sha256'])

    def _selected_fixture(self, output):
        scenario = study.replace(study.Scenario(), charging_utility_intercept=3.).validate()
        specs = [{'id': f'{family}_{level}', 'family': family, 'price_index': index}
                 for index, family in enumerate(study.FAMILIES) for level in ('low', 'high')]
        root = {'training_source_sha256': {}}
        study.write_json(output / 'protocol.json', root)
        root_hash = study.digest(output / 'protocol.json')
        calls = []
        def evaluate(config, seeds, spec, **kwargs):
            calls.append(list(seeds))
            value = 10*spec['price_index'] + (2 if spec['id'].endswith('high') else 1)
            return {'mean_discounted_profit': value, 'evaluation_seeds': list(seeds),
                    'policy_specification': spec}
        with patch.object(study, 'candidates', return_value=specs), \
                patch.object(study, 'source_hashes', return_value={}), \
                patch.object(study, 'evaluate_rule', side_effect=evaluate), \
                patch.object(study, 'VALIDATION_SEEDS', [181000001, 181000003]), \
                redirect_stdout(io.StringIO()):
            study.select_rules(output, scenario, root_hash)
        return scenario, root, root_hash, specs, calls

    def test_selection_reads_only_validation_and_freezes_each_family(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _, _, _, _, calls = self._selected_fixture(output)
            self.assertEqual(calls, [[181000001, 181000003]]*6)
            selected = json.loads((output / 'rule_selection/selection.json').read_text())
            self.assertEqual(set(selected['family_winners']), set(study.FAMILIES))
            self.assertTrue(all(row['id'].endswith('high') for row in selected['family_winners'].values()))
            self.assertEqual(selected['strongest_validation_family'], 'time_inventory')
            self.assertFalse(selected['test_scenarios_accessed'])

    def test_changed_winner_is_rejected_before_test_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scenario, root, root_hash, specs, _ = self._selected_fixture(output)
            path = output / 'rule_selection/selection.json'
            selected = json.loads(path.read_text())
            selected['family_winners']['service_first']['mean_discounted_profit'] = 99999.
            study.write_json(path, selected)
            with patch.object(study, 'candidates', return_value=specs), \
                    patch.object(study, 'VALIDATION_SEEDS', [181000001, 181000003]):
                with self.assertRaisesRegex(ValueError, 'do not maximize'):
                    study.prepare_evaluation(output, scenario, root, root_hash)
            self.assertFalse((output / 'benchmarks').exists())

    def test_missing_model_is_rejected_before_any_test_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scenario, root, root_hash, specs, _ = self._selected_fixture(output)
            with patch.object(study, 'candidates', return_value=specs), \
                    patch.object(study, 'VALIDATION_SEEDS', [181000001, 181000003]), \
                    patch.object(study, 'paper_metadata') as metadata:
                with self.assertRaisesRegex(ValueError, 'all nine complete'):
                    study.prepare_evaluation(output, scenario, root, root_hash)
                metadata.assert_not_called()
            self.assertFalse((output / 'benchmarks').exists())

    def test_nested_nonfinite_optimizer_tensor_and_wrong_algorithm_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite tensor'):
            study._all_tensors_finite({'optimizers': {'actor': {'state': [{'moment': torch.tensor(float('nan'))}]}}}, 'checkpoint')
        with patch('torch.load', return_value={'algorithm': 'legacy', 'format_version': 1}):
            with self.assertRaisesRegex(ValueError, 'new manuscript'):
                study.paper_metadata(Path('unused.pt'), study.Scenario(), 7, {})

    def test_existing_training_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = {'training_source_sha256': {}, 'configuration_by_seed': study.configuration_by_seed()}
            study.write_json(output / 'protocol.json', root)
            previous = output / 'training/seed_7'
            previous.mkdir(parents=True)
            saved = previous / 'checkpoint.pt'
            saved.write_bytes(b'preserve the previous checkpoint')
            with patch('charging_env.rl.paper_ppo.PaperPPOTrainer') as trainer:
                with self.assertRaisesRegex(FileExistsError, 'already has training evidence'):
                    study.train_seed(output, study.Scenario(), root, study.digest(output / 'protocol.json'), 7)
                trainer.assert_not_called()
            self.assertEqual(saved.read_bytes(), b'preserve the previous checkpoint')

    def test_show_prints_existing_analysis_without_audit_or_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            comparison = {'mean': {'mean_difference': -1., 'paired_bootstrap_confidence_interval_95': [-2., -.1]}}
            cases = {case: {'mean_discounted_profits': {name: float(i) for i, name in enumerate(summary.METHODS)}}
                     for case in ('base', 'high_demand', 'low_inventory')}
            cases['base'].update(new_vs_validation_best_rule=comparison, new_vs_legacy=comparison, new_vs_fitted=comparison)
            study.write_json(output / 'analysis.json', {'cases': cases})
            before = (output / 'analysis.json').read_bytes()
            captured = io.StringIO()
            with patch.object(summary, 'summarize', side_effect=AssertionError('Viewer must not audit')), redirect_stdout(captured):
                self.assertEqual(summary.main(['--study-dir', str(output), '--show']), 0)
            self.assertIn('Paper joint PPO', captured.getvalue())
            self.assertIn('paired 95% interval', captured.getvalue())
            self.assertEqual((output / 'analysis.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
