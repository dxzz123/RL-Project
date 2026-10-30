"""Scale-study guards: frozen settings, validation-only selection and model provenance."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scale_study_test_target', ROOT / 'experiments/run_scale_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class ScaleStudyTests(unittest.TestCase):
    def setUp(self):
        self.scenario = study.Scenario.from_json(ROOT / study.SCENARIO_FILE)
        self.protocol = dict(scenario=asdict(self.scenario),
            configurations={str(seed): asdict(study.config_for(seed)) for seed in study.SEEDS},
            validation_seeds=study.VALIDATION_SEEDS, test_seeds=study.TEST_SEEDS,
            action_seed=study.ACTION_SEED, rule_candidates=study.rule_candidates(self.scenario),
            source_sha256={'source.py': 'verified'})

    def selection_fixture(self, directory):
        study.write(directory / 'protocol.json', self.protocol)
        calls = []
        def evaluate(scenario, seeds, settings, **kwargs):
            calls.append(list(seeds))
            score = float(self.protocol['rule_candidates'].index(settings))
            return dict(label=settings['id'], mean_discounted_profit=score,
                evaluation_seeds=list(seeds), invalid_actions=0,
                policy_specification=settings)
        with patch.object(study, 'verify', return_value=self.protocol), \
                patch.object(study, 'evaluate_rule', side_effect=evaluate), \
                redirect_stdout(io.StringIO()):
            study.select(directory)
        return calls

    def model_fixture(self):
        c = study.config_for(7)
        warm = dict(completed=True, seeds=[2*(c.warmstart_seed_base+c.seed+i)
            for i in range(c.warmstart_episodes)], demonstration_episodes=c.warmstart_episodes,
            demonstration_periods=c.warmstart_episodes*self.scenario.horizon,
            epochs=c.warmstart_epochs, reward_scale=c.reward_scale,
            dispatch_solver='coordinate', coordinate_sweeps=2, solver_exact=False,
            source_sha256={'source.py': 'verified'})
        return SimpleNamespace(config=c, scenario=self.scenario, update_count=500,
            episode_count=8000, environment_periods=96000,
            warmstart_metadata={'completed': True, 'summary': warm},
            _source_hashes=lambda: {'source.py': 'verified'})

    def test_training_and_holdout_banks_are_disjoint_and_budget_is_fixed(self):
        study.validate_splits(self.protocol['configurations'], study.VALIDATION_SEEDS, study.TEST_SEEDS)
        c = study.config_for(7)
        self.assertEqual((c.updates, c.episodes_per_update, c.epochs, c.minibatch_size), (500, 16, 6, 192))
        self.assertEqual((c.warmstart_episodes, c.warmstart_epochs, c.warmstart_dispatch_solver), (256, 100, 'coordinate'))
        self.assertEqual(len(study.TEST_SEEDS), 256)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            study.validate_splits(self.protocol['configurations'], study.VALIDATION_SEEDS, [c.eval_seed_base])
        demo_seed = 2*(c.warmstart_seed_base+c.seed)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            study.validate_splits(self.protocol['configurations'], [demo_seed], study.TEST_SEEDS)

    def test_initialization_snapshots_sources_and_refuses_overwrite_or_protocol_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'study'
            with redirect_stdout(io.StringIO()):
                study.init(directory)
            protocol = study.verify(directory)
            self.assertFalse(protocol['approximate_dp']['supported'])
            self.assertAlmostEqual(protocol['scenario_description']['expected_daily_requests'], 1032.5)
            with self.assertRaises(FileExistsError):
                study.init(directory)
            source = directory / 'source_snapshot' / 'experiments/run_scale_study.py'
            source.write_text(source.read_text() + '\n# Changed snapshot\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                study.verify(directory)
            path = directory / 'protocol.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                study.verify(directory)

    def test_selection_uses_only_validation_and_verifies_winners_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calls = self.selection_fixture(directory)
            self.assertEqual(calls,
                             [study.VALIDATION_SEEDS]*len(study.rule_candidates(self.scenario)))
            selected = study.verify_selection(directory, self.protocol)
            self.assertEqual(set(selected['winners']), set(study.FAMILIES))
            self.assertFalse(selected['test_scenarios_accessed'])
            self.assertTrue(all(not set(seeds) & set(study.TEST_SEEDS) for seeds in calls))
            path = directory / 'rule_selection/selection.json'
            selected['winners']['time_inventory'] = selected['rows'][21]
            study.write(path, selected)
            (path.parent / 'selection.sha256').write_text(study.digest(path) + '\n')
            with self.assertRaisesRegex(ValueError, 'do not maximize'):
                study.verify_selection(directory, self.protocol)

    def test_missing_checkpoint_is_rejected_before_any_model_or_test_is_evaluated(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.selection_fixture(directory)
            with patch.object(study.PaperPPOTrainer, 'load_checkpoint') as load:
                with self.assertRaisesRegex(ValueError, 'all six complete'):
                    study.prepare_evaluation(directory, self.protocol)
                load.assert_not_called()
            self.assertFalse((directory / 'test').exists())

    def test_model_stage_and_initialization_provenance_are_required(self):
        model = self.model_fixture()
        study.validate_model(model, self.protocol, 7)
        with self.assertRaisesRegex(ValueError, 'fresh training budget'):
            study.validate_model(model, self.protocol, 7, initialized_only=True)
        model.update_count = model.episode_count = model.environment_periods = 0
        study.validate_model(model, self.protocol, 7, initialized_only=True)
        model.warmstart_metadata['summary']['seeds'][0] += 2
        with self.assertRaisesRegex(ValueError, 'Initialization provenance'):
            study.validate_model(model, self.protocol, 7, initialized_only=True)
        model = self.model_fixture()
        model.warmstart_metadata['summary']['source_sha256'] = {'source.py': 'changed'}
        with self.assertRaisesRegex(ValueError, 'Initialization provenance'):
            study.validate_model(model, self.protocol, 7)

    def test_invalid_rule_actions_fail_closed(self):
        settings = self.protocol['rule_candidates'][0]
        with patch.object(study, 'evaluate_policy', return_value={'invalid_actions': 1, 'label': settings['id']}):
            with self.assertRaisesRegex(RuntimeError, 'invalid actions'):
                study.evaluate_rule(self.scenario, study.VALIDATION_SEEDS, settings)
        wrong = deepcopy(settings)
        wrong['recharge_threshold'] = .99
        with self.assertRaisesRegex(ValueError, 'outside the frozen'):
            study.evaluate_rule(self.scenario, study.VALIDATION_SEEDS, wrong)


if __name__ == '__main__':
    unittest.main()
