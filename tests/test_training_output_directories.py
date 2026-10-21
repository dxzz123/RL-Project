"""Exercise automatic run directories through real, tiny training CLI calls."""
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from charging_env.config import Scenario

try:
    import torch
except ImportError:
    torch = None


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@unittest.skipIf(torch is None, "Install the train extra to test PPO/MAPPO")
class TrainingOutputDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scenario_file = self.root / 'scenario.json'
        replace(Scenario(), horizon=1, arrival_profile=(1.,),
                grid_buy_per_kwh=(0.12,), grid_sell_per_kwh=(0.09,)).validate().to_json(self.scenario_file)
        self.small_run = [
            '--updates', '1', '--episodes-per-update', '1', '--epochs', '1',
            '--eval-episodes', '1', '--hidden-size', '8', '--torch-threads', '1',
        ]

    def run_cli(self, *arguments, resume=False):
        from charging_env.rl.train import main

        output = io.StringIO()
        scenario = [] if resume else ['--scenario', str(self.scenario_file)]
        with redirect_stdout(output):
            result = main(scenario + self.small_run + list(arguments))
        self.assertEqual(result, 0)
        return output.getvalue()

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*') if path.is_file()}

    def training_rows(self, directory):
        with (directory / 'training_metrics.csv').open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def summary(self, directory):
        return json.loads((directory / 'summary.json').read_text(encoding='utf-8'))

    def assert_announced(self, output, directory):
        paths = [Path(line.removeprefix('Training outputs: '))
                 for line in output.splitlines() if line.startswith('Training outputs: ')]
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].is_absolute())
        self.assertEqual(paths[0].resolve(), directory.resolve())

    def test_repeated_explicit_path_preserves_runs_and_skips_occupied_siblings(self):
        requested = self.root / 'experiment'
        first_output = self.run_cli('--output-dir', str(requested), '--seed', '7')
        original_files = self.snapshot(requested)
        occupied_file = self.root / 'experiment_2'
        occupied_file.write_text('Keep this unrelated file.\n', encoding='utf-8')
        occupied_directory = self.root / 'experiment_3'
        occupied_directory.mkdir()

        second_output = self.run_cli('--output-dir', str(requested), '--seed', '19')
        second = self.root / 'experiment_4'

        self.assert_announced(first_output, requested)
        self.assert_announced(second_output, second)
        self.assertEqual(self.snapshot(requested), original_files)
        self.assertEqual(occupied_file.read_text(encoding='utf-8'), 'Keep this unrelated file.\n')
        self.assertEqual(list(occupied_directory.iterdir()), [])
        for directory, seed in ((requested, 7), (second, 19)):
            config = json.loads((directory / 'training_config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['seed'], seed)
            self.assertEqual([int(row['update']) for row in self.training_rows(directory)], [1])
            self.assertEqual(self.summary(directory)['total_updates'], 1)
            self.assertTrue((directory / 'checkpoint.pt').is_file())
        self.assertNotEqual((requested / 'checkpoint.pt').read_bytes(),
                            (second / 'checkpoint.pt').read_bytes())

    def test_repeated_default_command_creates_independent_runs(self):
        with working_directory(self.root):
            first_output = self.run_cli('--seed', '7')
            first = self.root / 'outputs' / 'sa_ppo_mappo'
            original_files = self.snapshot(first)
            second_output = self.run_cli('--seed', '19')

        second = self.root / 'outputs' / 'sa_ppo_mappo_2'
        self.assert_announced(first_output, first)
        self.assert_announced(second_output, second)
        self.assertEqual(self.snapshot(first), original_files)
        self.assertEqual([int(row['update']) for row in self.training_rows(second)], [1])
        self.assertEqual(self.summary(second)['training_physical_periods'], 1)
        self.assertTrue(self.summary(second)['checkpoint_reload_matches'])

    def test_resume_to_occupied_path_continues_in_new_run_without_changing_source(self):
        from charging_env.rl.trainer import Trainer

        requested = self.root / 'resumed_experiment'
        self.run_cli('--output-dir', str(requested), '--seed', '7')
        original_files = self.snapshot(requested)

        resumed_output = self.run_cli(
            '--resume', str(requested / 'checkpoint.pt'),
            '--output-dir', str(requested), resume=True,
        )

        resumed = self.root / 'resumed_experiment_2'
        self.assert_announced(resumed_output, resumed)
        self.assertEqual(self.snapshot(requested), original_files)
        self.assertEqual([int(row['update']) for row in self.training_rows(resumed)], [2])
        summary = self.summary(resumed)
        self.assertEqual(summary['updates_this_run'], 1)
        self.assertEqual(summary['total_updates'], 2)
        self.assertEqual(summary['training_episodes'], 2)
        self.assertEqual(summary['training_physical_periods'], 2)
        self.assertTrue(summary['checkpoint_reload_matches'])
        loaded = Trainer.load_checkpoint(resumed / 'checkpoint.pt')
        self.assertEqual(loaded.update_count, 2)
        self.assertEqual(loaded.episode_count, 2)
        self.assertEqual(loaded.environment_periods, 2)


if __name__ == '__main__':
    unittest.main()
