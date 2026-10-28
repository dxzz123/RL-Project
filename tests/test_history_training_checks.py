"""Saved training checks reject semantically altered, internally resealed data."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from history_training_checks import check_training_artifacts
from run_priced_adp_study import make_backend
from charging_env.config import Scenario
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PriceSearchConfig, fit_price_policy
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer


MODEL_HASH = 'a'*64
SETTING = dict(early_shadow_price=.3, late_shadow_price=.15)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def factory_for(scenario, phase):
    return lambda actor, seed: make_backend(scenario, phase, actor, seed, SETTING)


def complete(directory, phase, config):
    write(directory/'completed.json', dict(binding=dict(model_sha256=MODEL_HASH,
        phase=phase, seed=config.seed, config=asdict(config),
        setting=SETTING if phase == 'inventory' else None), files={}))


class HistoryTrainingChecksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.scenario = replace(Scenario(), horizon=2, arrival_profile=(1.,)*2,
            grid_buy_per_kwh=(.15,)*2, grid_sell_per_kwh=(.1,)*2,
            charging_utility_intercept=3., terminal_value_per_stored_kwh=.04).validate()
        cls.configs = {}
        with redirect_stdout(io.StringIO()):
            for phase in ('warm', 'rollout', 'inventory'):
                config = PriceSearchConfig(seed=7, generations=2 if phase == 'warm' else 1,
                    population=10 if phase == 'warm' else 4, elite_count=2,
                    episodes_per_generation=2, training_seed_base=4400000000+100*len(cls.configs),
                    fixed_anchor_indices=None if phase == 'warm' else ())
                cls.configs[phase] = config
                initial = None if phase == 'warm' else LearnedPricingPolicy.load_checkpoint(cls.root/'warm/checkpoint.npz')
                fit_price_policy(cls.scenario, config, factory_for(cls.scenario, phase), cls.root/phase, initial)
                complete(cls.root/phase, phase, config)
            config = PaperPPOConfig(seed=7, updates=2, episodes_per_update=2, epochs=1,
                minibatch_size=8, hidden_size=16, torch_threads=1, reward_scale=20.,
                warmstart_episodes=2, warmstart_epochs=2, warmstart_minibatch_size=4,
                warmstart_dispatch_solver='coordinate', warmstart_coordinate_sweeps=2,
                eval_episodes=2, eval_interval=1, eval_seed_base=4400001001,
                training_seed_base=2200002000, warmstart_seed_base=2200003000).validate()
            cls.configs['ppo'] = config
            PaperPPOTrainer(cls.scenario, config).train(cls.root/'ppo')
            complete(cls.root/'ppo', 'ppo', config)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def check(self, phase, directory=None, **overrides):
        values = dict(directory=self.root/phase if directory is None else directory,
            scenario=self.scenario, phase=phase, seed=7, config=self.configs[phase],
            model_sha256=MODEL_HASH, setting=SETTING, expected_factory_sha256=digest(__file__),
            warm_checkpoint=self.root/'warm/checkpoint.npz' if phase in ('rollout', 'inventory') else None)
        values.update(overrides)
        check_training_artifacts(**values)

    def altered(self, phase):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)/phase; shutil.copytree(self.root/phase, directory)
        return directory

    def repair_cem_summary_hash(self, directory, filename):
        summary = read(directory/'summary.json')
        summary[{'training_history.json': 'training_history_sha256', 'protocol.json': 'protocol_sha256'}[filename]] = digest(directory/filename)
        write(directory/'summary.json', summary)

    def test_real_small_training_all_four_phases_pass_without_evaluation(self):
        for phase in self.configs:
            with self.subTest(phase=phase): self.check(phase)

    def test_cem_rejects_changed_proposal_after_history_hash_is_repaired(self):
        directory = self.altered('warm')
        history = read(directory/'training_history.json')
        history[0]['candidates'][-1]['coefficients'][1] += .01
        write(directory/'training_history.json', history)
        self.repair_cem_summary_hash(directory, 'training_history.json')
        with self.assertRaisesRegex(ValueError, 'exact proposal'): self.check('warm', directory)

    def test_cem_rejects_edited_elite_update_and_means(self):
        for key in ('search_std', 'search_mean'):
            directory = self.altered('warm'); history = read(directory/'training_history.json')
            history[0][key][0] += .01
            write(directory/'training_history.json', history)
            self.repair_cem_summary_hash(directory, 'training_history.json')
            with self.assertRaisesRegex(ValueError, 'CEM elite'): self.check('warm', directory)
        directory = self.altered('warm'); history = read(directory/'training_history.json')
        history[0]['candidates'][0]['mean_discounted_profit'] += 1.
        write(directory/'training_history.json', history)
        self.repair_cem_summary_hash(directory, 'training_history.json')
        with self.assertRaisesRegex(ValueError, 'candidate mean'): self.check('warm', directory)

    def test_cem_rejects_factory_source_seed_and_model_binding_changes(self):
        with self.assertRaisesRegex(ValueError, 'factory'):
            self.check('warm', expected_factory_sha256='b'*64)
        with self.assertRaisesRegex(ValueError, 'model/backend binding'):
            self.check('warm', model_sha256='b'*64)
        directory = self.altered('warm'); history = read(directory/'training_history.json')
        history[0]['environment_seeds'][0] += 2
        write(directory/'training_history.json', history)
        self.repair_cem_summary_hash(directory, 'training_history.json')
        with self.assertRaisesRegex(ValueError, 'generation bank'): self.check('warm', directory)

    def test_refinement_rejects_wrong_warm_checkpoint_and_invalid_actions(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)/'wrong.npz'
        original = LearnedPricingPolicy.load_checkpoint(self.root/'warm/checkpoint.npz')
        coefficients = original.coefficients.copy(); coefficients[1] += .01
        LearnedPricingPolicy(self.scenario, coefficients).save_checkpoint(path)
        with self.assertRaisesRegex(ValueError, 'initialization binding'):
            self.check('inventory', warm_checkpoint=path)
        directory = self.altered('rollout'); history = read(directory/'training_history.json')
        history[0]['candidates'][0]['invalid_actions'] = 1
        write(directory/'training_history.json', history)
        self.repair_cem_summary_hash(directory, 'training_history.json')
        with self.assertRaisesRegex(ValueError, 'invalid actions'): self.check('rollout', directory)

    def test_ppo_rejects_wrong_best_checkpoint_scenario(self):
        directory = self.altered('ppo'); path = directory/'best_checkpoint.pt'
        data = torch.load(path, weights_only=True)
        data['scenario']['charging_utility_intercept'] += .1
        torch.save(data, path)
        with self.assertRaisesRegex(ValueError, 'best checkpoint uses different'):
            self.check('ppo', directory)

    def test_ppo_rejects_incomplete_metrics_and_false_demo_budget(self):
        directory = self.altered('ppo'); path = directory/'training_metrics.csv'
        lines = path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaisesRegex(ValueError, 'metrics are incomplete'): self.check('ppo', directory)
        directory = self.altered('ppo'); warm = read(directory/'warmstart_summary.json')
        warm['demonstration_episodes'] += 1; write(directory/'warmstart_summary.json', warm)
        with self.assertRaisesRegex(ValueError, 'warm-start provenance'): self.check('ppo', directory)

    def test_ppo_rejects_inconsistent_profit_ledger_and_monitor_bank(self):
        directory = self.altered('ppo'); result = read(directory/'initial_evaluation.json')
        result['episodes'][0]['discounted_reward_terms']['procurement_cost'] += 1.
        write(directory/'initial_evaluation.json', result)
        with self.assertRaisesRegex(ValueError, 'reward ledger'): self.check('ppo', directory)
        directory = self.altered('ppo'); result = read(directory/'final_evaluation.json')
        result['action_seed'] += 1; write(directory/'final_evaluation.json', result)
        with self.assertRaisesRegex(ValueError, 'monitor metadata'): self.check('ppo', directory)

    def test_ppo_all_profit_ties_keep_initial_untrained_monitor_best(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)/'ppo'
        c = replace(self.scenario, arrival_rates=((0.,)*4,)*2,
            grid_buy_per_kwh=(0.,)*2, grid_sell_per_kwh=(0.,)*2,
            fcs_discharge_price_profile=((0., 0.),)*2, operating_cost_per_mcv_period=0.,
            travel_cost_per_km=0., terminal_value_per_stored_kwh=0.)
        config = replace(self.configs['ppo'], updates=1, episodes_per_update=1,
            eval_episodes=1, warmstart_episodes=1, warmstart_epochs=1)
        with redirect_stdout(io.StringIO()): PaperPPOTrainer(c, config).train(directory)
        complete(directory, 'ppo', config)
        best = PaperPPOTrainer.load_checkpoint(directory/'best_checkpoint.pt')
        self.assertEqual(best.update_count, 0)
        self.assertEqual(best.warmstart_metadata, dict(completed=False, summary=None))
        self.check('ppo', directory, scenario=c, config=config)


if __name__ == '__main__': unittest.main()
