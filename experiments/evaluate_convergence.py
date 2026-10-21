"""Evaluate frozen convergence-study models after all three training runs finish.

Default runs: outputs/convergence_study/seed_{7,19,37}.
Example: python experiments/evaluate_convergence.py
Optional paths: --run 7=/path/to/run --run 19=/path/to/run --run 37=/path/to/run

The shared test set is never used for checkpoint selection. Bootstrap intervals
resample paired environment seeds, conditional on the three trained models;
they do not claim uncertainty over arbitrary future training random seeds.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from charging_env.rl.trainer import Trainer

TRAINING_SEEDS = (7, 19, 37)
TEST_SEEDS = [500001+2*i for i in range(256)]
ACTION_SEED = 70321
BOOTSTRAP_SEED = 852901
BOOTSTRAP_RESAMPLES = 5000
MODES = ('deterministic', 'stochastic')
VARIANTS = ('initial', 'checkpoint_250', 'validation_selected', 'final')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profits(evaluation):
    if evaluation['seeds'] != TEST_SEEDS:
        raise ValueError('Evaluation seed list differs from the fixed final test protocol')
    if evaluation['action_seed'] != ACTION_SEED:
        raise ValueError('Evaluation action seed differs from the fixed final test protocol')
    if [e['seed'] for e in evaluation['episodes']] != TEST_SEEDS:
        raise ValueError('Episode ordering differs from the fixed final test protocol')
    return np.asarray([e['discounted_profit'] for e in evaluation['episodes']], dtype=float)


def paired_summary(differences, indices):
    """Rows are fixed trained models; columns are paired environment seeds."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim == 1:
        differences = differences[None, :]
    environment_means = differences.mean(axis=0)
    bootstrap_means = environment_means[indices].mean(axis=1)
    training_seed_means = differences.mean(axis=1)
    return {
        'mean_difference': float(environment_means.mean()),
        'paired_environment_bootstrap_95_percent_ci': [float(x) for x in np.quantile(bootstrap_means, [.025, .975])],
        'training_seed_mean_differences': [float(x) for x in training_seed_means],
        'training_seed_sd_of_mean_difference': float(training_seed_means.std(ddof=1)) if len(training_seed_means)>1 else None,
        'environment_count': differences.shape[1],
        'fixed_model_count': differences.shape[0],
    }


def summarize(reports, control, protocol):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(TEST_SEEDS), size=(BOOTSTRAP_RESAMPLES, len(TEST_SEEDS)))
    aggregate = {'protocol': protocol,
        'interval_scope': 'Paired environment-seed bootstrap, conditional on these trained models; not a confidence interval over all possible training seeds.',
        'means': {}, 'comparisons': {}, 'per_training_seed': {}}
    for mode in MODES:
        values = {variant: np.stack([profits(reports[seed]['variants'][variant][mode]) for seed in TRAINING_SEEDS])
                  for variant in VARIANTS}
        control_values = profits(control['variants']['original_final_1000'][mode])
        aggregate['means'][mode] = {}
        for variant, array in values.items():
            seed_means = array.mean(axis=1)
            aggregate['means'][mode][variant] = {
                'mean_discounted_profit': float(array.mean()),
                'mean_by_training_seed': {str(seed): float(mean) for seed, mean in zip(TRAINING_SEEDS, seed_means)},
                'training_seed_sd': float(seed_means.std(ddof=1)),
                'training_physical_periods_by_seed': {str(seed): reports[seed]['variants'][variant].get('training_physical_periods') for seed in TRAINING_SEEDS},
            }
        aggregate['means'][mode]['original_final_1000'] = {
            'mean_discounted_profit': float(control_values.mean()), 'training_seed': 7,
            'training_physical_periods': control['variants']['original_final_1000'].get('training_physical_periods')}
        comparisons = {}
        for variant in VARIANTS[1:]:
            comparisons[f'{variant}_minus_initial'] = values[variant]-values['initial']
            comparisons[f'{variant}_minus_original_final_1000'] = values[variant]-control_values[None, :]
        comparisons['final_minus_checkpoint_250'] = values['final']-values['checkpoint_250']
        comparisons['validation_selected_minus_final'] = values['validation_selected']-values['final']
        aggregate['comparisons'][mode] = {name: paired_summary(delta, indices) for name, delta in comparisons.items()}
        for row, seed in enumerate(TRAINING_SEEDS):
            aggregate['per_training_seed'].setdefault(str(seed), {})[mode] = {
                name: paired_summary(delta[row], indices) for name, delta in comparisons.items()}
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT/'outputs/convergence_study')
    parser.add_argument('--run', action='append', metavar='SEED=PATH', help='Override all three run paths using one argument per seed')
    parser.add_argument('--expected-updates', type=int, default=500)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace previous deterministic final-test artifacts')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    runs = {seed: study/f'seed_{seed}' for seed in TRAINING_SEEDS}
    if args.run:
        runs = {}
        for spec in args.run:
            seed_text, separator, path = spec.partition('=')
            if not separator or int(seed_text) in runs:
                parser.error('--run must contain distinct SEED=PATH arguments')
            runs[int(seed_text)] = Path(path).resolve()
        if set(runs) != set(TRAINING_SEEDS):
            parser.error('--run must specify exactly seeds 7, 19, and 37')
    # Gate every candidate read/evaluation on completion of all three runs.
    missing = [str(path/'summary.json') for path in runs.values() if not (path/'summary.json').is_file()]
    if missing:
        parser.error('All training runs must finish before any candidate test evaluation. Missing: '+', '.join(missing))
    control_path = study/'control_test.json'
    if not control_path.is_file():
        parser.error('Missing frozen control evaluation: '+str(control_path))
    targets = [study/f'candidate_seed_{seed}_test.json' for seed in TRAINING_SEEDS]
    aggregate_path = study/'candidate_test_aggregate.json'
    targets.append(aggregate_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        parser.error('Final test artifacts already exist; use --overwrite explicitly: '+', '.join(existing))
    control = json.loads(control_path.read_text())
    for mode in MODES:
        profits(control['variants']['original_final_1000'][mode])
    protocol = {'training_seeds': list(TRAINING_SEEDS), 'environment_seeds': TEST_SEEDS,
                'action_seed': ACTION_SEED, 'bootstrap_seed': BOOTSTRAP_SEED,
                'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
                'bootstrap_unit': 'paired environment seed, averaging fixed training-seed models before resampling',
                'checkpoint_selection': 'best_checkpoint.pt selected by training validation only; final tests never select models',
                'expected_final_updates': args.expected_updates,
                'control_test_sha256': digest(control_path),
                'evaluation_source_sha256': digest(ROOT/'src/charging_env/rl/trainer.py')}
    reports = {}
    started = time.monotonic()
    for seed in TRAINING_SEEDS:
        run = runs[seed]
        paths = {'checkpoint_250': run/'checkpoint_250.pt',
                 'validation_selected': run/'best_checkpoint.pt', 'final': run/'checkpoint.pt'}
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            parser.error('Missing required frozen checkpoints: '+', '.join(absent))
        final = Trainer.load_checkpoint(paths['final'])
        if final.config.seed != seed or final.update_count != args.expected_updates:
            raise ValueError(f'Seed {seed}: final checkpoint seed/update count does not match the completed study')
        reports[seed] = {'protocol': protocol, 'training_seed': seed, 'run_directory': str(run),
                         'final_training_config': asdict(final.config), 'variants': {}}
        for variant in VARIANTS:
            if variant == 'initial':
                trainer = Trainer(final.scenario, final.config)
                metadata = {'construction': 'Fresh weights with final checkpoint scenario and training config', 'checkpoint': None}
            else:
                trainer = Trainer.load_checkpoint(paths[variant])
                if trainer.config.seed != seed or trainer.scenario != final.scenario:
                    raise ValueError(f'Seed/scenario mismatch in {paths[variant]}')
                if variant == 'checkpoint_250' and trainer.update_count != 250:
                    raise ValueError('checkpoint_250.pt must contain exactly update 250')
                if not 0 <= trainer.update_count <= args.expected_updates:
                    raise ValueError('Checkpoint update count exceeds the fixed budget')
                metadata = {'checkpoint': str(paths[variant]), 'checkpoint_sha256': digest(paths[variant])}
            rng_state = torch.get_rng_state().clone()
            evaluation = {mode: trainer.evaluate(TEST_SEEDS, deterministic=(mode=='deterministic'), action_seed=ACTION_SEED)
                          for mode in MODES}
            if not torch.equal(rng_state, torch.get_rng_state()):
                raise AssertionError('Evaluation modified global Torch RNG')
            for mode in MODES:
                profits(evaluation[mode])
            reports[seed]['variants'][variant] = dict(metadata, **evaluation,
                training_updates=trainer.update_count, training_episodes=trainer.episode_count,
                training_physical_periods=trainer.environment_periods, rng_preserved=True)
            print(f'seed={seed} variant={variant} '+json.dumps({mode: evaluation[mode]['mean_discounted_profit'] for mode in MODES}), flush=True)
        write_json(study/f'candidate_seed_{seed}_test.json', reports[seed])
    aggregate = summarize(reports, control, protocol)
    aggregate['elapsed_seconds'] = time.monotonic()-started
    write_json(aggregate_path, aggregate)
    print('Saved '+str(aggregate_path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
