"""Freeze rule choices on validation, then compare frozen RL on fresh test scenes.

Run: python experiments/run_policy_benchmarks.py
Writes a new directory; never changes environment parameters, policies or weights.
"""
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch

from charging_env.config import Scenario
from charging_env.simulator import DispatchAction, Mode
from charging_env.benchmarks import policies as rules
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.benchmarks.learned import evaluate_checkpoint

TRAINING_SEEDS = (7, 19, 37)
ACTION_SEED = 54123


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(config):
    result = [{'id': 'random', 'family': 'random'}]
    for price in range(len(config.price_levels)):
        result.append({'id': f'wait_p{price}', 'family': 'fixed_wait', 'price_index': price})
        for threshold in (.2, .4, .6):
            result.append({'id': f'service_p{price}_b{threshold}', 'family': 'fixed_service',
                           'price_index': price, 'recharge_threshold': threshold})
        result.append({'id': f'myopic_p{price}', 'family': 'fixed_myopic',
                       'price_index': price, 'shadow_price': 0.})
        for shadow in (.1, .2, .3):
            result.append({'id': f'inventory_p{price}_s{shadow}', 'family': 'fixed_inventory',
                           'price_index': price, 'shadow_price': shadow})
    for shadow in (0., .1, .2, .3):
        result.append({'id': f'adaptive_s{shadow}', 'family': 'adaptive_inventory', 'shadow_price': shadow})
    return result


def evaluate_rule(config, seeds, spec, label=None, *, action_seed=ACTION_SEED):
    family = spec['family']
    if family == 'random':
        pricing = lambda obs, rng: rules.random_prices(obs, config, rng)
        dispatch = lambda obs, rng: rules.random_dispatch(obs, config, rng)
    else:
        pricing = ((lambda obs, rng: rules.adaptive_prices(obs, config)) if family == 'adaptive_inventory'
                   else (lambda obs, rng: rules.fixed_prices(obs, config, spec['price_index'])))
        if family == 'fixed_wait':
            dispatch = lambda obs, rng: [DispatchAction(Mode.WAIT, int(z)) for z in obs['locations']]
        elif family == 'fixed_service':
            dispatch = lambda obs, rng: rules.service_first_dispatch(obs, config, spec['recharge_threshold'])
        else:
            dispatch = lambda obs, rng: rules.coordinated_dispatch(obs, config, spec['shadow_price'])
    result = evaluate_policy(config, seeds, pricing, dispatch, action_seed=action_seed,
                             label=label or spec['id'])
    result['policy_specification'] = spec
    if result['invalid_actions']:
        raise AssertionError('Benchmark emitted invalid actions: '+result['label'])
    return result


def average_trained_models(evaluations, label):
    """Summaries average three fixed models; no ensemble policy is executed.

    Per-environment average profit is retained only for paired uncertainty.
    Tail metrics average the actual per-model tails, not a fictitious averaged
    trajectory's smaller tail risk.
    """
    seeds = evaluations[0]['evaluation_seeds']
    if any(e['evaluation_seeds'] != seeds for e in evaluations):
        raise ValueError('Model average requires identical environment order')
    result = {'label': label, 'evaluation_seeds': seeds, 'n_episodes': len(seeds),
              'aggregation': 'Mean over three separately executed frozen training-seed models; not an ensemble policy.',
              'training_seeds': [e['training_seed'] for e in evaluations],
              'mean_by_training_seed': {str(e['training_seed']): e['mean_discounted_profit'] for e in evaluations},
              'training_seed_sd': float(np.std([e['mean_discounted_profit'] for e in evaluations], ddof=1)),
              'episodes': [{'seed': seed, 'discounted_profit': float(np.mean([
                  e['episodes'][i]['discounted_profit'] for e in evaluations]))} for i, seed in enumerate(seeds)]}
    scalar_fields = ['mean_discounted_profit', 'mean_profit', 'profit_p10', 'lower_tail_mean_10pct',
                    'loss_probability', 'mean_served_request_share', 'mean_outside_option_share',
                    'pooled_selected_service_fulfillment', 'mean_mobile_delivered_kwh',
                    'mean_fixed_delivered_kwh', 'mean_final_inventory_kwh', 'mean_export_grid_kwh',
                    'mean_charge_grid_kwh', 'mean_mobile_price']
    for key in scalar_fields:
        result[key] = float(np.mean([e[key] for e in evaluations]))
    for key in ('mean_reward_terms', 'mean_discounted_reward_terms', 'mean_policy_decision_ms'):
        result[key] = {term: float(np.mean([e[key][term] for e in evaluations])) for term in evaluations[0][key]}
    result['invalid_actions'] = sum(e['invalid_actions'] for e in evaluations)
    return result


def table_row(name, evaluation):
    return {'policy': name, 'mean_discounted_profit': evaluation['mean_discounted_profit'],
            'training_seed_sd': evaluation.get('training_seed_sd'),
            'mean_served_request_share': evaluation['mean_served_request_share'],
            'mean_outside_option_share': evaluation['mean_outside_option_share'],
            'selected_service_fulfillment': evaluation['pooled_selected_service_fulfillment'],
            'profit_p10': evaluation['profit_p10'], 'lower_tail_mean_10pct': evaluation['lower_tail_mean_10pct'],
            'loss_probability': evaluation['loss_probability'],
            'mean_mobile_delivered_kwh': evaluation['mean_mobile_delivered_kwh'],
            'mean_final_inventory_kwh': evaluation['mean_final_inventory_kwh'],
            'pricing_ms': evaluation['mean_policy_decision_ms']['pricing'],
            'dispatch_ms': evaluation['mean_policy_decision_ms']['dispatch'],
            'invalid_actions': evaluation['invalid_actions']}


def checkpoint_scenario(path):
    """Read validated base-scenario metadata without constructing a trainer.

    Missing fields retain Scenario defaults, including the legacy utility
    intercept of zero. New utility parameters must match training exactly.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    return Scenario(**{key:tuples(value) for key,value in payload['scenario'].items()}).validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/policy_benchmarks')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT/'outputs/convergence_study')
    parser.add_argument('--scenario', type=Path, help='Scenario JSON; all six checkpoints must match its base configuration')
    parser.add_argument('--validation-episodes', type=int, default=64)
    parser.add_argument('--test-episodes', type=int, default=256)
    parser.add_argument('--stress-episodes', type=int, default=128)
    parser.add_argument('--validation-seed-base', type=int, default=700001)
    parser.add_argument('--test-seed-base', type=int, default=900001)
    parser.add_argument('--stress-seed-base', type=int, default=1100001)
    parser.add_argument('--action-seed', type=int, default=ACTION_SEED)
    args = parser.parse_args(argv)
    if min(args.validation_episodes, args.test_episodes, args.stress_episodes) < 2:
        parser.error('Use at least two episodes per split')
    for name in ('validation_seed_base', 'test_seed_base', 'stress_seed_base'):
        value = getattr(args, name)
        if value < 0 or value % 2 != 1:
            parser.error(name.replace('_','-')+' must be a nonnegative odd integer')
    if args.action_seed < 0:
        parser.error('action-seed must be nonnegative')
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Choose a new output directory; existing benchmark evidence is not overwritten')
    try:
        config = Scenario.from_json(args.scenario) if args.scenario else Scenario().validate()
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    paths = {f'rl_{variant}_{seed}': args.checkpoint_root/f'seed_{seed}'/filename
             for seed in TRAINING_SEEDS for variant, filename in
             [('final', 'checkpoint.pt'), ('validation_selected', 'best_checkpoint.pt')]}
    if any(not p.is_file() for p in paths.values()):
        parser.error('All six frozen trained checkpoints must exist before benchmarking')
    for name, path in paths.items():
        try:
            trained = checkpoint_scenario(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(f'Cannot read checkpoint scenario for {name}: {error}')
        differences = [field for field in config.__dataclass_fields__
                       if getattr(trained, field) != getattr(config, field)]
        if differences:
            details = ', '.join(f'{field} (checkpoint={getattr(trained, field)!r}, selected={getattr(config, field)!r})'
                                for field in differences)
            parser.error(f'Checkpoint {name} does not match the selected base scenario: {details}. '
                         'Train matching checkpoints before benchmarking; rule search has not started.')
    hashes = {key: digest(p) for key, p in paths.items()}
    val_seeds = [args.validation_seed_base+2*i for i in range(args.validation_episodes)]
    test_seeds = [args.test_seed_base+2*i for i in range(args.test_episodes)]
    stress_seeds = [args.stress_seed_base+2*i for i in range(args.stress_episodes)]
    if (set(val_seeds)&set(test_seeds) or set(test_seeds)&set(stress_seeds)
            or set(val_seeds)&set(stress_seeds)):
        parser.error('Requested validation, test, and stress seed splits must be disjoint')
    specs = candidates(config)
    started = time.monotonic()
    output.mkdir(parents=True)
    protocol = {'scenario': asdict(config), 'validation_seeds': val_seeds, 'test_seeds': test_seeds,
                'stress_seeds': stress_seeds, 'action_seed': args.action_seed, 'training_seeds': list(TRAINING_SEEDS),
                'primary_learned_policy': 'Mean of all three precommitted final stochastic policies',
                'secondary_learned_policies': 'Final deterministic; historical validation-selected in both modes',
                'selection': 'Maximum validation discounted profit within each rule family; strongest nonrandom family selected on validation only. No training or test-driven changes.',
                'rule_candidates': specs, 'energy_grid': [0.5, 1.0],
                'checkpoint_sha256': hashes,
                'checkpoint_paths': {k: str(v.resolve()) for k,v in paths.items()},
                'stress_cases': {'high_demand': 'arrival means x1.5', 'low_inventory': '35 kWh per MCV'},
                'model_information': 'Rules use only current public observations. Coordinated rules additionally use exact known physical transition/reward models; no future realized demand/choice/RNG access.',
                'interval_scope': 'Paired environment bootstrap conditional on fixed models, with training-seed spread separately reported.',
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for directory in
                                  [ROOT/'src/charging_env', ROOT/'experiments'] for p in directory.rglob('*.py')}}
    write_json(output/'protocol.json', protocol)
    selections, validation = {}, []
    for spec in specs:
        result = evaluate_rule(config, val_seeds, spec, action_seed=args.action_seed)
        write_json(output/'validation'/f"{spec['id']}.json", result)
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        validation.append(row)
        old = selections.get(spec['family'])
        if old is None or row['mean_discounted_profit'] > old['mean_discounted_profit']:
            selections[spec['family']] = row
        print(f"validation {spec['id']} profit={result['mean_discounted_profit']:.3f}", flush=True)
    strongest = max((v for k,v in selections.items() if k != 'random'), key=lambda x: x['mean_discounted_profit'])['family']
    write_json(output/'selection.json', {'family_winners': selections, 'strongest_validation_family': strongest,
                                        'all_candidates': validation})
    print('Frozen strongest validation rule: '+strongest, flush=True)
    cases = {'base': (config, test_seeds),
             'high_demand': (replace(config, mean_requests_by_zone=tuple(1.5*x for x in config.mean_requests_by_zone)), stress_seeds),
             'low_inventory': (replace(config, initial_energy_kwh=tuple(35. for _ in config.initial_energy_kwh)), stress_seeds)}
    complete = {}
    for case, (scenario, seeds) in cases.items():
        results, averages = {}, {}
        for family, spec in selections.items():
            result = evaluate_rule(scenario, seeds, spec, label=family, action_seed=args.action_seed)
            results[family] = result
            write_json(output/case/f'{family}.json', result)
            print(f"test {case}/{family} profit={result['mean_discounted_profit']:.3f}", flush=True)
        variants = ('final', 'validation_selected') if case == 'base' else ('final',)
        for variant in variants:
            for deterministic in (False, True):
                mode = 'deterministic' if deterministic else 'stochastic'
                models = []
                for seed in TRAINING_SEEDS:
                    label = f'rl_{variant}_{mode}_{seed}'
                    result = evaluate_checkpoint(scenario, seeds, paths[f'rl_{variant}_{seed}'],
                        deterministic=deterministic, action_seed=args.action_seed, label=label)
                    if result['invalid_actions']:
                        raise AssertionError('Learned checkpoint emitted invalid actions')
                    models.append(result)
                    results[label] = result
                    write_json(output/case/f'{label}.json', result)
                    print(f"test {case}/{label} profit={result['mean_discounted_profit']:.3f}", flush=True)
                key = f'rl_{variant}_{mode}_mean'
                averages[key] = average_trained_models(models, key)
        all_results = dict(results, **averages)
        comparisons = {key: paired_comparison(value, results[strongest])
                       for key, value in all_results.items() if key != strongest}
        table = [table_row(key, result) for key, result in all_results.items()
                 if not key.startswith('rl_') or key.endswith('_mean')]
        with (output/case/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
        complete[case] = {'scenario': asdict(scenario), 'table': table,
                          'model_averages': averages, 'comparisons_to_validation_best_rule': comparisons,
                          'strongest_validation_rule': strongest,
                          'invalid_actions_all_evaluations': sum(e['invalid_actions'] for e in results.values())}
        write_json(output/case/'summary.json', complete[case])
    if any(digest(paths[key]) != value for key, value in hashes.items()):
        raise RuntimeError('A frozen checkpoint changed during evaluation')
    write_json(output/'summary.json', {'cases': complete, 'selection': selections,
                                      'strongest_validation_rule': strongest,
                                      'elapsed_seconds': time.monotonic()-started,
                                      'test_tuning_performed': False})
    print('Complete: '+str(output/'summary.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
