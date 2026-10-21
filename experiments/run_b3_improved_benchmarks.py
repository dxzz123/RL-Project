"""Select b=3 rules on development validation, then compare frozen old/new models.

The selection phase requires no checkpoints and never accesses test scenes.
Evaluation requires all three 500-update final and validation-best new models,
and the three frozen prior SiLU+b=3 final models. No test-based model selection
is performed. Reported profit is the original environment business reward.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from charging_env.config import Scenario
from charging_env.rl.trainer import TrainingConfig
from charging_env.benchmarks.learned import evaluate_checkpoint
from charging_env.benchmarks.evaluation import paired_comparison
from run_policy_benchmarks import (
    TRAINING_SEEDS, average_trained_models, candidates, digest, evaluate_rule,
    table_row, write_json,
)

VALIDATION_SEEDS = [4500001 + 2*i for i in range(128)]
TEST_SEEDS = [4700001 + 2*i for i in range(512)]
HIGH_DEMAND_SEEDS = [4900001 + 2*i for i in range(256)]
LOW_INVENTORY_SEEDS = [5100001 + 2*i for i in range(256)]
ACTION_SEED = 459991
PRIMARY = 'rl_final_stochastic_mean'
PREVIOUS = 'previous_final_stochastic_mean'
RULE_FAMILIES = {'random', 'fixed_wait', 'fixed_service', 'fixed_myopic',
                 'fixed_inventory', 'adaptive_inventory'}
SUCCESS_CRITERIA = {
    'consistent_rule_advantage': 'All three new final stochastic model means exceed the validation-selected strongest rule, and the primary paired 95% interval lower bound exceeds zero.',
    'consistent_previous_improvement': 'All three same-seed new-final-minus-previous-final stochastic means are positive, and the mean-model paired 95% interval lower bound exceeds zero.',
    'scope': 'These are predeclared empirical criteria on fixed models, not a proof of convergence or superiority under retraining. Rule advantage and improvement over previous models are separate criteria.',
}


def scenario_from_data(data):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    return Scenario(**{key: tuples(value) for key, value in data.items()}).validate()


def source_hashes(*, selection=False):
    paths = sorted((ROOT / 'src/charging_env').rglob('*.py'))
    if selection:
        # Training implementation may still be developed after rule selection.
        # These excluded modules do not execute any rule or environment logic.
        paths = [path for path in paths if 'rl' not in path.parts and path.name != 'learned.py']
    paths += [Path(__file__).resolve(), ROOT / 'experiments/run_policy_benchmarks.py']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_sources(hashes):
    changed = [path for path, value in hashes.items() if digest(ROOT / path) != value]
    if changed:
        raise RuntimeError('Frozen benchmark sources changed: ' + ', '.join(changed))


def load_study(study, scenario_path, study_protocol):
    scenario = Scenario.from_json(scenario_path)
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('This study requires the unchanged default physical b=3 scenario')
    data = json.loads(study_protocol.read_text())
    if 'scenario' in data and scenario_from_data(data['scenario']) != scenario:
        raise ValueError('Root study protocol scenario differs from the selected scenario')
    if 'training_seeds' in data and data['training_seeds'] != list(TRAINING_SEEDS):
        raise ValueError('Root study protocol must retain all three training seeds 7/19/37')
    provenance = {'study_protocol_path': str(study_protocol.resolve()),
                  'study_protocol_sha256': digest(study_protocol),
                  'scenario_path': str(scenario_path.resolve()),
                  'scenario_sha256': digest(scenario_path)}
    return scenario, data, provenance


def verify_provenance(provenance):
    for prefix in ('study_protocol', 'scenario'):
        if digest(Path(provenance[prefix + '_path'])) != provenance[prefix + '_sha256']:
            raise RuntimeError(f'Frozen {prefix} changed during execution')


def select_rules(study, scenario, provenance):
    output = study / 'rule_selection'
    if output.exists():
        raise ValueError('Rule-selection evidence already exists; it is never overwritten')
    specs = candidates(scenario)
    if len(specs) != 61 or {spec['family'] for spec in specs} != RULE_FAMILIES:
        raise ValueError('Expected the predeclared 61 candidates across six rule families')
    protocol = dict(provenance, scenario=asdict(scenario), validation_seeds=VALIDATION_SEEDS,
                    action_seed=ACTION_SEED, rule_candidates=specs,
                    source_sha256=source_hashes(selection=True), success_criteria=SUCCESS_CRITERIA,
                    selection='Maximize validation discounted profit within each family; select the strongest nonrandom family on validation only.',
                    study_protocol_scope='The root protocol is frozen during this phase. Training-bundle selection may subsequently finalize it without changing this scenario or rule-selection split.',
                    test_scenarios_accessed=False, validation_policy_episodes=61 * 128)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    protocol_hash, started = digest(output / 'protocol.json'), time.monotonic()
    winners, rows = {}, []
    for spec in specs:
        result = evaluate_rule(scenario, VALIDATION_SEEDS, spec, action_seed=ACTION_SEED)
        write_json(output / 'validation' / f'{spec["id"]}.json', result)
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        rows.append(row)
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
        print(f'validation {spec["id"]}: {result["mean_discounted_profit"]:.4f}', flush=True)
    strongest = max((row for name, row in winners.items() if name != 'random'),
                    key=lambda row: row['mean_discounted_profit'])['family']
    verify_sources(protocol['source_sha256'])
    verify_provenance(provenance)
    if digest(output / 'protocol.json') != protocol_hash:
        raise RuntimeError('Rule-selection protocol changed during selection')
    write_json(output / 'selection.json', {'family_winners': winners, 'strongest_validation_family': strongest,
        'all_candidates': rows, 'selection_protocol_sha256': protocol_hash,
        'elapsed_seconds': time.monotonic() - started, 'test_scenarios_accessed': False})
    print('Frozen strongest validation rule: ' + strongest, flush=True)


def old_training_config(seed):
    return TrainingConfig(seed=seed, updates=500, episodes_per_update=16, epochs=6,
        minibatch_size=192, learning_rate=3e-4, critic_learning_rate=1e-3,
        value_clip=0., entropy_coef=.005, eval_episodes=32, eval_interval=25,
        eval_seed_base=100001, eval_action_seed=12345, eval_policy='stochastic',
        discount_policy_loss=True, dispatch_final_activation='silu').validate()


def checkpoint_metadata(path, scenario, seed, variant, expected):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if scenario_from_data(payload['scenario']) != scenario:
        raise ValueError(f'{path}: checkpoint must match the exact b=3 base scenario')
    settings = TrainingConfig(**payload['training_config']).validate()
    mismatch = [key for key in expected.__dataclass_fields__ if getattr(settings, key) != getattr(expected, key)]
    if mismatch or settings.seed != seed:
        raise ValueError(f'{path}: training configuration differs from the frozen protocol: {mismatch}')
    if settings.updates != 500 or settings.episodes_per_update != 16 or settings.dispatch_final_activation != 'silu':
        raise ValueError(f'{path}: expected 500 updates, 16 episodes per update, and SiLU dispatch')
    updates, periods, episodes = (int(payload[key]) for key in ('update_count', 'environment_periods', 'episode_count'))
    if periods != updates * 16 * scenario.horizon or episodes != updates * 16:
        raise ValueError(f'{path}: inconsistent physical training-budget counters')
    if variant in ('final', 'previous_final') and (updates != 500 or periods != 96000):
        raise ValueError(f'{path}: final models require 500 updates and 96000 physical periods')
    if variant == 'validation_selected' and not 0 <= updates <= 500:
        raise ValueError(f'{path}: validation-selected checkpoint exceeds the training budget')
    if variant == 'warmstart' and (updates != 0 or periods != 0):
        raise ValueError(f'{path}: warm-start checkpoint must precede PPO rollouts')
    validation = {settings.eval_seed_base + 2*i for i in range(settings.eval_episodes)}
    held_out = set(VALIDATION_SEEDS + TEST_SEEDS + HIGH_DEMAND_SEEDS + LOW_INVENTORY_SEEDS)
    if validation & held_out:
        raise ValueError(f'{path}: checkpoint-selection scenarios overlap rule validation or held-out tests')
    # The collector uses consecutive even environment seeds; all study holdouts
    # are odd. Include this explicit guard if the collector convention changes.
    training = {2 * (settings.training_seed_base + settings.seed + i) for i in range(500 * 16)}
    if training & held_out:
        raise ValueError(f'{path}: training scenarios overlap held-out scenarios')
    return {'training_seed': seed, 'training_updates': updates, 'training_physical_periods': periods,
            'training_episodes': episodes, 'training_config': asdict(settings)}


def prepare_evaluation(study, scenario, root_protocol, provenance):
    output, selection_dir = study / 'benchmarks', study / 'rule_selection'
    if output.exists():
        raise ValueError('Final benchmark evidence already exists; it is never overwritten')
    selection_path, selection_protocol_path = selection_dir / 'selection.json', selection_dir / 'protocol.json'
    selection = json.loads(selection_path.read_text())
    selection_protocol = json.loads(selection_protocol_path.read_text())
    if selection['selection_protocol_sha256'] != digest(selection_protocol_path):
        raise ValueError('Rule-selection protocol no longer matches its frozen selection')
    if (scenario_from_data(selection_protocol['scenario']) != scenario
            or selection_protocol['validation_seeds'] != VALIDATION_SEEDS
            or selection_protocol['action_seed'] != ACTION_SEED
            or selection_protocol['rule_candidates'] != candidates(scenario)):
        raise ValueError('Frozen rule selection differs from this scenario, split, or candidate grid')
    if set(selection['family_winners']) != RULE_FAMILIES:
        raise ValueError('Frozen selection must contain all six rule families')
    verify_sources(selection_protocol['source_sha256'])
    expected_configs = root_protocol['configuration_by_seed']
    if set(expected_configs) != {str(seed) for seed in TRAINING_SEEDS}:
        raise ValueError('Root protocol must freeze the full configuration for each of 7/19/37')
    warm_paths = [study / 'training' / f'seed_{seed}' / 'warmstart_checkpoint.pt' for seed in TRAINING_SEEDS]
    if any(path.is_file() for path in warm_paths) != all(path.is_file() for path in warm_paths):
        raise ValueError('Warm-start secondary comparison requires all three checkpoints or none')
    variants = {'final': 'checkpoint.pt', 'validation_selected': 'best_checkpoint.pt'}
    if all(path.is_file() for path in warm_paths):
        variants['warmstart'] = 'warmstart_checkpoint.pt'
    paths, metadata = {}, {}
    for seed in TRAINING_SEEDS:
        expected = TrainingConfig(**expected_configs[str(seed)]).validate()
        for variant, filename in dict(variants, previous_final='checkpoint.pt').items():
            key = f'{variant}_{seed}'
            directory = 'previous' if variant == 'previous_final' else 'training'
            path = study / directory / f'seed_{seed}' / filename
            if not path.is_file():
                raise ValueError('Freeze every required model before testing: missing ' + str(path))
            paths[key] = path
            metadata[key] = checkpoint_metadata(path, scenario, seed, variant,
                old_training_config(seed) if variant == 'previous_final' else expected)
    protocol = dict(provenance, scenario=asdict(scenario), training_seeds=list(TRAINING_SEEDS),
        validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS,
        stress_seeds={'high_demand': HIGH_DEMAND_SEEDS, 'low_inventory': LOW_INVENTORY_SEEDS},
        action_seed=ACTION_SEED, checkpoint_metadata=metadata,
        checkpoint_paths={key: str(path.resolve()) for key, path in paths.items()},
        checkpoint_sha256={key: digest(path) for key, path in paths.items()},
        selection_path=str(selection_path.resolve()), selection_sha256=digest(selection_path),
        selection_protocol_path=str(selection_protocol_path.resolve()),
        selection_protocol_sha256=digest(selection_protocol_path), source_sha256=source_hashes(),
        primary_policy=PRIMARY, previous_policy=PREVIOUS, success_criteria=SUCCESS_CRITERIA,
        strongest_validation_rule=selection['strongest_validation_family'],
        secondary_policies='New final deterministic on all cases; new validation-best stochastic and optional warm-start stochastic on base only.',
        selection='All three fixed model seeds; no test-based checkpoint, seed, mode, hyperparameter, or rule selection.',
        stress_cases={'high_demand': 'arrival means x1.5', 'low_inventory': '35 kWh initial energy per vehicle'},
        interval_scope='Paired scene bootstrap conditional on fixed trained models; training-seed spread reported separately. Three seeds provide limited retraining evidence.',
        risk_aggregation='Mean of actual per-model tail risks and loss probabilities, not risk of a fictitious averaged trajectory.',
        comparison_scope='Within-case paired comparisons. All three cases have disjoint scene seeds; cross-case differences are descriptive.',
        reward_scope='Original environment discounted profit, without training-only potential shaping or imitation loss.')
    return output, selection, paths, protocol


def evaluate_models(study, scenario, root_protocol, provenance):
    output, selection, paths, protocol = prepare_evaluation(study, scenario, root_protocol, provenance)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)  # Freeze everything before the first test scene.
    protocol_hash, started = digest(output / 'protocol.json'), time.monotonic()
    strongest = selection['strongest_validation_family']
    cases = {'base': (scenario, TEST_SEEDS),
        'high_demand': (replace(scenario, mean_requests_by_zone=tuple(1.5*x for x in scenario.mean_requests_by_zone)), HIGH_DEMAND_SEEDS),
        'low_inventory': (replace(scenario, initial_energy_kwh=tuple(35. for _ in scenario.initial_energy_kwh)), LOW_INVENTORY_SEEDS)}
    complete, policy_episodes = {}, 0
    for case, (config, seeds) in cases.items():
        results, averages = {}, {}
        for family, spec in selection['family_winners'].items():
            result = evaluate_rule(config, seeds, spec, label=family, action_seed=ACTION_SEED)
            results[family] = result
            write_json(output / case / f'{family}.json', result)
            print(f'test {case}/{family}: {result["mean_discounted_profit"]:.4f}', flush=True)
        variants = [('previous_final', False), ('final', False), ('final', True)]
        if case == 'base':
            variants += [('validation_selected', False)]
            if 'warmstart_7' in paths:
                variants += [('warmstart', False)]
        for variant, deterministic in variants:
            mode = 'deterministic' if deterministic else 'stochastic'
            prefix = 'previous_final' if variant == 'previous_final' else f'rl_{variant}'
            models = []
            for seed in TRAINING_SEEDS:
                label = f'{prefix}_{mode}_{seed}'
                result = evaluate_checkpoint(config, seeds, paths[f'{variant}_{seed}'],
                    deterministic=deterministic, action_seed=ACTION_SEED, label=label)
                if result['invalid_actions']:
                    raise AssertionError('Learned policy emitted invalid actions: ' + label)
                results[label] = result
                models.append(result)
                write_json(output / case / f'{label}.json', result)
                print(f'test {case}/{label}: {result["mean_discounted_profit"]:.4f}', flush=True)
            key = f'{prefix}_{mode}_mean'
            averages[key] = average_trained_models(models, key)
        combined = dict(results, **averages)
        comparisons = {name: paired_comparison(result, results[strongest])
                       for name, result in combined.items() if name != strongest}
        previous_comparison = {
            'per_training_seed': {str(seed): paired_comparison(results[f'rl_final_stochastic_{seed}'],
                results[f'previous_final_stochastic_{seed}']) for seed in TRAINING_SEEDS},
            'mean': paired_comparison(averages[PRIMARY], averages[PREVIOUS])}
        table = [table_row(name, result) for name, result in combined.items()
                 if name in RULE_FAMILIES or name.endswith('_mean')]
        with (output / case / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        row = {'scenario': asdict(config), 'table': table, 'model_averages': averages,
            'comparisons_to_validation_best_rule': comparisons, 'strongest_validation_rule': strongest,
            'previous_to_final': previous_comparison,
            'invalid_actions_all_evaluations': sum(result['invalid_actions'] for result in results.values()),
            'test_policy_episodes': len(seeds) * len(results)}
        if case == 'base' and 'rl_warmstart_stochastic_mean' in averages:
            row['warmstart_to_final'] = {
                'per_training_seed': {str(seed): paired_comparison(results[f'rl_final_stochastic_{seed}'],
                    results[f'rl_warmstart_stochastic_{seed}']) for seed in TRAINING_SEEDS},
                'mean': paired_comparison(averages[PRIMARY], averages['rl_warmstart_stochastic_mean'])}
        complete[case] = row
        policy_episodes += row['test_policy_episodes']
        write_json(output / case / 'summary.json', row)
    if any(digest(paths[key]) != value for key, value in protocol['checkpoint_sha256'].items()):
        raise RuntimeError('A frozen checkpoint changed during evaluation')
    for prefix in ('selection', 'selection_protocol'):
        if digest(Path(protocol[prefix + '_path'])) != protocol[prefix + '_sha256']:
            raise RuntimeError('Frozen rule-selection evidence changed during evaluation')
    if digest(output / 'protocol.json') != protocol_hash:
        raise RuntimeError('Final benchmark protocol changed during evaluation')
    verify_sources(protocol['source_sha256'])
    verify_provenance(provenance)
    base = complete['base']
    primary = base['comparisons_to_validation_best_rule'][PRIMARY]
    individuals = [base['comparisons_to_validation_best_rule'][f'rl_final_stochastic_{seed}'] for seed in TRAINING_SEEDS]
    previous = base['previous_to_final']
    outcomes = {
        'consistent_rule_advantage': bool(all(row['mean_difference'] > 0 for row in individuals)
            and primary['paired_bootstrap_confidence_interval_95'][0] > 0),
        'consistent_previous_improvement': bool(all(row['mean_difference'] > 0 for row in previous['per_training_seed'].values())
            and previous['mean']['paired_bootstrap_confidence_interval_95'][0] > 0)}
    write_json(output / 'summary.json', {'cases': complete, 'selection': selection['family_winners'],
        'success_criteria': SUCCESS_CRITERIA, 'success_criteria_results': outcomes,
        'strongest_validation_rule': strongest, 'primary_policy': PRIMARY, 'test_tuning_performed': False,
        'test_policy_episodes': policy_episodes,
        'invalid_actions_all_evaluations': sum(row['invalid_actions_all_evaluations'] for row in complete.values()),
        'elapsed_seconds': time.monotonic() - started})
    print('Complete: ' + str(output / 'summary.json'), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('select', 'evaluate'), required=True)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/b3_improved_study')
    parser.add_argument('--scenario', type=Path, default=ROOT / 'configs/utility_b3.json')
    parser.add_argument('--study-protocol', type=Path, help='Defaults to <study-dir>/protocol.json')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    torch.set_num_threads(1)
    try:
        scenario, protocol, provenance = load_study(study, args.scenario, args.study_protocol or study / 'protocol.json')
        if args.phase == 'select':
            select_rules(study, scenario, provenance)
        else:
            evaluate_models(study, scenario, protocol, provenance)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
