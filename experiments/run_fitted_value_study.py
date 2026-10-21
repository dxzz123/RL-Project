"""Freeze a five-method fitted-value/PPO comparison on independent scenarios.

Run phases init, select, and evaluate in that order. Rule selection can run
while the three fitted-value models train. Only evaluate opens test scenarios.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

from charging_env.config import Scenario
from charging_env.benchmarks.compact import FAMILIES, candidates, evaluate_rule
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison

TRAINING_SEEDS = (7, 19, 37)
VALIDATION_SEEDS = [151000001 + 2*i for i in range(128)]
TEST_SEEDS = [153000001 + 2*i for i in range(512)]
HIGH_DEMAND_SEEDS = [155000001 + 2*i for i in range(256)]
LOW_INVENTORY_SEEDS = [157000001 + 2*i for i in range(256)]
ACTION_SEED = 159991
FITTED = 'fitted_value_mean'
PPO = 'ppo_mean'
DEFAULT_PPO_ROOT = Path('/Users/jiemingkong/Downloads/charging_env 2/outputs/b3_improved_study/training')
SUCCESS_CRITERIA = {
    'consistent_heuristic_advantage': 'Every fitted-value training seed has a positive paired 95% interval lower endpoint against the strongest rule selected only on validation, and the three-model mean paired interval also has a positive lower endpoint.',
    'consistent_ppo_advantage': 'Every fitted-value seed has a positive paired 95% interval lower endpoint against the same-seed frozen PPO model, and the three-model mean paired interval also has a positive lower endpoint.',
    'scope': 'Base-case primary criteria. Intervals describe scenario/action sampling conditional on these fixed models; they exclude retraining uncertainty. Stress cases are reported separately without choosing a controller on their results.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def scenario_from_data(data):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    return Scenario(**{key: tuples(value) for key, value in data.items()}).validate()


def validate_splits():
    splits = [VALIDATION_SEEDS, TEST_SEEDS, HIGH_DEMAND_SEEDS, LOW_INVENTORY_SEEDS]
    for i, block in enumerate(splits):
        if len(block) != len(set(block)) or any(seed < 0 or seed % 2 != 1 for seed in block):
            raise ValueError('Study holdouts must contain unique nonnegative odd seeds')
        if any(set(block) & set(other) for other in splits[i+1:]):
            raise ValueError('Rule validation and all three test splits must be disjoint')
    return {'validation': VALIDATION_SEEDS, 'base': TEST_SEEDS,
            'high_demand': HIGH_DEMAND_SEEDS, 'low_inventory': LOW_INVENTORY_SEEDS}


def source_hashes(*, selection=False):
    paths = sorted((ROOT / 'src/charging_env').rglob('*.py'))
    if selection:
        # Fitted-value/core development may continue while rules are selected.
        paths = [path for path in paths if 'rl' not in path.parts and path.name != 'learned.py']
    paths += [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_sources(hashes):
    changed = [name for name, value in hashes.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != value]
    if changed:
        raise ValueError('Frozen study sources changed: ' + ', '.join(changed))


def ppo_expected_config(seed):
    from charging_env.rl.trainer import TrainingConfig
    return TrainingConfig(seed=seed, updates=500, episodes_per_update=16, epochs=6,
        minibatch_size=192, hidden_size=64, learning_rate=3e-4,
        critic_learning_rate=1e-3, value_clip=0., entropy_coef=.005,
        gae_lambda=1., eval_episodes=64, eval_interval=25,
        eval_seed_base=4300001, eval_action_seed=439991, eval_policy='stochastic',
        discount_policy_loss=True, dispatch_final_activation='silu',
        dispatch_agent_id=True, encoder_layer_norm=True, inventory_potential=.30,
        dispatch_entropy_coef=.02, dispatch_entropy_final_coef=.003,
        dispatch_entropy_anneal_updates=500, separate_network_grad_clip=True,
        training_seed_base=10000000 + 20000*TRAINING_SEEDS.index(seed)).validate()


def fitted_expected_configs(scenario):
    from charging_env.rl.fitted_value import FittedValueConfig
    return {str(seed): json.loads(json.dumps(asdict(FittedValueConfig(seed=seed).validate(scenario))))
            for seed in TRAINING_SEEDS}


def ppo_metadata(path, scenario, seed):
    import torch
    from charging_env.rl.trainer import TrainingConfig
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if scenario_from_data(payload['scenario']) != scenario:
        raise ValueError(f'{path}: PPO checkpoint must match the exact base b=3 scenario')
    config = TrainingConfig(**payload['training_config']).validate()
    if config != ppo_expected_config(seed):
        raise ValueError(f'{path}: PPO training configuration differs from the frozen 500-update study')
    counters = {key: int(payload[key]) for key in ('update_count', 'environment_periods', 'episode_count')}
    if counters != {'update_count': 500, 'environment_periods': 96000, 'episode_count': 8000}:
        raise ValueError(f'{path}: PPO checkpoint is not the complete frozen 500-update model')
    def check_finite(value, name='checkpoint'):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f'{path}: nonfinite PPO tensor {name}')
        if isinstance(value, dict):
            for key, child in value.items():
                check_finite(child, name + '/' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                check_finite(child, name + '/' + str(index))
    check_finite(payload)
    holdouts = set(sum(validate_splits().values(), []))
    training = {2*(config.training_seed_base + seed + i) for i in range(8000)}
    validation = {config.eval_seed_base + 2*i for i in range(config.eval_episodes)}
    if holdouts & (training | validation):
        raise ValueError('PPO training or validation overlaps fresh study holdouts')
    return dict(training_seed=seed, training_config=asdict(config), **counters)


def initialize(study, scenario_path, ppo_root):
    if study.exists():
        raise FileExistsError('This study directory exists; choose a fresh study directory')
    scenario = Scenario.from_json(scenario_path)
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('The predeclared study requires the unchanged default physical b=3 scenario')
    validate_splits()
    ppo_paths = {str(seed): ppo_root / f'seed_{seed}' / 'checkpoint.pt' for seed in TRAINING_SEEDS}
    metadata = {seed: ppo_metadata(path, scenario, int(seed)) for seed, path in ppo_paths.items()}
    study.mkdir(parents=True)
    hashes = {}
    for seed, source in ppo_paths.items():
        target = study / 'ppo' / f'seed_{seed}' / 'checkpoint.pt'
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        hashes[seed] = digest(target)
        if hashes[seed] != digest(source):
            raise RuntimeError('PPO checkpoint copy differs from its frozen source')
    write_json(study / 'configuration/utility_b3.json', asdict(scenario))
    protocol = {
        'study': 'Learned continuation value with coordinated dispatch, 2026-10-20',
        'scenario': asdict(scenario), 'training_seeds': list(TRAINING_SEEDS),
        'fitted_configuration': {'grid_step': 10., 'samples': 8, 'fixed_price': .6,
                                 'energy_fractions': [.5, 1.]},
        'fitted_configuration_by_seed': fitted_expected_configs(scenario),
        'fitted_training_path': 'fitted/seed_{seed}/checkpoint.npz',
        'ppo_checkpoint_sha256': hashes, 'ppo_checkpoint_metadata': metadata,
        'ppo_checkpoint_source': str(ppo_root.resolve()),
        'splits': validate_splits(), 'action_seed': ACTION_SEED,
        'rule_candidates': candidates(scenario), 'validation_policy_episodes': 82*128,
        'test_policy_episodes': 9*(512+256+256),
        'primary_methods': [FITTED, PPO, *FAMILIES],
        'selection': 'Select one maximum-profit setting within each rule family using only the 128 validation scenes. Freeze all choices and six checkpoints before any test scene. No test tuning or model-seed selection.',
        'benchmark_grid_scope': 'The attachment gives family counts but not every search point. This explicit local grid is frozen before tests: all seven service prices x thresholds(0.2,0.4,0.6); constant prices(0.4,0.5,0.6,0.8,1.0) x shadows(0.05 through0.45 by0.05); fixed0.60 time schedules early(0.1,0.2,0.3,0.4) x late(0,0.05,0.10,0.15), switching halfway through the horizon; terminal shadow zero.',
        'success_criteria': SUCCESS_CRITERIA,
        'information_and_budget_scope': 'Fitted value and coordinated rules use known exact one-period physics and the same finite half/full exchange action grid. Fitted value additionally uses arbitrary synthetic state resets and sampled transition queries. Frozen PPO uses trajectory-based learning. These are not equal-information or equal-compute training budgets.',
        'attachment_scope': 'The attachment is an algorithm description and external performance claim, not executable code or reproduced evidence. All local results will be generated independently.',
    }
    write_json(study / 'protocol.json', protocol)
    print('Initialized: ' + str(study), flush=True)


def load_study(study):
    path = study / 'protocol.json'
    protocol = json.loads(path.read_text())
    scenario = scenario_from_data(protocol['scenario'])
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('Study scenario differs from the frozen default b=3 physical scenario')
    if (protocol['training_seeds'] != list(TRAINING_SEEDS)
            or protocol['splits'] != validate_splits()
            or protocol['action_seed'] != ACTION_SEED
            or protocol['rule_candidates'] != candidates(scenario)
            or protocol['fitted_configuration_by_seed'] != fitted_expected_configs(scenario)
            or protocol['fitted_configuration'] != {'grid_step': 10., 'samples': 8,
                'fixed_price': .6, 'energy_fractions': [.5, 1.]}):
        raise ValueError('Root protocol differs from the predeclared training, rule, or holdout settings')
    return scenario, protocol, digest(path)


def select_rules(study, scenario, root_protocol_hash):
    output = study / 'rule_selection'
    if output.exists():
        raise FileExistsError('Rule-selection evidence already exists; use a fresh study directory')
    specs = candidates(scenario)
    protocol = {'scenario': asdict(scenario), 'validation_seeds': VALIDATION_SEEDS,
        'action_seed': ACTION_SEED, 'rule_candidates': specs,
        'root_protocol_sha256': root_protocol_hash, 'source_sha256': source_hashes(selection=True),
        'test_scenarios_accessed': False, 'validation_policy_episodes': len(specs)*len(VALIDATION_SEEDS)}
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
    strongest = max(winners, key=lambda name: winners[name]['mean_discounted_profit'])
    verify_sources(protocol['source_sha256'])
    if (digest(study / 'protocol.json') != root_protocol_hash
            or digest(output / 'protocol.json') != protocol_hash):
        raise RuntimeError('Frozen protocol changed during rule selection')
    write_json(output / 'selection.json', {'family_winners': winners,
        'strongest_validation_family': strongest, 'all_candidates': rows,
        'selection_protocol_sha256': protocol_hash,
        'elapsed_seconds': time.monotonic()-started, 'test_scenarios_accessed': False})
    print('Frozen strongest validation rule: ' + strongest, flush=True)


def average_models(results, label):
    """Average actual policy metrics; retain scene means only for paired CIs."""
    seeds = results[0]['evaluation_seeds']
    if any(result['evaluation_seeds'] != seeds for result in results):
        raise ValueError('Model averages require identical ordered scenario seeds')
    result = {'label': label, 'evaluation_seeds': seeds, 'n_episodes': len(seeds),
        'aggregation': 'Mean over three separately executed frozen training-seed models; no ensemble is executed.',
        'training_seeds': [item['training_seed'] for item in results],
        'mean_by_training_seed': {str(item['training_seed']): item['mean_discounted_profit'] for item in results},
        'training_seed_sd': float(np.std([item['mean_discounted_profit'] for item in results], ddof=1)),
        'episodes': [{'seed': seed, 'discounted_profit': float(np.mean([
            item['episodes'][i]['discounted_profit'] for item in results]))} for i, seed in enumerate(seeds)],
        'invalid_actions': sum(item['invalid_actions'] for item in results)}
    scalar_fields = ('mean_discounted_profit', 'mean_profit', 'profit_p10', 'lower_tail_mean_10pct',
        'loss_probability', 'mean_served_request_share', 'mean_outside_option_share',
        'pooled_selected_service_fulfillment', 'mean_mobile_delivered_kwh', 'mean_fixed_delivered_kwh',
        'mean_final_inventory_kwh', 'mean_export_grid_kwh', 'mean_charge_grid_kwh', 'mean_mobile_price')
    result.update({key: float(np.mean([item[key] for item in results])) for key in scalar_fields})
    for key in ('mean_reward_terms', 'mean_discounted_reward_terms', 'mean_policy_decision_ms'):
        result[key] = {term: float(np.mean([item[key][term] for item in results])) for term in results[0][key]}
    return result


def table_row(name, result):
    return {'policy': name, 'mean_discounted_profit': result['mean_discounted_profit'],
        'training_seed_sd': result.get('training_seed_sd'),
        'profit_p10': result['profit_p10'], 'lower_tail_mean_10pct': result['lower_tail_mean_10pct'],
        'loss_probability': result['loss_probability'],
        'mean_served_request_share': result['mean_served_request_share'],
        'mean_mobile_delivered_kwh': result['mean_mobile_delivered_kwh'],
        'mean_charge_grid_kwh': result['mean_charge_grid_kwh'],
        'mean_final_inventory_kwh': result['mean_final_inventory_kwh'],
        'pricing_ms': result['mean_policy_decision_ms']['pricing'],
        'dispatch_ms': result['mean_policy_decision_ms']['dispatch'],
        'invalid_actions': result['invalid_actions']}


def fitted_metadata(path, scenario, seed):
    from charging_env.rl.fitted_value import FittedValueConfig, FittedValuePolicy
    policy = FittedValuePolicy.load_checkpoint(path)
    if policy.scenario != scenario:
        raise ValueError(f'{path}: fitted checkpoint must match the exact base b=3 scenario')
    settings = asdict(policy.config)
    if policy.config != FittedValueConfig(seed=seed).validate(scenario):
        raise ValueError(f'{path}: fitted configuration differs from the complete frozen default grid10/sample8 method')
    # Config field checks are explicit so adding an unrelated diagnostic setting
    # cannot silently change the training method used in this comparison.
    if settings.get('seed') != seed or settings.get('grid_step') != 10.:
        raise ValueError(f'{path}: expected fitted seed {seed} and grid step 10')
    if settings.get('samples_per_period') != 8 or scenario.price_levels[settings.get('price_index', -1)] != .6:
        raise ValueError(f'{path}: expected eight samples and fixed mobile price 0.60')
    if tuple(settings.get('energy_fractions', ())) != (.5, 1.):
        raise ValueError(f'{path}: fitted exchange grid differs from the coordinated rules')
    expected_grid = np.array([5., 15., 25., 35., 45., 55., 65., 75., 80.])
    if not np.array_equal(policy.energy_grid, expected_grid):
        raise ValueError(f'{path}: fitted battery grid differs from the frozen method')
    if policy.values.shape != (13, 4, 4, 9, 9) or not np.isfinite(policy.values).all():
        raise ValueError(f'{path}: fitted table must contain every finite time/state value')
    if not np.array_equal(policy.values[-1], np.zeros((4, 4, 9, 9))):
        raise ValueError(f'{path}: terminal continuation table must be zero')
    summary_path = path.parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary.get('completed_layers') != 12 or summary.get('states_per_period') != 1296
            or summary.get('state_draw_queries') != 124416
            or summary.get('unique_exogenous_draw_seeds') != 96
            or summary.get('value_entries') != 16848
            or summary.get('all_values_finite') is not True
            or summary.get('exact_checkpoint_reload') is not True
            or summary.get('checkpoint_sha256') != digest(path)):
        raise ValueError(f'{path}: fitted training is incomplete or its counters/reload checks are inconsistent')
    training_protocol_path = path.parent / 'protocol.json'
    training_protocol = json.loads(training_protocol_path.read_text())
    if summary.get('protocol_sha256') != digest(training_protocol_path):
        raise ValueError(f'{path}: fitted training protocol changed')
    if (scenario_from_data(training_protocol['scenario']) != scenario
            or training_protocol['training_config'] != json.loads(json.dumps(settings))
            or training_protocol.get('expected_state_draw_queries') != 124416
            or training_protocol.get('evaluation_data_used') is not False
            or summary.get('evaluation_data_used') is not False):
        raise ValueError(f'{path}: fitted training protocol differs from the model or uses evaluation data')
    draw_seeds = [int(value) for block in training_protocol['period_draw_seeds'] for value in block]
    holdouts = set(sum(validate_splits().values(), []))
    if (len(draw_seeds) != 96 or len(set(draw_seeds)) != 96
            or any(value < 0 or value % 2 != 0 for value in draw_seeds)
            or set(draw_seeds) & holdouts):
        raise ValueError(f'{path}: fitted synthetic draws overlap holdouts or do not retain96 unique even seeds')
    if (policy.training_provenance['state_draw_queries'] != 124416
            or policy.training_provenance['protocol_sha256'] != digest(training_protocol_path)
            or policy.training_provenance['source_sha256'] != training_protocol['source_sha256']
            or policy.training_provenance['candidate_outcomes'] != summary['candidate_outcomes']
            or summary['candidate_outcomes'] < 124416):
        raise ValueError(f'{path}: fitted checkpoint provenance has incomplete query counters')
    verify_sources(training_protocol['source_sha256'])
    with (path.parent / 'training_metrics.csv').open(newline='') as stream:
        layers = list(csv.DictReader(stream))
    if (len(layers) != 12 or [int(row['period']) for row in layers] != list(range(11, -1, -1))
            or any(int(row['grid_states']) != 1296 or int(row['state_draw_queries']) != 10368
                   for row in layers)
            or sum(int(row['candidate_outcomes']) for row in layers) != summary['candidate_outcomes']
            or not all(np.isfinite(float(row[key])) for row in layers
                       for key in ('mean_value', 'min_value', 'max_value', 'wall_seconds'))):
        raise ValueError(f'{path}: fitted layer records are missing or inconsistent')
    return {'training_seed': seed, 'training_config': settings,
            'table_shape': list(policy.values.shape), 'stored_values': int(policy.values.size),
            'training_summary': summary, 'training_summary_sha256': digest(summary_path),
            'training_protocol_sha256': digest(training_protocol_path),
            'synthetic_draw_seeds': draw_seeds}


def prepare_evaluation(study, scenario, root_protocol, root_hash):
    output, selection_dir = study / 'benchmarks', study / 'rule_selection'
    if output.exists():
        raise FileExistsError('Final evaluation evidence already exists; use a fresh study directory')
    selection_path, selection_protocol_path = selection_dir / 'selection.json', selection_dir / 'protocol.json'
    selection = json.loads(selection_path.read_text())
    selection_protocol = json.loads(selection_protocol_path.read_text())
    if (selection['selection_protocol_sha256'] != digest(selection_protocol_path)
            or selection_protocol['root_protocol_sha256'] != root_hash
            or selection_protocol['validation_seeds'] != VALIDATION_SEEDS
            or selection_protocol['action_seed'] != ACTION_SEED
            or selection_protocol['rule_candidates'] != candidates(scenario)
            or scenario_from_data(selection_protocol['scenario']) != scenario):
        raise ValueError('Frozen rule selection differs from this study or has changed')
    if set(selection['family_winners']) != set(FAMILIES):
        raise ValueError('Expected exactly the three selected compact rule families')
    # Recompute winners from immutable validation records, rather than trusting
    # a hand-edited selection.json or accepting test-driven family choices.
    winners = {}
    for spec in candidates(scenario):
        result = json.loads((selection_dir / 'validation' / f'{spec["id"]}.json').read_text())
        if result['evaluation_seeds'] != VALIDATION_SEEDS or result['policy_specification'] != spec:
            raise ValueError('Rule validation record differs from its frozen candidate or split')
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
    strongest = max(winners, key=lambda name: winners[name]['mean_discounted_profit'])
    if selection['family_winners'] != winners or selection['strongest_validation_family'] != strongest:
        raise ValueError('Selected benchmark settings do not maximize validation profit')
    verify_sources(selection_protocol['source_sha256'])
    paths, metadata = {}, {}
    for seed in TRAINING_SEEDS:
        for method, filename in (('fitted', 'checkpoint.npz'), ('ppo', 'checkpoint.pt')):
            key = f'{method}_{seed}'
            path = study / method / f'seed_{seed}' / filename
            if not path.is_file():
                raise ValueError('All six complete frozen checkpoints are required: ' + str(path))
            paths[key] = path
            metadata[key] = (fitted_metadata(path, scenario, seed) if method == 'fitted'
                             else ppo_metadata(path, scenario, seed))
            if (method == 'fitted' and json.loads(json.dumps(metadata[key]['training_config']))
                    != root_protocol['fitted_configuration_by_seed'][str(seed)]):
                raise ValueError('Fitted checkpoint configuration differs from the root study protocol')
            if method == 'ppo' and digest(path) != root_protocol['ppo_checkpoint_sha256'][str(seed)]:
                raise ValueError('Frozen original PPO checkpoint has changed')
    protocol = {'scenario': asdict(scenario), 'training_seeds': list(TRAINING_SEEDS),
        'splits': validate_splits(), 'action_seed': ACTION_SEED,
        'root_protocol_sha256': root_hash, 'source_sha256': source_hashes(),
        'checkpoint_paths': {key: str(path.relative_to(study)) for key, path in paths.items()},
        'checkpoint_sha256': {key: digest(path) for key, path in paths.items()},
        'checkpoint_metadata': metadata, 'selected_rules': winners,
        'strongest_validation_rule': strongest,
        'selection_sha256': digest(selection_path), 'selection_protocol_sha256': digest(selection_protocol_path),
        'success_criteria': SUCCESS_CRITERIA,
        'main_methods': [FITTED, PPO, *FAMILIES], 'test_policy_episodes': 9216,
        'evaluation': 'All three complete fitted models and all three frozen final stochastic PPO models. No best-checkpoint, deterministic, random, wait, or extra benchmark rows.',
        'stress_cases': {'high_demand': 'Arrival means multiplied by1.5', 'low_inventory': 'Initial energy35kWh per vehicle'},
        'stress_policy': 'Keep base-trained continuation values and PPO feature scales frozen; change only runtime demand or initial battery energy.',
        'comparison_scope': 'Paired within each case; all three case seed sets are disjoint. Primary superiority criteria use base only.',
        'information_and_budget_scope': root_protocol['information_and_budget_scope'],
        'risk_scope': 'Model-average risk metrics average actual per-model risks, rather than risk of an averaged artificial trajectory.',
        'test_tuning_performed': False}
    synthetic_blocks = [set(metadata[f'fitted_{seed}']['synthetic_draw_seeds']) for seed in TRAINING_SEEDS]
    if any(left & right for i, left in enumerate(synthetic_blocks) for right in synthetic_blocks[i+1:]):
        raise ValueError('Independent fitted training seeds must have disjoint synthetic draw seeds')
    return output, selection, paths, protocol


def _positive_interval(comparison):
    interval = comparison['paired_bootstrap_confidence_interval_95']
    return interval is not None and interval[0] > 0


def evaluate_models(study, scenario, root_protocol, root_hash):
    from charging_env.rl.fitted_value import FittedValuePolicy
    from charging_env.benchmarks.learned import evaluate_checkpoint
    import torch
    torch.set_num_threads(1)
    output, selection, paths, protocol = prepare_evaluation(study, scenario, root_protocol, root_hash)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)  # Must precede every test episode.
    protocol_hash, started = digest(output / 'protocol.json'), time.monotonic()
    strongest = selection['strongest_validation_family']
    cases = {'base': (scenario, TEST_SEEDS),
        'high_demand': (replace(scenario, mean_requests_by_zone=tuple(1.5*x for x in scenario.mean_requests_by_zone)), HIGH_DEMAND_SEEDS),
        'low_inventory': (replace(scenario, initial_energy_kwh=tuple(35. for _ in scenario.initial_energy_kwh)), LOW_INVENTORY_SEEDS)}
    complete, executions = {}, 0
    for case, (config, seeds) in cases.items():
        results, averages = {}, {}
        for family, spec in selection['family_winners'].items():
            result = evaluate_rule(config, seeds, spec, label=family, action_seed=ACTION_SEED)
            results[family] = result
            write_json(output / case / f'{family}.json', result)
            print(f'test {case}/{family}: {result["mean_discounted_profit"]:.4f}', flush=True)
        for method, mean_label in (('fitted', FITTED), ('ppo', PPO)):
            models = []
            for seed in TRAINING_SEEDS:
                label = f'{method}_{seed}'
                if method == 'fitted':
                    policy = FittedValuePolicy.load_checkpoint(paths[label]).with_scenario(config, allow_distribution_shift=True)
                    result = evaluate_policy(config, seeds, lambda obs, rng: policy.prices(obs),
                        lambda obs, rng: policy.dispatch(obs), action_seed=ACTION_SEED, label=label)
                    result.update(training_seed=seed, checkpoint=str(paths[label].relative_to(study)),
                                  continuation_values='Frozen base-trained table; no refit on evaluation scenes')
                else:
                    result = evaluate_checkpoint(config, seeds, paths[label], deterministic=False,
                                                 action_seed=ACTION_SEED, label=label)
                    result['checkpoint'] = str(paths[label].relative_to(study))
                if result['invalid_actions']:
                    raise RuntimeError('A frozen learned policy emitted invalid actions: ' + label)
                results[label] = result
                models.append(result)
                write_json(output / case / f'{label}.json', result)
                print(f'test {case}/{label}: {result["mean_discounted_profit"]:.4f}', flush=True)
            averages[mean_label] = average_models(models, mean_label)
        combined = dict(results, **averages)
        comparisons = {name: paired_comparison(result, results[strongest])
                       for name, result in combined.items() if name != strongest}
        versus_ppo = {'per_training_seed': {str(seed): paired_comparison(results[f'fitted_{seed}'],
                       results[f'ppo_{seed}']) for seed in TRAINING_SEEDS},
                      'mean': paired_comparison(averages[FITTED], averages[PPO])}
        table = [table_row(name, combined[name]) for name in [FITTED, PPO, *FAMILIES]]
        with (output / case / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        row = {'scenario': asdict(config), 'table': table, 'model_averages': averages,
            'comparisons_to_validation_best_rule': comparisons,
            'strongest_validation_rule': strongest, 'fitted_vs_ppo': versus_ppo,
            'test_policy_episodes': len(seeds)*len(results),
            'invalid_actions_all_evaluations': sum(result['invalid_actions'] for result in results.values())}
        complete[case] = row
        executions += row['test_policy_episodes']
        write_json(output / case / 'summary.json', row)
    for key, value in protocol['checkpoint_sha256'].items():
        if digest(paths[key]) != value:
            raise RuntimeError('Frozen checkpoint changed during evaluation: ' + key)
    for name in ('selection', 'selection_protocol'):
        file = study / 'rule_selection' / ('selection.json' if name == 'selection' else 'protocol.json')
        if digest(file) != protocol[name + '_sha256']:
            raise RuntimeError('Frozen rule selection changed during final evaluation')
    for seed in TRAINING_SEEDS:
        summary_path = paths[f'fitted_{seed}'].parent / 'summary.json'
        if digest(summary_path) != protocol['checkpoint_metadata'][f'fitted_{seed}']['training_summary_sha256']:
            raise RuntimeError('Fitted training summary changed during evaluation')
        if digest(summary_path.parent / 'protocol.json') != protocol['checkpoint_metadata'][f'fitted_{seed}']['training_protocol_sha256']:
            raise RuntimeError('Fitted training protocol changed during evaluation')
    verify_sources(protocol['source_sha256'])
    if digest(study / 'protocol.json') != root_hash or digest(output / 'protocol.json') != protocol_hash:
        raise RuntimeError('A frozen study protocol changed during evaluation')
    base = complete['base']
    versus_rule = base['comparisons_to_validation_best_rule']
    criteria = {'consistent_heuristic_advantage': all(_positive_interval(versus_rule[f'fitted_{seed}'])
                   for seed in TRAINING_SEEDS) and _positive_interval(versus_rule[FITTED]),
                'consistent_ppo_advantage': all(_positive_interval(item) for item in
                   base['fitted_vs_ppo']['per_training_seed'].values()) and _positive_interval(base['fitted_vs_ppo']['mean'])}
    if executions != 9216:
        raise RuntimeError('Unexpected number of policy-episode executions')
    write_json(output / 'summary.json', {'cases': complete, 'selection': selection['family_winners'],
        'strongest_validation_rule': strongest, 'primary_policy': FITTED,
        'success_criteria': SUCCESS_CRITERIA, 'success_criteria_results': criteria,
        'test_policy_episodes': executions, 'test_tuning_performed': False,
        'invalid_actions_all_evaluations': sum(row['invalid_actions_all_evaluations'] for row in complete.values()),
        'elapsed_seconds': time.monotonic()-started})
    print('Complete: ' + str(output / 'summary.json'), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'select', 'evaluate'))
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/fitted_value_study')
    parser.add_argument('--scenario', type=Path, default=ROOT / 'configs/utility_b3.json')
    parser.add_argument('--ppo-root', type=Path, default=DEFAULT_PPO_ROOT,
                        help='Existing three-seed final500-update PPO study; copied and frozen at initialization')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    try:
        if args.phase == 'init':
            initialize(study, args.scenario, args.ppo_root)
        else:
            scenario, protocol, root_hash = load_study(study)
            if args.phase == 'select':
                select_rules(study, scenario, root_hash)
            else:
                evaluate_models(study, scenario, protocol, root_hash)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
