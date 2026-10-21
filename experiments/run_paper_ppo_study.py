"""Train and compare the Joint PPO controller.

Phases: init freezes the protocol and copies six existing controls; select tunes
three heuristic families using validation only; train fits one predeclared new
seed; evaluate compares all fixed models on fresh independent test scenarios.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np

from charging_env.config import Scenario
from charging_env.benchmarks.compact import FAMILIES, candidates, evaluate_rule
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from run_fitted_value_study import (
    average_models, digest, fitted_metadata, ppo_metadata, scenario_from_data,
    table_row, write_json,
)

TRAINING_SEEDS = (7, 19, 37)
MONITOR_SEEDS = [179000001 + 2*i for i in range(64)]
VALIDATION_SEEDS = [181000001 + 2*i for i in range(128)]
TEST_SEEDS = [183000001 + 2*i for i in range(512)]
HIGH_DEMAND_SEEDS = [185000001 + 2*i for i in range(256)]
LOW_INVENTORY_SEEDS = [187000001 + 2*i for i in range(256)]
ACTION_SEED = 189991
PAPER = 'paper_ppo_mean'
LEGACY = 'legacy_ppo_mean'
FITTED = 'fitted_value_mean'
METHODS = (PAPER, LEGACY, FITTED, *FAMILIES)
DEFAULT_CONTROL_ROOT = Path('/Users/jiemingkong/Downloads/charging_env 2/outputs/fitted_value_study')
SUCCESS_CRITERIA = {
    'consistent_heuristic_advantage': 'Each new final Joint PPO training seed and their mean have a positive paired 95% interval lower endpoint against the strongest heuristic chosen only on validation.',
    'consistent_legacy_ppo_improvement': 'Each new final Joint PPO model has a positive paired 95% interval lower endpoint against the same-seed frozen legacy PPO model; the three-model mean interval also has a positive lower endpoint.',
    'consistent_fitted_value_advantage': 'Each new final Joint PPO model has a positive paired 95% interval lower endpoint against the same-seed frozen fitted-value model; the three-model mean interval also has a positive lower endpoint.',
    'scope': 'Predeclared base-case criteria conditional on these fixed models. Stress cases remain descriptive and do not select another controller. Three training seeds provide limited retraining evidence.',
}


def validate_splits():
    splits = {'monitor': MONITOR_SEEDS, 'validation': VALIDATION_SEEDS,
              'base': TEST_SEEDS, 'high_demand': HIGH_DEMAND_SEEDS,
              'low_inventory': LOW_INVENTORY_SEEDS}
    blocks = list(splits.values())
    for i, block in enumerate(blocks):
        if len(block) != len(set(block)) or any(seed < 0 or seed % 2 != 1 for seed in block):
            raise ValueError('All monitor, validation and test seeds must be unique nonnegative odd integers')
        if any(set(block) & set(other) for other in blocks[i+1:]):
            raise ValueError('Monitoring, rule validation and all three test cases must be disjoint')
    return splits


def config_for(seed):
    from charging_env.rl.paper_ppo import PaperPPOConfig
    if seed not in TRAINING_SEEDS:
        raise ValueError('This study freezes exactly training seeds 7, 19, and 37')
    return PaperPPOConfig(seed=seed, updates=500, episodes_per_update=16,
        epochs=6, minibatch_size=192, hidden_size=64,
        actor_learning_rate=3e-4, critic_learning_rate=1e-3, clip_ratio=.2,
        gae_lambda=.95, pricing_entropy_coef=0., dispatch_entropy_coef=.01,
        max_grad_norm=.5, target_kl=0., eval_episodes=64, eval_interval=25,
        eval_seed_base=179000001, eval_action_seed=ACTION_SEED,
        eval_policy='stochastic', normalize_advantages=False,
        dispatch_critic_target='dispatch',
        training_seed_base=210000000 + 20000*TRAINING_SEEDS.index(seed),
        torch_threads=1, device='cpu').validate()


def configuration_by_seed():
    return {str(seed): asdict(config_for(seed)) for seed in TRAINING_SEEDS}


def source_hashes(*, selection=False):
    paths = sorted((ROOT / 'src/charging_env').rglob('*.py'))
    if selection:
        paths = [path for path in paths if 'rl' not in path.parts and path.name != 'learned.py']
    paths += [Path(__file__).resolve(), ROOT / 'experiments/run_fitted_value_study.py',
              ROOT / 'docs/MDP_RL_2.pdf']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_sources(hashes):
    changed = [name for name, value in hashes.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != value]
    if changed:
        raise ValueError('Frozen Joint PPO study sources changed: ' + ', '.join(changed))


def snapshot_training_sources(study, protocol):
    """Preserve every frozen implementation file automatically before training."""
    output = study / 'provenance/executed_training_source'
    output.mkdir(parents=True, exist_ok=False)
    verify_sources(protocol['training_source_sha256'])
    entries = {}
    for name, value in protocol['training_source_sha256'].items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        if digest(target) != value:
            raise RuntimeError('Source snapshot changed during initialization: ' + name)
        entries[name] = {'snapshot_relative_path': name, 'frozen_protocol_sha256': value,
                         'snapshot_sha256': value, 'bytes': target.stat().st_size}
        target.chmod(0o444)
    shutil.copy2(study / 'protocol.json', output / 'root_protocol.json')
    manifest = {'schema': 'charging_env.paper_ppo.executed_source.v1',
        'root_protocol_sha256': digest(study / 'protocol.json'), 'sources': entries,
        'scope': 'All frozen training and evaluation implementation files and the algorithm specification PDF, copied at initialization before any new model trains; current sources are rechecked before and after each training run.'}
    write_json(output / 'manifest.json', manifest)
    for name in ('root_protocol.json', 'manifest.json'):
        (output / name).chmod(0o444)
    verify_sources(protocol['training_source_sha256'])


def held_out():
    return {seed for block in validate_splits().values() for seed in block}


def check_control_splits(metadata, method):
    if method == 'fitted':
        sampled = set(metadata['synthetic_draw_seeds'])
    else:
        settings = metadata['training_config']
        sampled = {2*(settings['training_seed_base']+settings['seed']+i) for i in range(8000)}
        sampled |= {settings['eval_seed_base']+2*i for i in range(settings['eval_episodes'])}
    if sampled & held_out():
        raise ValueError('A frozen control trained or selected on this study\'s new holdouts')


def control_metadata(path, scenario, seed, method):
    result = (fitted_metadata(path, scenario, seed) if method == 'fitted'
              else ppo_metadata(path, scenario, seed))
    check_control_splits(result, method)
    return result


def initialize(study, scenario_path, control_root):
    if study.exists():
        raise FileExistsError('Use a new study directory; recorded experiments are preserved')
    scenario = Scenario.from_json(scenario_path)
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('The Joint PPO study requires the unchanged default physical b=3 scenario')
    validate_splits()
    configurations = configuration_by_seed()
    metadata, sources = {}, {}
    for seed in TRAINING_SEEDS:
        for method, directory, filename in (('legacy', 'ppo', 'checkpoint.pt'),
                                             ('fitted', 'fitted', 'checkpoint.npz')):
            key = f'{method}_{seed}'
            source = control_root / directory / f'seed_{seed}' / filename
            if not source.is_file():
                raise FileNotFoundError('Missing complete frozen control: ' + str(source))
            metadata[key] = control_metadata(source, scenario, seed, method)
            sources[key] = source
    study.mkdir(parents=True)
    hashes, paths = {}, {}
    for key, source in sources.items():
        method, seed = key.rsplit('_', 1)
        target = study / method / f'seed_{seed}' / source.name
        if method == 'fitted':
            shutil.copytree(source.parent, target.parent)
        else:
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError('Frozen control copy changed its checkpoint bytes')
        hashes[key], paths[key] = digest(target), str(target.relative_to(study))
    write_json(study / 'configuration/utility_b3.json', asdict(scenario))
    protocol = {'study': 'Joint PPO comparison, 2026-10-21',
        'algorithm_specification_pdf': 'docs/MDP_RL_2.pdf',
        'scenario': asdict(scenario), 'training_seeds': list(TRAINING_SEEDS),
        'configuration_by_seed': configurations, 'training_source_sha256': source_hashes(),
        'control_checkpoint_sha256': hashes, 'control_checkpoint_paths': paths,
        'control_checkpoint_metadata': metadata, 'control_source': str(control_root.resolve()),
        'splits': validate_splits(), 'action_seed': ACTION_SEED,
        'rule_candidates': candidates(scenario), 'methods': list(METHODS),
        'training_updates_per_new_model': 500, 'training_episodes_per_new_model': 8000,
        'training_physical_periods_per_new_model': 96000,
        'new_training_episodes_total': 24000, 'new_training_physical_periods_total': 288000,
        'validation_policy_episodes': 10496, 'test_policy_episodes': 12288,
        'primary': 'All three new final 500-update stochastic Joint PPO models, averaged only for reporting; no ensemble or test-based checkpoint/seed selection.',
        'algorithm_scope': 'Interleaved pricing/dispatch GAE, one joint dispatch likelihood ratio over the fleet, separate stage critics and actor objectives, dispatch entropy only, raw advantages and critic MSE. Four optimizer/gradient-clipping groups. Current simulator pricing reward is zero.',
        'equation11_interpretation': 'The dispatch critic target is the dispatch advantage plus the old dispatch value (Equation 11). The optional pricing_advantage setting is a separate experimental deviation that substitutes pricing advantage; it differs from Equation 11 and is not used in this primary experiment.',
        'heuristic_selection': 'Select one maximum discounted-profit setting independently within each of three heuristic families (82 candidates in total) on rule validation only; choose the strongest family using the same validation scores before final testing.',
        'model_selection': 'Monitor 64 separate scenarios every 25 updates; retain the final 500-update checkpoint as primary regardless of monitored best score. Never choose a checkpoint, execution mode or seed on final tests.',
        'information_and_budget_scope': 'New and legacy PPO use sampled episode rollouts. Fitted-value controls use arbitrary synthetic fleet-state resets and a known one-period model. Coordinated heuristic controls also use the known one-period model. These are not equal-information or equal-compute budgets.',
        'success_criteria': SUCCESS_CRITERIA, 'test_tuning_performed': False}
    write_json(study / 'protocol.json', protocol)
    snapshot_training_sources(study, protocol)
    print('Initialized frozen Joint PPO study: ' + str(study), flush=True)


def load_study(study):
    protocol_path = study / 'protocol.json'
    data = json.loads(protocol_path.read_text())
    scenario = scenario_from_data(data['scenario'])
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('Root protocol scenario differs from the frozen base b=3 scenario')
    if (data['training_seeds'] != list(TRAINING_SEEDS)
            or data['configuration_by_seed'] != configuration_by_seed()
            or data['splits'] != validate_splits() or data['action_seed'] != ACTION_SEED
            or data['rule_candidates'] != candidates(scenario)
            or data['methods'] != list(METHODS)):
        raise ValueError('Root protocol differs from the predeclared study configurations or splits')
    return scenario, data, digest(protocol_path)


def select_rules(study, scenario, root_hash):
    output = study / 'rule_selection'
    if output.exists():
        raise FileExistsError('Rule-selection evidence exists; use a fresh study directory')
    protocol = {'scenario': asdict(scenario), 'validation_seeds': VALIDATION_SEEDS,
        'action_seed': ACTION_SEED, 'rule_candidates': candidates(scenario),
        'root_protocol_sha256': root_hash, 'source_sha256': source_hashes(selection=True),
        'test_scenarios_accessed': False, 'validation_policy_episodes': 10496}
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    selection_hash, started = digest(output / 'protocol.json'), time.monotonic()
    winners, rows = {}, []
    for spec in candidates(scenario):
        result = evaluate_rule(scenario, VALIDATION_SEEDS, spec, action_seed=ACTION_SEED)
        write_json(output / 'validation' / f'{spec["id"]}.json', result)
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        rows.append(row)
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
        print(f'validation {spec["id"]}: {result["mean_discounted_profit"]:.4f}', flush=True)
    verify_sources(protocol['source_sha256'])
    if digest(study / 'protocol.json') != root_hash or digest(output / 'protocol.json') != selection_hash:
        raise RuntimeError('A frozen protocol changed during rule selection')
    strongest = max(winners, key=lambda name: winners[name]['mean_discounted_profit'])
    write_json(output / 'selection.json', {'family_winners': winners,
        'strongest_validation_family': strongest, 'all_candidates': rows,
        'selection_protocol_sha256': selection_hash, 'test_scenarios_accessed': False,
        'elapsed_seconds': time.monotonic()-started})
    print('Frozen strongest validation rule: ' + strongest, flush=True)


def train_seed(study, scenario, protocol, root_hash, seed):
    from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer
    if seed not in TRAINING_SEEDS:
        raise ValueError('--seed must be 7, 19, or 37 for the frozen study')
    verify_sources(protocol['training_source_sha256'])
    if digest(study / 'protocol.json') != root_hash:
        raise ValueError('Root protocol changed before training')
    settings = PaperPPOConfig(**protocol['configuration_by_seed'][str(seed)]).validate()
    if settings != config_for(seed):
        raise ValueError('Training configuration differs from the frozen seed configuration')
    output = study / 'training' / f'seed_{seed}'
    if output.exists():
        raise FileExistsError('This seed already has training evidence: ' + str(output))
    result = PaperPPOTrainer(scenario, settings).train(output)
    verify_sources(protocol['training_source_sha256'])
    if digest(study / 'protocol.json') != root_hash:
        raise ValueError('Root protocol changed during training')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def _all_tensors_finite(value, label):
    import torch
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise ValueError('Nonfinite tensor in Joint PPO checkpoint: ' + label)
    if isinstance(value, dict):
        for key, child in value.items():
            _all_tensors_finite(child, label + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _all_tensors_finite(child, label + '/' + str(index))


def paper_metadata(path, scenario, seed, expected_configuration):
    import torch
    from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('algorithm') != 'paper_joint_ppo_v1' or payload.get('format_version') != 1:
        raise ValueError('Expected the new Joint PPO checkpoint format')
    if scenario_from_data(payload['scenario']) != scenario:
        raise ValueError('Joint PPO checkpoint scenario differs from the exact b=3 base scenario')
    config = PaperPPOConfig(**payload['training_config']).validate()
    if asdict(config) != expected_configuration or config != config_for(seed):
        raise ValueError('Joint PPO checkpoint training configuration differs from the frozen root protocol')
    counters = {key: int(payload[key]) for key in ('update_count', 'episode_count', 'environment_periods')}
    if counters != {'update_count': 500, 'episode_count': 8000, 'environment_periods': 96000}:
        raise ValueError('Primary Joint PPO checkpoints must complete 500 updates, 8000 episodes, and 96000 periods')
    _all_tensors_finite(payload, 'checkpoint')
    training_seeds = {2*(config.training_seed_base + seed + i) for i in range(8000)}
    if training_seeds & held_out() or config.eval_seed_base != MONITOR_SEEDS[0] or config.eval_episodes != len(MONITOR_SEEDS):
        raise ValueError('New training/monitoring seeds overlap holdouts or differ from the frozen protocol')
    summary_path = path.parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary.get('update_count') != 500 or summary.get('episode_count') != 8000
            or summary.get('environment_periods') != 96000
            or summary.get('all_metrics_finite') is not True
            or summary.get('checkpoint_reload_matches') is not True
            or summary.get('invalid_evaluation_actions') != 0
            or summary.get('checkpoint_sha256') != digest(path)):
        raise ValueError('Joint PPO training is incomplete or failed its finite/replay/action checks')
    training_protocol_path = path.parent / 'protocol.json'
    training_protocol = json.loads(training_protocol_path.read_text())
    if (summary.get('protocol_sha256') != digest(training_protocol_path)
            or training_protocol['training_config'] != expected_configuration
            or scenario_from_data(training_protocol['scenario']) != scenario
            or training_protocol.get('dispatch_critic_target') != 'dispatch'
            or training_protocol.get('raw_rewards') is not True
            or training_protocol.get('reward_shaping') is not False):
        raise ValueError('Joint PPO training protocol differs from the frozen method or scenario')
    verify_sources(training_protocol['source_sha256'])
    with (path.parent / 'training_metrics.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if len(metrics) != 500 or not all(np.isfinite(float(value)) for row in metrics for value in row.values()):
        raise ValueError('Joint PPO training metrics are missing or nonfinite')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        restored = PaperPPOTrainer.load_checkpoint(path, device='cpu')
        if restored.scenario != scenario or restored.config != config:
            raise ValueError('Joint PPO checkpoint strict reload changed the scenario or configuration')
    return {'training_seed': seed, 'training_config': asdict(config), **counters,
            'training_summary_sha256': digest(summary_path), 'training_summary': summary,
            'training_protocol_sha256': digest(training_protocol_path)}


def prepare_evaluation(study, scenario, root_protocol, root_hash):
    output = study / 'benchmarks'
    if output.exists():
        raise FileExistsError('Final benchmark evidence exists; use a fresh study directory')
    verify_sources(root_protocol['training_source_sha256'])
    selection_dir = study / 'rule_selection'
    selection_path, selection_protocol_path = selection_dir / 'selection.json', selection_dir / 'protocol.json'
    selection = json.loads(selection_path.read_text())
    selection_protocol = json.loads(selection_protocol_path.read_text())
    if (selection['selection_protocol_sha256'] != digest(selection_protocol_path)
            or selection_protocol['root_protocol_sha256'] != root_hash
            or selection_protocol['validation_seeds'] != VALIDATION_SEEDS
            or selection_protocol['rule_candidates'] != candidates(scenario)
            or selection_protocol['action_seed'] != ACTION_SEED
            or scenario_from_data(selection_protocol['scenario']) != scenario):
        raise ValueError('Rule-selection evidence differs from the frozen protocol')
    winners = {}
    for spec in candidates(scenario):
        result = json.loads((selection_dir / 'validation' / f'{spec["id"]}.json').read_text())
        if result['evaluation_seeds'] != VALIDATION_SEEDS or result['policy_specification'] != spec:
            raise ValueError('A rule was selected using changed settings or nonvalidation scenarios')
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
    strongest = max(winners, key=lambda name: winners[name]['mean_discounted_profit'])
    if selection['family_winners'] != winners or selection['strongest_validation_family'] != strongest:
        raise ValueError('Frozen heuristic settings do not maximize their validation family scores')
    verify_sources(selection_protocol['source_sha256'])
    paths, metadata = {}, {}
    for seed in TRAINING_SEEDS:
        for method, directory, filename in (('paper', 'training', 'checkpoint.pt'),
                ('legacy', 'legacy', 'checkpoint.pt'), ('fitted', 'fitted', 'checkpoint.npz')):
            key = f'{method}_{seed}'
            path = study / directory / f'seed_{seed}' / filename
            if not path.is_file():
                raise ValueError('Freeze all nine complete model checkpoints before testing: ' + str(path))
            paths[key] = path
            metadata[key] = (paper_metadata(path, scenario, seed, root_protocol['configuration_by_seed'][str(seed)])
                if method == 'paper' else control_metadata(path, scenario, seed, method))
            if method != 'paper' and digest(path) != root_protocol['control_checkpoint_sha256'][key]:
                raise ValueError('Frozen control checkpoint differs from its initialization hash')
    protocol = {'scenario': asdict(scenario), 'training_seeds': list(TRAINING_SEEDS),
        'root_protocol_sha256': root_hash, 'source_sha256': source_hashes(),
        'splits': validate_splits(), 'action_seed': ACTION_SEED,
        'checkpoint_paths': {key: str(path.relative_to(study)) for key, path in paths.items()},
        'checkpoint_sha256': {key: digest(path) for key, path in paths.items()},
        'checkpoint_metadata': metadata, 'selected_rules': winners,
        'strongest_validation_rule': strongest, 'selection_sha256': digest(selection_path),
        'selection_protocol_sha256': digest(selection_protocol_path),
        'main_methods': list(METHODS), 'test_policy_episodes': 12288,
        'primary': root_protocol['primary'], 'success_criteria': SUCCESS_CRITERIA,
        'information_and_budget_scope': root_protocol['information_and_budget_scope'],
        'interval_scope': 'Paired scene/action bootstrap conditional on fixed models. Three training-seed means are shown separately; their average is not an ensemble.',
        'risk_scope': 'Average actual individual-model risk statistics; scene-average profits are used only for conditional paired intervals.',
        'stress_scope': 'Freeze every base-trained model and rule. Runtime arrival means increase 50% or initial battery energy falls to 35 kWh; do not refit, retune, or switch the primary baseline.',
        'test_tuning_performed': False}
    return output, selection, paths, protocol


def evaluate_paper_checkpoint(config, seeds, checkpoint, *, action_seed=ACTION_SEED, label=''):
    import torch
    from charging_env.rl.paper_ppo import PaperPPOTrainer
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        trainer = PaperPPOTrainer.load_checkpoint(checkpoint, device='cpu')
        trainer = trainer.with_scenario(config, allow_distribution_shift=True)
        trainer.set_evaluation_mode(deterministic=False)
        result = evaluate_policy(config, seeds, trainer.prices, trainer.dispatch,
            action_seed=action_seed, label=label, on_episode_start=trainer.on_episode_start)
        result.update(training_seed=trainer.config.seed, training_updates=trainer.update_count,
            training_physical_periods=trainer.environment_periods, deterministic=False,
            feature_scaling='Frozen base-training scales; no adaptation to test scenes')
        return result


def _positive_interval(comparison):
    interval = comparison['paired_bootstrap_confidence_interval_95']
    return interval is not None and interval[0] > 0


def evaluate_models(study, scenario, root_protocol, root_hash):
    import torch
    from charging_env.rl.fitted_value import FittedValuePolicy
    from charging_env.benchmarks.learned import evaluate_checkpoint
    torch.set_num_threads(1)
    output, selection, paths, protocol = prepare_evaluation(study, scenario, root_protocol, root_hash)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)  # Freeze every model and rule before opening any test scene.
    protocol_hash, started = digest(output / 'protocol.json'), time.monotonic()
    cases = {'base': (scenario, TEST_SEEDS),
        'high_demand': (replace(scenario, mean_requests_by_zone=tuple(1.5*x for x in scenario.mean_requests_by_zone)), HIGH_DEMAND_SEEDS),
        'low_inventory': (replace(scenario, initial_energy_kwh=tuple(35. for _ in scenario.initial_energy_kwh)), LOW_INVENTORY_SEEDS)}
    strongest, complete, executions = selection['strongest_validation_family'], {}, 0
    for case, (config, seeds) in cases.items():
        results, averages = {}, {}
        for family, spec in selection['family_winners'].items():
            result = evaluate_rule(config, seeds, spec, label=family, action_seed=ACTION_SEED)
            results[family] = result
            write_json(output / case / f'{family}.json', result)
            print(f'test {case}/{family}: {result["mean_discounted_profit"]:.4f}', flush=True)
        for method, average_label in (('paper', PAPER), ('legacy', LEGACY), ('fitted', FITTED)):
            models = []
            for seed in TRAINING_SEEDS:
                label, path = f'{method}_{seed}', paths[f'{method}_{seed}']
                if method == 'paper':
                    result = evaluate_paper_checkpoint(config, seeds, path, action_seed=ACTION_SEED, label=label)
                elif method == 'legacy':
                    result = evaluate_checkpoint(config, seeds, path, deterministic=False, action_seed=ACTION_SEED, label=label)
                else:
                    policy = FittedValuePolicy.load_checkpoint(path).with_scenario(config, allow_distribution_shift=True)
                    result = evaluate_policy(config, seeds, policy.prices, policy.dispatch, action_seed=ACTION_SEED, label=label)
                    result.update(training_seed=seed, continuation_values='Frozen base-trained table')
                if result['invalid_actions']:
                    raise RuntimeError('A frozen learned controller emitted invalid actions: ' + label)
                result['checkpoint'] = str(path.relative_to(study))
                results[label] = result
                models.append(result)
                write_json(output / case / f'{label}.json', result)
                print(f'test {case}/{label}: {result["mean_discounted_profit"]:.4f}', flush=True)
            averages[average_label] = average_models(models, average_label)
        combined = dict(results, **averages)
        comparisons = {name: paired_comparison(value, results[strongest])
                       for name, value in combined.items() if name != strongest}
        versus = {}
        for method, mean_label in (('legacy', LEGACY), ('fitted', FITTED)):
            versus[method] = {'per_training_seed': {str(seed): paired_comparison(results[f'paper_{seed}'],
                results[f'{method}_{seed}']) for seed in TRAINING_SEEDS},
                'mean': paired_comparison(averages[PAPER], averages[mean_label])}
        table = [table_row(name, combined[name]) for name in METHODS]
        with (output / case / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        row = {'scenario': asdict(config), 'table': table, 'model_averages': averages,
            'comparisons_to_validation_best_rule': comparisons,
            'strongest_validation_rule': strongest, 'paper_vs_legacy': versus['legacy'],
            'paper_vs_fitted': versus['fitted'], 'test_policy_episodes': len(seeds)*len(results),
            'invalid_actions_all_evaluations': sum(value['invalid_actions'] for value in results.values())}
        complete[case] = row
        executions += row['test_policy_episodes']
        write_json(output / case / 'summary.json', row)
    if executions != 12288:
        raise RuntimeError('Unexpected number of final policy-episode executions')
    for key, value in protocol['checkpoint_sha256'].items():
        if digest(paths[key]) != value:
            raise RuntimeError('A frozen checkpoint changed during testing: ' + key)
    for name, filename in (('selection', 'selection.json'), ('selection_protocol', 'protocol.json')):
        if digest(study / 'rule_selection' / filename) != protocol[name + '_sha256']:
            raise RuntimeError('Frozen validation selection changed during testing')
    for seed in TRAINING_SEEDS:
        if digest(paths[f'paper_{seed}'].parent / 'summary.json') != protocol['checkpoint_metadata'][f'paper_{seed}']['training_summary_sha256']:
            raise RuntimeError('Joint PPO training summary changed during testing')
        if digest(paths[f'paper_{seed}'].parent / 'protocol.json') != protocol['checkpoint_metadata'][f'paper_{seed}']['training_protocol_sha256']:
            raise RuntimeError('Joint PPO training protocol changed during testing')
    verify_sources(protocol['source_sha256'])
    if digest(study / 'protocol.json') != root_hash or digest(output / 'protocol.json') != protocol_hash:
        raise RuntimeError('A frozen study protocol changed during testing')
    base = complete['base']
    criteria = {'consistent_heuristic_advantage': all(_positive_interval(base['comparisons_to_validation_best_rule'][f'paper_{seed}']) for seed in TRAINING_SEEDS)
                and _positive_interval(base['comparisons_to_validation_best_rule'][PAPER])}
    for key, field in (('consistent_legacy_ppo_improvement', 'paper_vs_legacy'),
                       ('consistent_fitted_value_advantage', 'paper_vs_fitted')):
        criteria[key] = all(_positive_interval(value) for value in base[field]['per_training_seed'].values()) and _positive_interval(base[field]['mean'])
    write_json(output / 'summary.json', {'cases': complete, 'selection': selection['family_winners'],
        'strongest_validation_rule': strongest, 'primary_policy': PAPER,
        'success_criteria': SUCCESS_CRITERIA, 'success_criteria_results': criteria,
        'test_policy_episodes': executions, 'test_tuning_performed': False,
        'invalid_actions_all_evaluations': sum(value['invalid_actions_all_evaluations'] for value in complete.values()),
        'elapsed_seconds': time.monotonic()-started})
    print('Complete: ' + str(output / 'summary.json'), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'select', 'train', 'evaluate'))
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/paper_ppo_study')
    parser.add_argument('--scenario', type=Path, default=ROOT / 'configs/utility_b3.json')
    parser.add_argument('--control-root', type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS, help='Required for train phase')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    try:
        if args.phase == 'init':
            initialize(study, args.scenario, args.control_root)
        else:
            scenario, protocol, root_hash = load_study(study)
            if args.phase == 'select':
                select_rules(study, scenario, root_hash)
            elif args.phase == 'train':
                if args.seed is None:
                    parser.error('--seed is required for train phase')
                train_seed(study, scenario, protocol, root_hash, args.seed)
            else:
                evaluate_models(study, scenario, protocol, root_hash)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
