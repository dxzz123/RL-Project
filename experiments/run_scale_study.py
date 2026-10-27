"""Train and compare policies in the fixed synthetic 25-zone scenario.

Initialize the protocol, select rules on validation days, train each seed,
then evaluate frozen final checkpoints once on independent test days. The
coordinate inventory rule is a bounded search heuristic; the two-vehicle
approximate-DP implementation is not applicable to this ten-vehicle study.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from charging_env.config import Scenario
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer, _scenario_data
from charging_env.benchmarks.compact import candidates, inventory_shadow
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.benchmarks.policies import fixed_prices, service_first_dispatch
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from run_fitted_value_study import average_models, table_row

SEEDS = (7, 19, 37)
SCENARIO_FILE = 'configs/synthetic_25_zones.json'
MONITOR_SEEDS = [603000001 + 2*i for i in range(32)]
VALIDATION_SEEDS = [605000001 + 2*i for i in range(32)]
TEST_SEEDS = [621000001 + 2*i for i in range(256)]
ACTION_SEED = 629991
DEMONSTRATION_EPISODES = 256
FAMILIES = ('time_inventory', 'service_first')
PHYSICAL_FIELDS = ('mean_unmet_mobile_kwh', 'mean_unmet_fixed_kwh',
    'mean_fixed_delivered_kwh', 'mean_mobile_service_fulfillment',
    'mean_selected_service_fulfillment', 'mean_mobile_service_utilization',
    'mean_total_requests', 'mean_outside_option_share', 'mean_travel_kwh')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def config_for(seed):
    if seed not in SEEDS:
        raise ValueError('Choose training seed 7, 19, or 37')
    offset = 20000 * SEEDS.index(seed)
    return PaperPPOConfig(seed=seed, updates=500, episodes_per_update=16,
        epochs=6, minibatch_size=192, hidden_size=64,
        actor_learning_rate=1e-4, critic_learning_rate=1e-3,
        gae_lambda=1., normalize_advantages=True, reward_scale=20.,
        dispatch_entropy_coef=.005, target_kl=.03,
        warmstart_episodes=DEMONSTRATION_EPISODES, warmstart_epochs=100,
        warmstart_seed_base=602000000 + offset,
        warmstart_dispatch_solver='coordinate', warmstart_coordinate_sweeps=2,
        training_seed_base=600000000 + offset, eval_episodes=len(MONITOR_SEEDS),
        eval_interval=100, eval_seed_base=MONITOR_SEEDS[0],
        eval_action_seed=ACTION_SEED).validate()


def validate_splits(configurations, validation_seeds, test_seeds):
    """Check all training, demonstration, monitoring and holdout seed banks."""
    training, demonstrations, monitoring = set(), set(), set()
    for value in configurations.values():
        c = PaperPPOConfig(**value).validate()
        rollout = {2*(c.training_seed_base+c.seed+i)
                   for i in range(c.updates*c.episodes_per_update)}
        demos = {2*(c.warmstart_seed_base+c.seed+i) for i in range(c.warmstart_episodes)}
        if training & rollout or demonstrations & demos:
            raise ValueError('Training seeds must be disjoint across models')
        training |= rollout
        demonstrations |= demos
        monitoring |= {c.eval_seed_base + 2*i for i in range(c.eval_episodes)}
    blocks = [training, demonstrations, monitoring, set(validation_seeds), set(test_seeds)]
    if (len(validation_seeds) != len(blocks[3]) or len(test_seeds) != len(blocks[4])
            or any(blocks[i] & blocks[j] for i in range(len(blocks)) for j in range(i))):
        raise ValueError('Training, demonstrations, monitoring, validation and test seeds must be disjoint')


def rule_candidates(scenario):
    return [spec for spec in candidates(scenario) if spec['family'] in FAMILIES]


def evaluate_rule(scenario, seeds, spec, *, action_seed=ACTION_SEED, label=None):
    """Evaluate a frozen rule with its declared bounded search budget."""
    if spec not in rule_candidates(scenario):
        raise ValueError('Rule settings are outside the frozen scale-study grid')
    pricing = lambda obs, rng: fixed_prices(obs, scenario, spec['price_index'])
    solver_statistics = {}
    if spec['family'] == 'service_first':
        dispatch = lambda obs, rng: service_first_dispatch(obs, scenario, spec['recharge_threshold'])
    else:
        dispatch = lambda obs, rng: coordinate_inventory_dispatch(
            obs, scenario, inventory_shadow(obs, scenario, spec),
            sweeps=2, energy_fractions=(1.,), diagnostics=solver_statistics)
    result = evaluate_policy(scenario, seeds, pricing, dispatch,
        action_seed=action_seed, label=label or spec['id'])
    if result['invalid_actions']:
        raise RuntimeError('Rule emitted invalid actions: ' + result['label'])
    result['policy_specification'] = spec
    result['solver_statistics'] = solver_statistics if spec['family'] == 'time_inventory' else None
    return result


def init(study):
    if study.exists():
        raise FileExistsError('Choose an unused study directory')
    scenario = Scenario.from_json(ROOT / SCENARIO_FILE)
    if (scenario.n_zones, scenario.n_mcv, scenario.n_fcs) != (25, 10, 5):
        raise ValueError('Scale study requires 25 zones, 10 vehicles and 5 stations')
    configurations = {str(seed): asdict(config_for(seed)) for seed in SEEDS}
    validate_splits(configurations, VALIDATION_SEEDS, TEST_SEEDS)
    files = sorted((ROOT / 'src/charging_env').rglob('*.py')) + [
        ROOT / 'experiments/run_scale_study.py', ROOT / 'experiments/run_fitted_value_study.py',
        ROOT / SCENARIO_FILE]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    protocol = dict(schema='charging_env.scale_study.v1', scenario=asdict(scenario),
        training_seeds=list(SEEDS), configurations=configurations,
        scenario_description=dict(zones=25, vehicles=10, fixed_stations=5,
            footprint_km=[10., 10.], period_hours=scenario.period_hours,
            service_hours=scenario.horizon*scenario.period_hours,
            expected_daily_requests=float(np.sum(scenario.resolved_arrival_rates)),
            demand='Synthetic spatial and hourly intensities explicitly specified by arrival_rates; independent simulated arrivals, not observed city data.'),
        monitor_seeds=MONITOR_SEEDS, validation_seeds=VALIDATION_SEEDS,
        test_seeds=TEST_SEEDS, action_seed=ACTION_SEED, source_sha256=hashes,
        rule_candidates=rule_candidates(scenario), rule_selection='Best mean validation profit within each family; first candidate wins ties.',
        inventory_solver=dict(method='coordinate', sweeps=2, energy_fractions=[1.],
            action_scope='Native physically feasible actions. PPO teacher labels additionally obey the actor action mask.',
            globally_optimal=False),
        initialization='Fixed price 0.60 and inventory shadows 0.30 early, 0.15 late, zero terminal; bounded coordinate teacher.',
        checkpoint_rule='Final update 500, regardless of monitoring scores; also report initialization before any PPO update.',
        execution_mode='Stochastic PPO and cloning-only execution; deterministic rule implementations.',
        budget=f'{DEMONSTRATION_EPISODES} teacher episodes, 100 initialization epochs, and 8000 fresh on-policy episodes per model.',
        selection_budget='16 inventory schedules and 21 service-first settings on the same 32 validation days.',
        success_rule='PPO beats inventory consistently only if the paired 95% lower bound is positive for all three seeds.',
        approximate_dp=dict(supported=False, reason='The tabular fitted-value implementation supports two vehicles, not this ten-vehicle fleet.'),
        limitations='Synthetic same-distribution scenarios, not real-world data or a distribution-shift study. Intervals condition on three fixed trained models. Coordinate search is a bounded heuristic, not exhaustive optimization.')
    study.mkdir(parents=True)
    write(study / 'protocol.json', protocol)
    (study / 'protocol.sha256').write_text(digest(study / 'protocol.json') + '\n')
    for name in hashes:
        target = study / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    verify(study)
    print('Initialized ' + str(study), flush=True)


def verify(study):
    path = study / 'protocol.json'
    if digest(path) != (study / 'protocol.sha256').read_text().strip():
        raise ValueError('Frozen study protocol changed')
    p = json.loads(path.read_text())
    if (p['schema'] != 'charging_env.scale_study.v1' or p['training_seeds'] != list(SEEDS)
            or p['configurations'] != {str(seed): asdict(config_for(seed)) for seed in SEEDS}
            or p['monitor_seeds'] != MONITOR_SEEDS or p['validation_seeds'] != VALIDATION_SEEDS
            or p['test_seeds'] != TEST_SEEDS or p['action_seed'] != ACTION_SEED):
        raise ValueError('Study settings differ from the prescribed scale protocol')
    validate_splits(p['configurations'], p['validation_seeds'], p['test_seeds'])
    scenario = _scenario_data(p['scenario'])
    if scenario != Scenario.from_json(ROOT / SCENARIO_FILE) or p['rule_candidates'] != rule_candidates(scenario):
        raise ValueError('Scenario or benchmark grid changed')
    for name, expected in p['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(study / 'source_snapshot' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    return p


def select(study):
    p = verify(study)
    directory = study / 'rule_selection'
    if directory.exists():
        raise FileExistsError('Rule selection already exists')
    scenario = _scenario_data(p['scenario'])
    rows, result_hashes = [], {}
    started = time.perf_counter()
    directory.mkdir()
    for spec in p['rule_candidates']:
        result = evaluate_rule(scenario, p['validation_seeds'], spec, action_seed=p['action_seed'])
        path = directory / (spec['id'] + '.json')
        write(path, result)
        result_hashes[path.name] = digest(path)
        rows.append(dict(specification=spec, mean_discounted_profit=result['mean_discounted_profit']))
        print(f"Validation {spec['id']}: {result['mean_discounted_profit']:.3f}", flush=True)
    winners = {family: max((row for row in rows if row['specification']['family'] == family),
        key=lambda row: row['mean_discounted_profit']) for family in FAMILIES}
    verify(study)
    write(directory / 'selection.json', dict(protocol_sha256=digest(study / 'protocol.json'),
        validation_seeds=p['validation_seeds'], rows=rows, winners=winners,
        result_sha256=result_hashes, seconds=time.perf_counter()-started, test_scenarios_accessed=False))
    (directory / 'selection.sha256').write_text(digest(directory / 'selection.json') + '\n')
    verify_selection(study, p)


def verify_selection(study, protocol):
    directory = study / 'rule_selection'
    path = directory / 'selection.json'
    if digest(path) != (directory / 'selection.sha256').read_text().strip():
        raise ValueError('Frozen rule selection changed')
    s = json.loads(path.read_text())
    if (s['protocol_sha256'] != digest(study / 'protocol.json')
            or s['validation_seeds'] != protocol['validation_seeds']
            or s['test_scenarios_accessed'] is not False
            or [row['specification'] for row in s['rows']] != protocol['rule_candidates']):
        raise ValueError('Rule selection does not match the frozen validation protocol')
    if set(s['result_sha256']) != {spec['id'] + '.json' for spec in protocol['rule_candidates']}:
        raise ValueError('Rule selection evidence is incomplete')
    for row in s['rows']:
        name = row['specification']['id'] + '.json'
        if digest(directory / name) != s['result_sha256'][name]:
            raise ValueError('Validation evidence changed: ' + name)
        result = json.loads((directory / name).read_text())
        if (result['evaluation_seeds'] != protocol['validation_seeds'] or result['invalid_actions']
                or result['policy_specification'] != row['specification']
                or not np.isfinite(row['mean_discounted_profit'])
                or result['mean_discounted_profit'] != row['mean_discounted_profit']):
            raise ValueError('Validation score has inconsistent evidence: ' + name)
    expected = {family: max((row for row in s['rows'] if row['specification']['family'] == family),
        key=lambda row: row['mean_discounted_profit']) for family in FAMILIES}
    if s['winners'] != expected:
        raise ValueError('Selected rules do not maximize the frozen validation scores')
    return s


def train(study, seed):
    p = verify(study)
    if seed not in SEEDS:
        raise ValueError('Training requires --seed 7, 19, or 37')
    trainer = PaperPPOTrainer(_scenario_data(p['scenario']), PaperPPOConfig(**p['configurations'][str(seed)]))
    trainer.train(study / 'training' / f'seed_{seed}')
    verify(study)


def validate_model(model, protocol, seed, *, initialized_only=False):
    scenario = _scenario_data(protocol['scenario'])
    c = model.config
    if model.scenario != scenario or asdict(c) != protocol['configurations'][str(seed)]:
        raise ValueError('Checkpoint scenario or configuration differs from the frozen protocol')
    updates = 0 if initialized_only else c.updates
    episodes = updates*c.episodes_per_update
    if (model.update_count, model.episode_count, model.environment_periods) != (updates, episodes, episodes*scenario.horizon):
        raise ValueError('Checkpoint does not have the required fresh training budget')
    if not model.warmstart_metadata['completed']:
        raise ValueError('Checkpoint has not completed the prescribed initialization')
    warm = model.warmstart_metadata['summary']
    expected_seeds = [2*(c.warmstart_seed_base+c.seed+i) for i in range(c.warmstart_episodes)]
    expected_sources = model._source_hashes()
    if (warm['seeds'] != expected_seeds or warm['demonstration_episodes'] != c.warmstart_episodes
            or warm['demonstration_periods'] != c.warmstart_episodes*scenario.horizon
            or warm['epochs'] != c.warmstart_epochs or warm['reward_scale'] != c.reward_scale
            or warm.get('dispatch_solver') != 'coordinate' or warm.get('coordinate_sweeps') != 2
            or warm.get('solver_exact') is not False or warm['source_sha256'] != expected_sources
            or any(protocol['source_sha256'].get(k) != v for k, v in expected_sources.items())):
        raise ValueError('Initialization provenance differs from the frozen protocol')
    validate_splits({str(seed): asdict(c)}, protocol['validation_seeds'], protocol['test_seeds'])


def prepare_evaluation(study, protocol):
    selection = verify_selection(study, protocol)
    paths = {f'{family}/{seed}': study / 'training' / f'seed_{seed}' / filename
        for seed in SEEDS for family, filename in
        (('ppo', 'checkpoint.pt'), ('cloning_only', 'warmstart_checkpoint.pt'))}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError('Evaluation requires all six complete PPO and initialization checkpoints')
    for key, path in paths.items():
        family, seed = key.split('/')
        validate_model(PaperPPOTrainer.load_checkpoint(path), protocol, int(seed),
            initialized_only=family == 'cloning_only')
    return selection, paths, {key: digest(path) for key, path in paths.items()}


def evaluate(study):
    p = verify(study)
    destination = study / 'test'
    if destination.exists():
        raise FileExistsError('Independent tests already exist; use --phase show')
    selection, paths, hashes = prepare_evaluation(study, p)
    selection_hash = digest(study / 'rule_selection/selection.json')
    destination.mkdir()
    write(destination / 'frozen_inputs.json', dict(models=hashes, selection_sha256=selection_hash,
        protocol_sha256=digest(study / 'protocol.json')))
    started = time.perf_counter()
    scenario = _scenario_data(p['scenario'])
    results = {'ppo': [], 'cloning_only': []}
    for key, path in paths.items():
        family, seed = key.split('/')
        policy = PaperPPOTrainer.load_checkpoint(path)
        result = policy.evaluate(p['test_seeds'], False, p['action_seed'])
        if result['invalid_actions']:
            raise RuntimeError('Neural policy emitted invalid actions: ' + key)
        result.update(label=key, training_seed=int(seed), checkpoint_sha256=hashes[key])
        results[family].append(result)
        write(destination / (key.replace('/', '_') + '.json'), result)
        print(f"{key}: test profit {result['mean_discounted_profit']:.3f}", flush=True)
    rules = {}
    for family in FAMILIES:
        name = 'inventory_rule' if family == 'time_inventory' else 'service_first'
        result = evaluate_rule(scenario, p['test_seeds'], selection['winners'][family]['specification'],
            action_seed=p['action_seed'], label=name)
        rules[name] = result
        write(destination / (name + '.json'), result)
    aggregate = {family: average_models(items, family) for family, items in results.items()}
    for family, items in results.items():
        for metric in PHYSICAL_FIELDS:
            values = [item[metric] for item in items if item[metric] is not None]
            aggregate[family][metric] = float(np.mean(values)) if values else None
    aggregate.update(rules)
    comparisons = {name: paired_comparison(aggregate['ppo'], aggregate[name])
        for name in ('cloning_only', 'inventory_rule', 'service_first')}
    per_seed = {str(seed): {
        'inventory_rule': paired_comparison(results['ppo'][index], rules['inventory_rule']),
        'cloning_only': paired_comparison(results['ppo'][index], results['cloning_only'][index]),
        'service_first': paired_comparison(results['ppo'][index], rules['service_first'])}
        for index, seed in enumerate(SEEDS)}
    if hashes != {key: digest(path) for key, path in paths.items()}:
        raise RuntimeError('A checkpoint changed during test evaluation')
    if digest(study / 'rule_selection/selection.json') != selection_hash:
        raise RuntimeError('Benchmark selection changed during test evaluation')
    verify(study)
    verify_selection(study, p)
    training_summaries = {str(seed): json.loads((study / 'training' / f'seed_{seed}' / 'summary.json').read_text()) for seed in SEEDS}
    training_seconds = {seed: summary['training_seconds'] for seed, summary in training_summaries.items()}
    initialization_seconds = {seed: summary['warmstart']['seconds'] for seed, summary in training_summaries.items()}
    analysis = dict(protocol_sha256=digest(study / 'protocol.json'), checkpoint_sha256=hashes,
        selection_sha256=selection_hash,
        scorecard=[dict(table_row(name, result), **{metric: result[metric] for metric in PHYSICAL_FIELDS})
            for name, result in aggregate.items()],
        per_seed=per_seed, comparisons=comparisons,
        by_model={family: {str(result['training_seed']): result['mean_discounted_profit'] for result in items}
            for family, items in results.items()},
        consistent_inventory_advantage=all(per_seed[str(seed)]['inventory_rule']['paired_bootstrap_confidence_interval_95'][0] > 0 for seed in SEEDS),
        consistent_gain_over_initialization=all(per_seed[str(seed)]['cloning_only']['paired_bootstrap_confidence_interval_95'][0] > 0 for seed in SEEDS),
        invalid_actions=sum(result['invalid_actions'] for items in results.values() for result in items)
            + sum(result['invalid_actions'] for result in rules.values()),
        reward_accounting={name: result['mean_discounted_reward_terms'] for name, result in aggregate.items()},
        policy_episodes=len(p['test_seeds'])*(len(paths)+len(rules)),
        training_seconds_by_seed=training_seconds, initialization_seconds_by_seed=initialization_seconds,
        recorded_total_seconds_by_seed={seed: training_seconds[seed]+initialization_seconds[seed] for seed in training_seconds},
        runtime_scope='Initialization includes demonstration collection and supervised fitting. Training includes PPO updates, monitoring and checkpoint replay; their sum excludes initial random-policy and initialization-policy evaluations.',
        test_seconds=time.perf_counter()-started,
        approximate_dp=p['approximate_dp'],
        uncertainty='Paired scenario bootstrap, 5000 resamples; conditional on fixed trained models, excludes retraining uncertainty.')
    write(study / 'analysis.json', analysis)
    show(study)


def show(study):
    data = json.loads((study / 'analysis.json').read_text())
    print(f"{'Method':18} {'Profit/day':>12} {'Recharge':>10} {'Mobile kWh':>12} {'End battery':>12} {'Unmet mobile':>13}")
    for row in data['scorecard']:
        print(f"{row['policy']:18} {row['mean_discounted_profit']:12.2f} {row['mean_charge_grid_kwh']:10.2f} {row['mean_mobile_delivered_kwh']:12.2f} {row['mean_final_inventory_kwh']:12.2f} {row['mean_unmet_mobile_kwh']:13.2f}")
    print('Consistent improvement over inventory rule:', data['consistent_inventory_advantage'])
    print('Consistent improvement over initialization:', data['consistent_gain_over_initialization'])
    print('Approximate DP:', data['approximate_dp']['reason'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'select', 'train', 'evaluate', 'show'))
    parser.add_argument('--study-dir', type=Path, default=Path('outputs/scale_25'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == 'init': init(args.study_dir)
    elif args.phase == 'select': select(args.study_dir)
    elif args.phase == 'train': train(args.study_dir, args.seed)
    elif args.phase == 'evaluate': evaluate(args.study_dir)
    else: show(args.study_dir)


if __name__ == '__main__':
    main()
