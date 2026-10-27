"""Compare online rollout ADP, frozen PPO and rules on independent scenarios.

Run init, rules, validate, select, evaluate, show. Rollout ADP has no offline
training or learned checkpoint. Validation selects its fixed price across all
three planning seeds. Final testing uses frozen settings and separate scenes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments'))
from charging_env.config import Scenario
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer, _scenario_data
from charging_env.rl.fitted_value import FittedValueConfig, FittedValuePolicy, FittedValueTrainer
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy, evaluate_rollout_policy
from charging_env.benchmarks.policies import fixed_prices, service_first_dispatch
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch, _individual_candidates
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison

SEEDS = (7, 19, 37)
PRICE_INDICES = (4, 5)
CASES = {'small': 'configs/utility_b3.json', 'city': 'configs/synthetic_25_zones.json'}
VALIDATION_SEEDS = {case: [base+2*i for i in range(32)]
    for case, base in [('small', 697000001), ('city', 699000001)]}
TEST_SEEDS = {case: [base+2*i for i in range(256)]
    for case, base in [('small', 701000001), ('city', 703000001)]}
ACTION_SEED = 709991
SCHEDULES = ((0., 0.), (.1, 0.), (.2, 0.), (.2, .1), (.3, .15), (.4, 0.), (.4, .1))
FAMILIES = ('inventory_rule', 'service_first')
PHYSICAL_FIELDS = ('mean_unmet_mobile_kwh', 'mean_unmet_fixed_kwh', 'mean_fixed_delivered_kwh',
    'mean_mobile_service_fulfillment', 'mean_selected_service_fulfillment',
    'mean_mobile_service_utilization', 'mean_total_requests', 'mean_outside_option_share',
    'mean_travel_kwh', 'mean_mobile_selected_requests', 'mean_mobile_served_requests')
CONTROL_FILES = ('protocol.json', 'scenario.json', 'training_config.json', 'summary.json', 'training_metrics.csv')
SCHEMA = 'charging_env.rollout_adp_study.v1'


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text())


def seal(path, value):
    write(path, value)
    path.with_suffix('.sha256').write_text(digest(path)+'\n')


def sealed(path):
    if digest(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen evidence changed: '+str(path))
    return read(path)


def finite(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Evidence contains a nonfinite number')
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)


def control_seeds(metadata):
    c = metadata['training_config']
    if metadata['family'] == 'ppo':
        return ({2*(c['training_seed_base']+c['seed']+i) for i in range(metadata['episode_count'])}
            | {c['eval_seed_base']+2*i for i in range(c['eval_episodes'])}
            | {2*(c['warmstart_seed_base']+c['seed']+i) for i in range(c['warmstart_episodes'])})
    return set(sum(metadata['period_draw_seeds'], []))


def rule_candidates(scenario):
    if len(scenario.price_levels) != 7:
        raise ValueError('This study requires the seven configured price levels')
    return ([dict(id=f'service_p{p}_b{b:g}', family='service_first', price_index=p, recharge_threshold=b)
             for p in range(7) for b in (.2, .4, .6)]
        + [dict(id=f'inventory_p{p}_e{e:g}_l{l:g}', family='inventory_rule', price_index=p,
                early_shadow_price=e, late_shadow_price=l, switch_period=scenario.horizon//2)
           for p in range(7) for e, l in SCHEDULES])


def check_result(result, seeds):
    finite(result)
    if (result['evaluation_seeds'] != list(seeds) or result['invalid_actions']
            or result['n_episodes'] != len(seeds)
            or [row['seed'] for row in result['episodes']] != list(seeds)):
        raise ValueError('Evaluation has invalid actions or inconsistent scenario evidence')
    for row in result['episodes']:
        terms = row['discounted_reward_terms']
        accounted = sum(terms[k] for k in ('mobile_service_revenue', 'fixed_service_revenue',
            'discharge_revenue', 'terminal_inventory_value'))-sum(terms[k] for k in
            ('procurement_cost', 'travel_cost', 'operating_cost', 'unmet_demand_cost'))
        if not np.isclose(row['discounted_profit'], accounted, rtol=0., atol=1e-7):
            raise ValueError('Episode profit disagrees with its reward ledger')
    episode_mean = np.mean([row['discounted_profit'] for row in result['episodes']])
    if not np.isclose(result['mean_discounted_profit'], episode_mean, rtol=0., atol=1e-9):
        raise ValueError('Reported mean profit disagrees with the stored episode profits')


def counted_dispatch(callback, statistics):
    def dispatch(obs, rng):
        actions = callback(obs, rng)
        statistics['candidate_budget_total'] = statistics.get('candidate_budget_total', 0)+statistics['last_call']['candidate_budget']
        return actions
    return dispatch


def evaluate_rule(scenario, seeds, spec, *, label=None):
    if spec not in rule_candidates(scenario):
        raise ValueError('Rule is outside the frozen validation grid')
    statistics = {}
    if spec['family'] == 'service_first':
        dispatch = lambda obs, rng: service_first_dispatch(obs, scenario, spec['recharge_threshold'])
    else:
        def choose(obs, rng):
            t = int(obs['period'][0])
            shadow = spec['early_shadow_price'] if t < spec['switch_period'] else spec['late_shadow_price']
            return coordinate_inventory_dispatch(obs, scenario, shadow, sweeps=2,
                energy_fractions=(1.,), diagnostics=statistics)
        dispatch = counted_dispatch(choose, statistics)
    result = evaluate_policy(scenario, seeds,
        lambda obs, rng: fixed_prices(obs, scenario, spec['price_index']), dispatch,
        action_seed=ACTION_SEED, label=label or spec['id'])
    check_result(result, seeds)
    result.update(policy_specification=spec, solver_statistics=statistics or None)
    return result


def control_locations(root, case, family, seed):
    if family == 'original_adp':
        return (root/'outputs/paper_ppo_study/fitted'/f'seed_{seed}',
                root/'outputs/paper_ppo_study/provenance/executed_training_source')
    name = 'recharge_study_20261024' if case == 'small' else 'scale_25_study_20261024'
    return root/'outputs'/name/'training'/f'seed_{seed}', root/'outputs'/name/'source_snapshot'


def inspect_control(directory, source_root, case, family, seed, scenario):
    """Validate the model against its own original training evidence and source."""
    name = 'checkpoint.pt' if family == 'ppo' else 'checkpoint.npz'
    path = directory/name
    protocol, summary = read(directory/'protocol.json'), read(directory/'summary.json')
    finite(summary)
    model = (PaperPPOTrainer if family == 'ppo' else FittedValuePolicy).load_checkpoint(path)
    if model.scenario != scenario or model.config.seed != seed:
        raise ValueError('Frozen control scenario or seed mismatch')
    if (summary['checkpoint_sha256'] != digest(path) or summary['protocol_sha256'] != digest(directory/'protocol.json')
            or _scenario_data(protocol['scenario']) != scenario
            or _scenario_data(read(directory/'scenario.json')) != scenario):
        raise ValueError('Frozen control checkpoint or protocol provenance differs')
    for file, expected in protocol['source_sha256'].items():
        if digest(source_root/file) != expected:
            raise ValueError('Original control source snapshot differs: '+file)
    with (directory/'training_metrics.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    if not metrics or any(not np.isfinite(float(v)) for row in metrics for v in row.values()):
        raise ValueError('Control training metrics are missing or nonfinite')
    if family == 'ppo':
        from run_recharge_study import config_for as small_config
        from run_scale_study import config_for as city_config
        expected = (small_config if case == 'small' else city_config)(seed)
        if (model.config != expected or PaperPPOConfig(**protocol['training_config']) != expected
                or PaperPPOConfig(**read(directory/'training_config.json')) != expected
                or (model.update_count, model.episode_count, model.environment_periods) != (500, 8000, 8000*scenario.horizon)
                or len(metrics) != 500 or not summary['all_metrics_finite']
                or not summary['checkpoint_reload_matches'] or summary['invalid_evaluation_actions']):
            raise ValueError('PPO control does not match its complete frozen 500-update run')
        warm = model.warmstart_metadata
        if (not warm['completed'] or warm['summary'] != read(directory/'warmstart_summary.json')
                or warm['summary']['source_sha256'] != protocol['source_sha256']
                or warm['summary']['seeds'] != [2*(expected.warmstart_seed_base+seed+i)
                    for i in range(expected.warmstart_episodes)]):
            raise ValueError('PPO control warm-start provenance differs')
        metadata = dict(episode_count=model.episode_count)
    else:
        expected = FittedValueConfig(seed=seed)
        actual = dict(protocol['training_config']); actual['energy_fractions'] = tuple(actual['energy_fractions'])
        saved = dict(read(directory/'training_config.json')); saved['energy_fractions'] = tuple(saved['energy_fractions'])
        trainer = FittedValueTrainer(scenario, expected)
        queries = scenario.horizon*scenario.n_zones**2*len(model.energy_grid)**2*expected.samples_per_period
        provenance = model.training_provenance
        if (case != 'small' or model.config != expected or FittedValueConfig(**actual) != expected
                or FittedValueConfig(**saved) != expected or len(metrics) != scenario.horizon
                or not summary['all_values_finite'] or not summary['exact_checkpoint_reload']
                or summary['completed_layers'] != scenario.horizon or summary['state_draw_queries'] != queries
                or provenance['state_draw_queries'] != queries
                or provenance['protocol_sha256'] != digest(directory/'protocol.json')
                or provenance['source_sha256'] != protocol['source_sha256']
                or protocol['period_draw_seeds'] != trainer.query_seeds()):
            raise ValueError('Original ADP control training provenance differs')
        metadata = dict(period_draw_seeds=trainer.query_seeds())
    return dict(case=case, family=family, seed=seed, training_config=normalized(asdict(model.config)),
                source_sha256=protocol['source_sha256'], checkpoint=name, **metadata)


def selected_cases(case):
    if case is not None and case not in CASES:
        raise ValueError('Unknown study case')
    return list(CASES) if case is None else [case]


def rules(study, case=None):
    p = verify(study)
    for name in selected_cases(case):
        data = p['cases'][name]
        directory = study/'rule_selection'/name
        if directory.exists():
            raise FileExistsError('Rule selection already exists: '+name)
        directory.mkdir(parents=True)
        rows, hashes = [], {}
        started = time.perf_counter()
        for spec in data['rule_candidates']:
            result = evaluate_rule(_scenario_data(data['scenario']), data['validation_seeds'], spec)
            path = directory/(spec['id']+'.json'); write(path, result)
            hashes[path.name] = digest(path)
            rows.append(dict(specification=spec, mean_discounted_profit=result['mean_discounted_profit']))
            print(f"{name} validation {spec['id']}: {result['mean_discounted_profit']:.3f}", flush=True)
        winners = {family: max((row for row in rows if row['specification']['family'] == family),
            key=lambda row: row['mean_discounted_profit']) for family in FAMILIES}
        verify(study)
        seal(directory/'selection.json', dict(protocol_sha256=digest(study/'protocol.json'),
            validation_seeds=data['validation_seeds'], rows=rows, winners=winners,
            result_sha256=hashes, seconds=time.perf_counter()-started, test_scenarios_accessed=False))
        verify_rules(study, p, name)


def verify_rules(study, p, case):
    data, directory = p['cases'][case], study/'rule_selection'/case
    selection = sealed(directory/'selection.json')
    if (selection['protocol_sha256'] != digest(study/'protocol.json')
            or selection['validation_seeds'] != data['validation_seeds']
            or selection['test_scenarios_accessed'] is not False
            or [r['specification'] for r in selection['rows']] != data['rule_candidates']
            or set(selection['result_sha256']) != {s['id']+'.json' for s in data['rule_candidates']}):
        raise ValueError('Rule selection differs from the validation protocol')
    for row in selection['rows']:
        path = directory/(row['specification']['id']+'.json')
        if digest(path) != selection['result_sha256'][path.name]:
            raise ValueError('Rule validation evidence changed')
        result = read(path); check_result(result, data['validation_seeds'])
        if (result['mean_discounted_profit'] != row['mean_discounted_profit']
                or result['policy_specification'] != row['specification']):
            raise ValueError('Rule validation score or specification differs')
    expected = {family: max((r for r in selection['rows'] if r['specification']['family'] == family),
        key=lambda r: r['mean_discounted_profit']) for family in FAMILIES}
    if selection['winners'] != expected:
        raise ValueError('Rules do not maximize the frozen validation scores')
    return selection


def evaluate_original(model, seeds, label):
    statistics = dict(calls=0, candidate_outcomes=0, max_candidates_per_call=0)
    def dispatch(obs, rng):
        choices = _individual_candidates(obs, model.scenario, np.asarray(obs['action_mask'], dtype=bool),
                                         np.unique(model.config.energy_fractions))
        count = 1
        for options in choices: count *= len(options)
        statistics['calls'] += 1; statistics['candidate_outcomes'] += count
        statistics['max_candidates_per_call'] = max(statistics['max_candidates_per_call'], count)
        return model.dispatch(obs, rng)
    result = evaluate_policy(model.scenario, seeds, model.prices, dispatch, action_seed=ACTION_SEED, label=label)
    check_result(result, seeds)
    result['solver_statistics'] = dict(statistics, counting='Exact finite-grid cardinality; no duplicate physical evaluation.')
    return result


def show(study):
    analysis = read(study/'analysis.json')
    for case, data in analysis['cases'].items():
        print(f"\n{case}: selected ADP price {data['selected_adp_price']:.2f}")
        print(f"{'Method':20} {'Profit/day':>11} {'Mobile kWh':>11} {'Recharge':>10} {'Unmet kWh':>10} {'End > reserve':>14}")
        for row in data['scorecard']:
            print(f"{row['policy']:20} {row['mean_discounted_profit']:11.2f} {row['mean_mobile_delivered_kwh']:11.2f} "
                  f"{row['mean_charge_grid_kwh']:10.2f} {row['mean_unmet_mobile_kwh']:10.2f} {row['mean_battery_above_reserve_kwh']:14.2f}")
        for name, comparison in data['comparisons'].items():
            lo, hi = comparison['paired_bootstrap_confidence_interval_95']
            print(f"ADP minus {name}: {comparison['mean_difference']:.2f}, paired 95% CI [{lo:.2f}, {hi:.2f}]")
    print('\n'+analysis['uncertainty'])



def config_for(case, seed, price_index):
    if case not in CASES or seed not in SEEDS or price_index not in PRICE_INDICES:
        raise ValueError('Unknown case, planning seed or fixed price')
    return RolloutADPConfig(seed=seed, rollout_samples=8, price_index=price_index,
        base_recharge_threshold=.2, alternative_recharge_thresholds=(.4, .6),
        inventory_early_shadow=.3, inventory_late_shadow=.15, planner_sweeps=2,
        energy_fractions=(1.,))


def config_from_data(value):
    copied = dict(value)
    for key in ('alternative_recharge_thresholds', 'energy_fractions'): copied[key] = tuple(copied[key])
    return RolloutADPConfig(**copied)


def configuration(case, scenario):
    return {str(price): {str(seed): normalized(asdict(config_for(case, seed, price).validate(scenario)))
        for seed in SEEDS} for price in PRICE_INDICES}


def validate_splits(protocol):
    previous = set().union(*(control_seeds(m) for m in protocol.get('controls', {}).values()))
    for data in protocol['cases'].values():
        for kind in ('validation_seeds', 'test_seeds'):
            values = data[kind]
            if (not values or len(set(values)) != len(values) or previous.intersection(values)
                    or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s % 2 != 1 for s in values)):
                raise ValueError('Control training, validation and test banks must be disjoint')
            previous.update(values)


def init(study, control_root):
    if study.exists(): raise FileExistsError('Choose an unused rollout study directory')
    cases = {}
    for case, file in CASES.items():
        scenario = Scenario.from_json(ROOT/file)
        if (scenario.n_zones, scenario.n_mcv) != ((4, 2) if case == 'small' else (25, 10)):
            raise ValueError('Scenario dimensions differ from the two frozen cases')
        cases[case] = dict(scenario=normalized(asdict(scenario)), scenario_file=file,
            configurations=configuration(case, scenario), validation_seeds=VALIDATION_SEEDS[case],
            test_seeds=TEST_SEEDS[case], rule_candidates=rule_candidates(scenario))
    controls, copies = {}, {}
    for case, data in cases.items():
        for family in (('ppo', 'original_adp') if case == 'small' else ('ppo',)):
            for seed in SEEDS:
                directory, sources = control_locations(control_root, case, family, seed)
                key = f'{case}/{family}/seed_{seed}'
                metadata = inspect_control(directory, sources, case, family, seed, _scenario_data(data['scenario']))
                controls[key] = metadata
                for name in (*CONTROL_FILES, metadata['checkpoint'], *(('warmstart_summary.json',) if family == 'ppo' else ())):
                    copies[f'{key}/{name}'] = directory/name
                for name in metadata['source_sha256']:
                    copies[f'{case}/{family}/source_snapshot/{name}'] = sources/name
    files = sorted((ROOT/'src/charging_env').rglob('*.py'))+[ROOT/'experiments'/name for name in
        ('run_general_adp_study.py', 'run_recharge_study.py', 'run_scale_study.py')]
    files += [ROOT/name for name in CASES.values()]
    protocol = dict(schema=SCHEMA, cases=cases, planning_seeds=list(SEEDS), control_training_seeds=list(SEEDS),
        price_indices=list(PRICE_INDICES), action_seed=ACTION_SEED, controls=controls,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in files},
        control_sha256={name: digest(path) for name, path in copies.items()},
        offline_training=False, final_policy_episodes=5376,
        selection='Each rule family maximizes validation mean. Rollout ADP selects one price per case by the mean of planning seeds 7,19,37; ties use declared order.',
        planning='Five proposal families before exact action deduplication: service-first thresholds .2,.4,.6; inventory .3 early/.15 late; zero shadow. Eight full remaining-horizon service-first(.2) continuations per unique plan, with paired planning draws.',
        planning_rng='Local SeedSequence([planning_seed, episode_key, 0x524F4C4C]); fresh even future-model seeds, shared across candidate plans, no live simulator RNG access. All exact tables are saved.',
        information='Rollout ADP uses known one-period physics and online future model queries. PPO controls are frozen stochastic policies. These are not equal-information or equal-compute methods.',
        action_scope='Inventory proposal and zero control use native feasible masks, two coordinate sweeps and maximum exchange. Service-first proposals retain their physical commands. Original small ADP retains exhaustive half/full exchange. PPO retains its four-operation mask.',
        zero_control='Zero continuation uses the selected rollout price and the same zero-shadow coordinate inventory proposal.',
        uncertainty='Paired scenario bootstrap conditional on frozen policies and planning seeds; excludes retraining uncertainty.',
        limitations='Both cases are simulated under their training distributions. Restricted candidate plans and finite rollouts give no optimality, policy-improvement or convergence guarantee.')
    validate_splits(protocol)
    study.mkdir(parents=True); seal(study/'protocol.json', protocol)
    for name in protocol['source_sha256']:
        target = study/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    for name, source in copies.items():
        target = study/'controls'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    verify(study)
    print('Initialized '+str(study), flush=True)


def verify(study):
    p = sealed(study/'protocol.json')
    if (p['schema'] != SCHEMA or p['planning_seeds'] != list(SEEDS)
            or p['control_training_seeds'] != list(SEEDS) or p['price_indices'] != list(PRICE_INDICES)
            or p['action_seed'] != ACTION_SEED or p['offline_training'] is not False
            or set(p['cases']) != set(CASES)):
        raise ValueError('Study protocol differs from the prescribed comparison')
    for case, data in p['cases'].items():
        scenario = Scenario.from_json(ROOT/CASES[case])
        if (data['scenario'] != normalized(asdict(scenario)) or data['configurations'] != configuration(case, scenario)
                or data['validation_seeds'] != VALIDATION_SEEDS[case] or data['test_seeds'] != TEST_SEEDS[case]
                or data['rule_candidates'] != rule_candidates(scenario)):
            raise ValueError('Case settings or holdouts changed: '+case)
    for name, expected in p['source_sha256'].items():
        if digest(ROOT/name) != expected or digest(study/'source_snapshot'/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    for name, expected in p['control_sha256'].items():
        if digest(study/'controls'/name) != expected: raise ValueError('Frozen control changed: '+name)
    validate_splits(p)
    return p


def check_planning_trace(trace, statistics, scenario, config, seeds):
    """Reconstruct local query RNG and exact budget from saved decision records."""
    if len(trace) != len(seeds)*scenario.horizon:
        raise ValueError('Planning trace is incomplete')
    totals = dict(decisions=len(trace), candidate_plans=0, current_plan_outcomes=0,
        future_model_queries=0, exogenous_draws=0, coordinate_candidate_outcomes=0)
    query_seeds = set()
    names_allowed = {'service_first_base', 'inventory', 'current_profit', 'service_first_0.4', 'service_first_0.6'}
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, seed, 0x524F4C4C]))
        for t in range(scenario.horizon):
            row = trace[i*scenario.horizon+t]
            remaining = scenario.horizon-t-1
            expected = 2*rng.integers(0, np.iinfo(np.int64).max//2,
                                      size=(remaining, config.rollout_samples), dtype=np.int64)
            while len(np.unique(expected)) != expected.size:
                expected = 2*rng.integers(0, np.iinfo(np.int64).max//2,
                                          size=(remaining, config.rollout_samples), dtype=np.int64)
            count = len(row['candidates'])
            if (row['episode_key'] != seed or row['period'] != t or row['episode_decision'] != t
                    or not 1 <= count <= 5 or row['candidates'][0] != 'service_first_base'
                    or len(set(row['candidates'])) != count or not set(row['candidates']) <= names_allowed
                    or row['future_periods'] != list(range(t+1, scenario.horizon))
                    or row['future_draw_seeds'] != expected.tolist()):
                raise ValueError('Planning trace or local query seed protocol differs')
            scores = np.asarray(row['sampled_total_scores'])
            if scores.shape != (count, config.rollout_samples) or not np.isfinite(scores).all():
                raise ValueError('Invalid sampled rollout values')
            means = scores.mean(axis=1)
            if (not np.array_equal(means, row['mean_total_scores'])
                    or row['selected'] != row['candidates'][int(np.argmax(means))]
                    or row['future_model_queries'] != count*config.rollout_samples*remaining
                    or row['query_budget'] != row['future_model_queries']
                    or row['coordinate_candidate_outcomes'] < 2 or row['seconds'] < 0):
                raise ValueError('Rollout selection, timing or query budget differs')
            if not remaining and not np.array_equal(scores, np.repeat(np.asarray(row['current_profit'])[:, None], config.rollout_samples, axis=1)):
                raise ValueError('Terminal rollout must contain current profit only')
            for key in ('future_model_queries', 'coordinate_candidate_outcomes'): totals[key] += row[key]
            totals['candidate_plans'] += count; totals['current_plan_outcomes'] += count
            totals['exogenous_draws'] += expected.size
            query_seeds.update(map(int, expected.ravel()))
    if any(statistics.get(key) != value for key, value in totals.items()):
        raise ValueError('Planning statistics disagree with the decision trace')
    if (statistics['boundary_roundoff_corrections'] < 0
            or not 0 <= statistics['max_boundary_roundoff_kwh'] <= 1e-8):
        raise ValueError('Planning battery correction exceeds numerical roundoff')
    return query_seeds


def rollout_paths(path):
    return path.with_name(path.stem+'_planning_trace.json'), path.with_name(path.stem+'_provenance.json')


def evaluate_rollout(scenario, config, seeds, path, p, *, label):
    if any(x.exists() for x in (path, *rollout_paths(path))): raise FileExistsError('Rollout output already exists')
    policy = RolloutADPPolicy(scenario, config)
    initial = policy.provenance()
    if any(p['source_sha256'].get(k) != v for k, v in initial['source_sha256'].items()):
        raise ValueError('Rollout source differs from study protocol')
    result = evaluate_rollout_policy(policy, seeds, action_seed=ACTION_SEED, label=label)
    check_result(result, seeds)
    provenance = policy.provenance()
    if provenance['source_sha256'] != initial['source_sha256']: raise RuntimeError('Rollout source changed')
    check_planning_trace(policy.decisions, policy.statistics, scenario, config, seeds)
    trace_path, provenance_path = rollout_paths(path)
    write(trace_path, policy.decisions); write(provenance_path, provenance)
    result.update(planning_seed=config.seed, price_index=config.price_index,
        planning_trace_sha256=digest(trace_path), planning_provenance_sha256=digest(provenance_path))
    write(path, result)
    return result


def inspect_rollout(path, p, case, price, seed, seeds):
    scenario = _scenario_data(p['cases'][case]['scenario'])
    config = config_from_data(p['cases'][case]['configurations'][str(price)][str(seed)])
    result = read(path); check_result(result, seeds)
    trace_path, provenance_path = rollout_paths(path)
    provenance = read(provenance_path); finite(provenance)
    expected = normalized(RolloutADPPolicy(scenario, config).provenance())
    expected['statistics'] = result['planning_statistics']
    if (provenance != expected or result['planning_seed'] != seed or result['price_index'] != price
            or result['planning_trace_sha256'] != digest(trace_path)
            or result['planning_provenance_sha256'] != digest(provenance_path)
            or any(p['source_sha256'].get(k) != v for k, v in provenance['source_sha256'].items())):
        raise ValueError('Rollout configuration or provenance differs from frozen settings')
    used = check_planning_trace(read(trace_path), result['planning_statistics'], scenario, config, seeds)
    forbidden = set().union(*(control_seeds(m) for m in p.get('controls', {}).values()))
    forbidden |= {s for data in p['cases'].values() for key in ('validation_seeds', 'test_seeds') for s in data[key]}
    if used & forbidden: raise ValueError('Planning query seeds overlap control or holdout environment streams')
    return result


def run_jobs(jobs, workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError('workers must be an integer from one to three')
    if workers == 1:
        for job in jobs: evaluation_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(evaluation_job, jobs): pass


def validate(study, case=None, seed=None, price_index=None, workers=1):
    p = verify(study)
    if seed not in (*SEEDS, None) or price_index not in (*PRICE_INDICES, None):
        raise ValueError('Unknown planning seed or price')
    jobs = [(str(study), 'validation', name, 'rollout_adp', s, price)
        for name in selected_cases(case) for price in (PRICE_INDICES if price_index is None else (price_index,))
        for s in (SEEDS if seed is None else (seed,))]
    run_jobs(jobs, workers)
    verify(study)


def validation_path(study, case, price, seed):
    return study/'price_selection'/case/f'price_{price}_seed_{seed}.json'


def select(study, case=None):
    p = verify(study)
    for name in selected_cases(case):
        directory, data = study/'price_selection'/name, p['cases'][name]
        if (directory/'selection.json').exists(): raise FileExistsError('Price selection already exists')
        rows, hashes = [], {}
        for price in PRICE_INDICES:
            profits = {}
            for seed in SEEDS:
                path = validation_path(study, name, price, seed)
                result = inspect_rollout(path, p, name, price, seed, data['validation_seeds'])
                profits[str(seed)] = result['mean_discounted_profit']
                for file in (path, *rollout_paths(path)): hashes[file.name] = digest(file)
            rows.append(dict(price_index=price, by_planning_seed=profits, mean_discounted_profit=float(np.mean(list(profits.values())))))
        seal(directory/'selection.json', dict(protocol_sha256=digest(study/'protocol.json'),
            validation_seeds=data['validation_seeds'], rows=rows,
            winner=max(rows, key=lambda row: row['mean_discounted_profit']), result_sha256=hashes,
            test_scenarios_accessed=False, offline_training=False))
        verify_prices(study, p, name)


def verify_prices(study, p, case):
    data, directory = p['cases'][case], study/'price_selection'/case
    selection = sealed(directory/'selection.json')
    expected_files = {file.name for price in PRICE_INDICES for seed in SEEDS
        for file in (validation_path(study, case, price, seed), *rollout_paths(validation_path(study, case, price, seed)))}
    if (selection['protocol_sha256'] != digest(study/'protocol.json')
            or selection['validation_seeds'] != data['validation_seeds']
            or selection['test_scenarios_accessed'] is not False or selection['offline_training'] is not False
            or [row['price_index'] for row in selection['rows']] != list(PRICE_INDICES)
            or set(selection['result_sha256']) != expected_files):
        raise ValueError('Price selection differs from the frozen validation protocol')
    for filename, expected in selection['result_sha256'].items():
        if digest(directory/filename) != expected: raise ValueError('Price validation evidence changed')
    for row in selection['rows']:
        price, profits = row['price_index'], {}
        for seed in SEEDS:
            result = inspect_rollout(validation_path(study, case, price, seed), p, case, price, seed, data['validation_seeds'])
            profits[str(seed)] = result['mean_discounted_profit']
        if row['by_planning_seed'] != profits or row['mean_discounted_profit'] != float(np.mean(list(profits.values()))):
            raise ValueError('Price selection must average all three planning seeds')
    if selection['winner'] != max(selection['rows'], key=lambda row: row['mean_discounted_profit']):
        raise ValueError('Selected price does not maximize validation mean')
    return selection


def prepare_evaluation(study, p):
    selections, paths = {}, {}
    for case in CASES:
        selections[case] = dict(rules=verify_rules(study, p, case), prices=verify_prices(study, p, case))
        for family in (('ppo', 'original_adp') if case == 'small' else ('ppo',)):
            for seed in SEEDS:
                directory = study/'controls'/case/family/f'seed_{seed}'
                metadata = inspect_control(directory, study/'controls'/case/family/'source_snapshot',
                    case, family, seed, _scenario_data(p['cases'][case]['scenario']))
                if metadata != p['controls'][f'{case}/{family}/seed_{seed}']: raise ValueError('Control metadata changed')
                paths[f'{case}/{family}/{seed}'] = directory/metadata['checkpoint']
    evidence = {str(path.relative_to(study)): digest(path) for folder in ('rule_selection', 'price_selection')
        for path in (study/folder).rglob('*') if path.is_file()}
    return selections, paths, evidence


def evaluate_zero(scenario, seeds, price):
    statistics = {}
    def dispatch(obs, rng):
        return coordinate_inventory_dispatch(obs, scenario, 0., sweeps=2,
                                             energy_fractions=(1.,), diagnostics=statistics)
    result = evaluate_policy(scenario, seeds, lambda obs, rng: fixed_prices(obs, scenario, price),
        counted_dispatch(dispatch, statistics), action_seed=ACTION_SEED, label='zero_continuation')
    check_result(result, seeds)
    result.update(solver_statistics=statistics, price_index=price)
    return result


def evaluation_job(job):
    study_text, phase, case, family, seed, price = job
    study = Path(study_text); p = verify(study)
    data = p['cases'][case]; scenario = _scenario_data(data['scenario'])
    seeds = data['validation_seeds' if phase == 'validation' else 'test_seeds']
    path = (validation_path(study, case, price, seed) if phase == 'validation'
        else study/'test'/case/(f'{family}_{seed}.json' if seed is not None else family+'.json'))
    if path.exists(): raise FileExistsError('Evaluation already exists: '+str(path))
    if phase == 'test':
        frozen = sealed(study/'test/frozen_inputs.json')
        if frozen['protocol_sha256'] != digest(study/'protocol.json'): raise ValueError('Final protocol differs')
    if family == 'rollout_adp':
        result = evaluate_rollout(scenario, config_from_data(data['configurations'][str(price)][str(seed)]),
            seeds, path, p, label=f'{family}/{seed}')
    elif family in ('ppo', 'original_adp'):
        key = f'{case}/{family}/{seed}'
        checkpoint = study/'controls'/case/family/f'seed_{seed}'/('checkpoint.pt' if family == 'ppo' else 'checkpoint.npz')
        if digest(checkpoint) != frozen['models'][key]: raise ValueError('Frozen checkpoint differs')
        result = (PaperPPOTrainer.load_checkpoint(checkpoint).evaluate(seeds, False, ACTION_SEED)
            if family == 'ppo' else evaluate_original(FittedValuePolicy.load_checkpoint(checkpoint), seeds, f'{family}/{seed}'))
        check_result(result, seeds)
        result.update(label=f'{family}/{seed}', training_seed=seed, checkpoint_sha256=digest(checkpoint))
        write(path, result)
    elif family in FAMILIES:
        selection = sealed(study/'rule_selection'/case/'selection.json')
        result = evaluate_rule(scenario, seeds, selection['winners'][family]['specification'], label=family)
        write(path, result)
    elif family == 'zero_continuation':
        result = evaluate_zero(scenario, seeds, price); write(path, result)
    else: raise ValueError('Unknown evaluation family')
    verify(study)
    print(f"{phase} {case}/{family}/{seed}: {result['mean_discounted_profit']:.3f}", flush=True)


def aggregate_models(items, label):
    seeds = items[0]['evaluation_seeds']
    if any(r['evaluation_seeds'] != seeds for r in items): raise ValueError('Model scene banks differ')
    seed_field = 'planning_seed' if label == 'rollout_adp' else 'training_seed'
    result = dict(label=label, evaluation_seeds=seeds, n_episodes=len(seeds),
        seed_role=seed_field, seed_sd=float(np.std([r['mean_discounted_profit'] for r in items], ddof=1)),
        episodes=[dict(seed=s, discounted_profit=float(np.mean([r['episodes'][i]['discounted_profit'] for r in items]))) for i, s in enumerate(seeds)],
        invalid_actions=sum(r['invalid_actions'] for r in items))
    fields = ('mean_discounted_profit', 'profit_p10', 'lower_tail_mean_10pct', 'loss_probability',
        'mean_served_request_share', 'mean_mobile_delivered_kwh', 'mean_charge_grid_kwh',
        'mean_final_inventory_kwh', *PHYSICAL_FIELDS)
    for key in fields:
        values = [r[key] for r in items if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    for key in ('mean_discounted_reward_terms', 'mean_policy_decision_ms'):
        result[key] = {term: float(np.mean([r[key][term] for r in items])) for term in items[0][key]}
    return result


def summarize_case(case, scenario, models, rules):
    aggregate = {name: aggregate_models(items, name) for name, items in models.items()}
    aggregate.update(rules)
    other = [name for name in aggregate if name != 'rollout_adp']
    comparisons = {name: paired_comparison(aggregate['rollout_adp'], aggregate[name]) for name in other}
    per_seed = {str(seed): {name: paired_comparison(models['rollout_adp'][i],
        models[name][i] if name in models else rules[name]) for name in other} for i, seed in enumerate(SEEDS)}
    scorecard = []
    for name, result in aggregate.items():
        scorecard.append(dict(policy=name, mean_discounted_profit=result['mean_discounted_profit'],
            seed_role=result.get('seed_role'), seed_sd=result.get('seed_sd'),
            **{key: result[key] for key in ('profit_p10', 'lower_tail_mean_10pct', 'loss_probability',
                'mean_served_request_share', 'mean_mobile_delivered_kwh', 'mean_charge_grid_kwh',
                'mean_final_inventory_kwh', *PHYSICAL_FIELDS)},
            pricing_ms=result['mean_policy_decision_ms']['pricing'], dispatch_ms=result['mean_policy_decision_ms']['dispatch'],
            mean_selected_mobile_kwh=result['mean_mobile_delivered_kwh']+result['mean_unmet_mobile_kwh'],
            mean_battery_above_reserve_kwh=result['mean_final_inventory_kwh']-scenario.reserve_kwh*scenario.n_mcv,
            invalid_actions=result['invalid_actions']))
    return dict(case=case, scorecard=scorecard, comparisons=comparisons, per_seed=per_seed,
        by_seed={name: {str(item['planning_seed'] if name == 'rollout_adp' else item['training_seed']):
            item['mean_discounted_profit'] for item in items} for name, items in models.items()},
        consistent_advantage={name: all(per_seed[str(seed)][name]['paired_bootstrap_confidence_interval_95'][0] > 0 for seed in SEEDS) for name in other},
        reward_accounting={name: result['mean_discounted_reward_terms'] for name, result in aggregate.items()},
        planning_statistics={str(r['planning_seed']): r['planning_statistics'] for r in models['rollout_adp']},
        solver_statistics={name: ({str(r['training_seed']): r.get('solver_statistics') for r in items}
            if name != 'rollout_adp' else None) for name, items in models.items()}
            | {name: r.get('solver_statistics') for name, r in rules.items()},
        invalid_actions=sum(r['invalid_actions'] for r in aggregate.values()), original_adp_supported=case == 'small')


def evaluate(study, workers=1):
    p = verify(study)
    destination = study/'test'
    if destination.exists(): raise FileExistsError('Independent test output already exists')
    selections, paths, evidence = prepare_evaluation(study, p)
    frozen = dict(protocol_sha256=digest(study/'protocol.json'), models={key: digest(path) for key, path in paths.items()},
        evidence_sha256=evidence, selected_rollout_config={case: p['cases'][case]['configurations'][str(sel['prices']['winner']['price_index'])]
            for case, sel in selections.items()})
    destination.mkdir(); seal(destination/'frozen_inputs.json', frozen)
    started, jobs = time.perf_counter(), []
    for case in CASES:
        price = selections[case]['prices']['winner']['price_index']
        for family in (('rollout_adp', 'ppo', 'original_adp') if case == 'small' else ('rollout_adp', 'ppo')):
            jobs.extend((str(study), 'test', case, family, seed, price) for seed in SEEDS)
        jobs.extend((str(study), 'test', case, family, None, price) for family in (*FAMILIES, 'zero_continuation'))
    run_jobs(jobs, workers)
    analyses, episodes, result_hashes = {}, 0, {}
    for case, data in p['cases'].items():
        scenario = _scenario_data(data['scenario']); price = selections[case]['prices']['winner']['price_index']
        models = {family: [] for family in (('rollout_adp', 'ppo', 'original_adp') if case == 'small' else ('rollout_adp', 'ppo'))}
        for family in models:
            for seed in SEEDS:
                path = destination/case/f'{family}_{seed}.json'
                result = (inspect_rollout(path, p, case, price, seed, data['test_seeds']) if family == 'rollout_adp' else read(path))
                check_result(result, data['test_seeds']); models[family].append(result); episodes += result['n_episodes']
        rules = {name: read(destination/case/(name+'.json')) for name in (*FAMILIES, 'zero_continuation')}
        for result in rules.values(): check_result(result, data['test_seeds']); episodes += result['n_episodes']
        analyses[case] = summarize_case(case, scenario, models, rules)
        analyses[case].update(selected_adp_price=scenario.price_levels[price], selected_rules=selections[case]['rules']['winners'])
    if episodes != p['final_policy_episodes']: raise RuntimeError('Final episode count differs')
    verify(study)
    if frozen != sealed(destination/'frozen_inputs.json'): raise RuntimeError('Final inputs changed')
    if any(digest(study/name) != expected for name, expected in evidence.items()): raise RuntimeError('Validation evidence changed')
    if frozen['models'] != {key: digest(path) for key, path in paths.items()}: raise RuntimeError('Control checkpoint changed')
    result_hashes = {str(path.relative_to(study)): digest(path) for case in CASES for path in (destination/case).glob('*.json')}
    write(study/'analysis.json', dict(protocol_sha256=digest(study/'protocol.json'), cases=analyses,
        policy_episodes=episodes, test_seconds=time.perf_counter()-started, offline_training=False,
        uncertainty=p['uncertainty'], limitations=p['limitations'], result_sha256=result_hashes))
    show(study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'rules', 'validate', 'select', 'evaluate', 'show'))
    parser.add_argument('--study-dir', type=Path, default=Path('outputs/general_adp_study_20261025'))
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--case', choices=tuple(CASES))
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Planning seed for validation; no offline training')
    parser.add_argument('--price-index', type=int, choices=PRICE_INDICES)
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.phase == 'init':
        if args.control_root is None: parser.error('--control-root is required for init')
        init(args.study_dir, args.control_root)
    elif args.phase == 'rules': rules(args.study_dir, args.case)
    elif args.phase == 'validate': validate(args.study_dir, args.case, args.seed, args.price_index, args.workers)
    elif args.phase == 'select': select(args.study_dir, args.case)
    elif args.phase == 'evaluate': evaluate(args.study_dir, args.workers)
    else: show(args.study_dir)


if __name__ == '__main__': main()
