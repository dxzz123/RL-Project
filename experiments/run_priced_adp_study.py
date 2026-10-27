"""Learn pricing for rollout and inventory dispatch on separate simulator banks.

Run init, rules, train, validate, select, evaluate, show. Completed individual
jobs are verified and reused; incomplete training directories are never erased.
Final evaluation requires all declared candidates and freezes their evidence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
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
import run_general_adp_study as previous
from charging_env.config import Scenario
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.benchmarks.policies import service_first_dispatch, joint_dispatch_outcomes
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from charging_env.simulator import DispatchAction, Mode
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PriceSearchConfig, fit_price_policy, PARAMETER_COUNT, pricing_observation, _sources as learner_sources
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy
from charging_env.rl.priced_rollout import PricedRolloutADPPolicy

SEEDS = (7, 19, 37)
CASES = {'small': 'configs/utility_b3.json', 'city': 'configs/synthetic_25_zones.json'}
BACKENDS = ('rollout', 'inventory')
PHASES = ('warm', *BACKENDS)
PRICE_INDICES = tuple(range(7))
ACTION_SEED = 719991
SCHEMA = 'charging_env.priced_adp_study.v1'
VALIDATION = {case: [base+2*i for i in range(32)] for case, base in
              [('small', 2_300_000_001), ('city', 2_320_000_001)]}
TEST = {case: [base+2*i for i in range(256)] for case, base in
        [('small', 2_310_000_001), ('city', 2_330_000_001)]}
EXACT = dict(price_index=4, early_shadow_price=.3, late_shadow_price=.15,
             energy_fractions=[.5, 1.], max_joint_actions=200000)
read, write, digest, seal, sealed = previous.read, previous.write, previous.digest, previous.seal, previous.sealed
normalized, check_result = previous.normalized, previous.check_result


def training_config(case, phase, seed):
    if case not in CASES or phase not in PHASES or seed not in SEEDS:
        raise ValueError('Unknown training case, phase or seed')
    warm = phase == 'warm'
    base = (2_100_000_000 if case == 'small' else 2_200_000_000)
    base += PHASES.index(phase)*1_000_000+SEEDS.index(seed)*100_000
    return PriceSearchConfig(seed=seed, generations=12 if warm else 4,
        population=24 if warm else 8, elite_count=6 if warm else 3,
        episodes_per_generation=16 if warm else 4, training_seed_base=base,
        planning_seed=seed, fixed_anchor_indices=None if warm else (), action_seed=ACTION_SEED)


def train_path(study, case, phase, seed):
    return study/'training'/case/phase/f'seed_{seed}'


def scenario_from(data):
    return previous._scenario_data(data)


def rule_grid(scenario):
    return [row for row in previous.rule_candidates(scenario) if row['family'] == 'inventory_rule']


def constant_pricer(scenario, index):
    if index not in range(len(scenario.price_levels)):
        raise ValueError('Invalid fixed price index')
    coefficients = np.zeros(PARAMETER_COUNT)
    coefficients[0] = scenario.price_levels[index]
    return LearnedPricingPolicy(scenario, coefficients)


class RuleBackend:
    """Pricing adapter with observable current-period physical accounting."""
    def __init__(self, scenario, pricer, backend, setting):
        self.scenario, self.pricer, self.backend, self.setting = scenario, pricer, backend, setting
        self.statistics = dict(decisions=0, coordinate_candidate_outcomes=0,
            current_pricing_calls=0, current_active_price_counts=[0]*len(scenario.price_levels))
        self.solver_statistics, self.pricing_decisions = {}, []
        self.reset_episode()

    def reset_episode(self, episode_key=0, action_seed=None):
        self.episode_key = int(episode_key)

    def prices(self, obs, rng=None):
        levels = self.pricer.prices(obs, None)
        active = pricing_observation(obs, self.scenario)['pricing_mask'].any(axis=1)
        width = self.scenario.battery_capacity_kwh-self.scenario.reserve_kwh
        stock = (float(np.mean(obs['energy_kwh']))-self.scenario.reserve_kwh)/width if width else 0.
        self.statistics['current_pricing_calls'] += 1
        self.statistics['current_active_price_counts'] = (np.asarray(self.statistics['current_active_price_counts'])
            +np.bincount(levels[active], minlength=len(self.scenario.price_levels))).tolist()
        self.pricing_decisions.append(dict(episode_key=self.episode_key,
            period=int(obs['period'][0]), price_levels=levels.tolist(), mobile_available=active.tolist(),
            fleet_usable_battery_fraction=stock))
        return levels

    def dispatch(self, obs, rng=None):
        self.statistics['decisions'] += 1
        if self.backend == 'warm':
            return service_first_dispatch(obs, self.scenario, .2)
        t = int(obs['period'][0]); s = self.setting
        shadow = s['early_shadow_price'] if t < self.scenario.horizon//2 else s['late_shadow_price']
        if self.backend == 'exact':
            modes, targets, amounts, outcomes = joint_dispatch_outcomes(obs, self.scenario,
                energy_fractions=(.5, 1.), max_joint_actions=EXACT['max_joint_actions'])
            effective = 0. if t+1 == self.scenario.horizon else shadow
            scores = outcomes['reward']+effective*(outcomes['next_energy_kwh'].sum(axis=1)-sum(obs['energy_kwh']))
            winner = int(np.argmax(scores))
            self.statistics['exhaustive_candidate_outcomes'] = self.statistics.get('exhaustive_candidate_outcomes', 0)+len(scores)
            return [DispatchAction(Mode(int(modes[winner, m])), int(targets[winner, m]), float(amounts[winner, m]))
                    for m in range(self.scenario.n_mcv)]
        actions = coordinate_inventory_dispatch(obs, self.scenario, shadow, sweeps=2,
            energy_fractions=(1.,), diagnostics=self.solver_statistics)
        self.statistics['coordinate_candidate_outcomes'] = self.solver_statistics['candidate_outcomes']
        return actions


def make_backend(scenario, backend, pricer, seed, setting=None, fixed_price=None):
    if backend == 'rollout':
        config = RolloutADPConfig(seed=seed, rollout_samples=8, price_index=fixed_price,
            planner_sweeps=2, energy_fractions=(1.,))
        if fixed_price is not None:
            return RolloutADPPolicy(scenario, config)
        return PricedRolloutADPPolicy(scenario, pricer, config)
    return RuleBackend(scenario, pricer, backend, setting or {})


def source_files():
    return sorted([*ROOT.glob('src/charging_env/**/*.py'),
        ROOT/'experiments/run_priced_adp_study.py', ROOT/'experiments/run_general_adp_study.py',
        *(ROOT/path for path in CASES.values())])


def historical_seeds(control_root):
    """Collect recorded streams and reconstruct both generations of PPO banks."""
    values, hashes = set(), {}
    def walk(value, context=''):
        if isinstance(value, dict):
            if {'query_seed_start', 'query_seed_count', 'query_seed_stride'} <= value.keys():
                values.update(int(value['query_seed_start'])+i*int(value['query_seed_stride'])
                    for i in range(int(value['query_seed_count'])))
            if {'training_seed_base', 'seed', 'updates', 'episodes_per_update'} <= value.keys():
                count = int(value['updates'])*int(value['episodes_per_update'])
                values.update(2*(int(value['training_seed_base'])+int(value['seed'])+i) for i in range(count))
            if {'warmstart_seed_base', 'warmstart_episodes', 'seed'} <= value.keys():
                values.update(2*(int(value['warmstart_seed_base'])+int(value['seed'])+i)
                    for i in range(int(value['warmstart_episodes'])))
            if {'eval_seed_base', 'eval_episodes'} <= value.keys():
                values.update(int(value['eval_seed_base'])+2*i for i in range(int(value['eval_episodes'])))
            for key, item in value.items(): walk(item, key if 'seed' in key.lower() else context)
        elif isinstance(value, list):
            for item in value: walk(item, context)
        elif isinstance(value, int) and not isinstance(value, bool) and 'seed' in context.lower():
            values.add(value)
    for path in sorted((Path(control_root)/'outputs').rglob('*.json')):
        walk(read(path)); hashes[str(path.relative_to(control_root))] = digest(path)
    return values, hashes


def all_environment_seeds(p):
    values = set()
    for case, data in p['cases'].items():
        for key in ('validation_seeds', 'test_seeds'): values.update(data[key])
        for phase in PHASES:
            for seed in SEEDS:
                values.update(sum(training_config(case, phase, seed).generation_seeds(), []))
    return values


@lru_cache(maxsize=32)
def planning_query_seeds(horizon, planning_seed, environment_seeds):
    values = set()
    for env_seed in environment_seeds:
        rng = np.random.default_rng(np.random.SeedSequence([planning_seed, env_seed, 0x524F4C4C]))
        for t in range(horizon):
            shape = (horizon-t-1, 8)
            draws = 2*rng.integers(0, np.iinfo(np.int64).max//2, size=shape, dtype=np.int64)
            while len(np.unique(draws)) != draws.size:
                draws = 2*rng.integers(0, np.iinfo(np.int64).max//2, size=shape, dtype=np.int64)
            values.update(map(int, draws.ravel()))
    return frozenset(values)


def training_queries(p):
    result = set()
    for case, data in p['cases'].items():
        for seed in SEEDS:
            config = training_config(case, 'rollout', seed)
            queries = planning_query_seeds(data['scenario']['horizon'], seed,
                tuple(sum(config.generation_seeds(), [])))
            if result & queries: raise ValueError('Refinement planning streams overlap across runs')
            result.update(queries)
    return result


def validation_queries(p):
    return set().union(*(planning_query_seeds(data['scenario']['horizon'], seed,
        tuple(data['validation_seeds'])) for data in p['cases'].values() for seed in SEEDS))


def validate_splits(p, historical=()):
    seen = set(historical)
    for case, data in p['cases'].items():
        banks = [data['validation_seeds'], data['test_seeds']]
        banks += [sum(training_config(case, phase, seed).generation_seeds(), []) for phase in PHASES for seed in SEEDS]
        for bank in banks:
            if len(set(bank)) != len(bank) or seen.intersection(bank):
                raise ValueError('Training, validation, test and historical seeds must be disjoint')
            seen.update(bank)


@lru_cache(maxsize=2)
def history_bank(path, expected_hash):
    if digest(path) != expected_hash: raise ValueError('Historical seed registry changed')
    with np.load(path, allow_pickle=False) as data:
        return set(map(int, data['seeds']))


def init(study, control_root):
    if study.exists(): raise FileExistsError('Use a new study directory')
    cases = {}
    for case, filename in CASES.items():
        scenario = Scenario.from_json(ROOT/filename)
        cases[case] = dict(scenario=normalized(asdict(scenario)), validation_seeds=VALIDATION[case],
            test_seeds=TEST[case], rules=rule_grid(scenario), training={phase: {
                str(seed): normalized(asdict(training_config(case, phase, seed))) for seed in SEEDS} for phase in PHASES})
    old_seeds, old_hashes = historical_seeds(control_root)
    old_hashes = {'control_root/'+name: sha for name, sha in old_hashes.items()}
    if Path(control_root).resolve() != ROOT.resolve():
        stage_seeds, stage_hashes = historical_seeds(ROOT)
        old_seeds.update(stage_seeds)
        old_hashes.update({'stage/'+name: sha for name, sha in stage_hashes.items()})
    p = dict(schema=SCHEMA, cases=cases, seeds=list(SEEDS), price_indices=list(PRICE_INDICES),
        action_seed=ACTION_SEED, final_policy_episodes=6912, exact_inventory=EXACT,
        selection='Fixed inventory: best of 49 settings. Fixed rollout: mean of all three planning seeds at all seven prices. Each backend/seed selects fixed, warm, or refined on validation, ties in that order.',
        pricing_fallback='Selected pricing includes a fixed-price fallback; a selected row need not use learned pricing. Every choice is reported.',
        inventory_backend='The validation-selected fixed-inventory shadow schedule is shared by inventory refinement, selected inventory, and rollout-price/inventory swap.',
        refinement='Each backend starts from the same seed-specific service-first warm actor and receives 128 candidate episodes, with no fixed anchors. Rollout uses eight future draws.',
        information='Prescribed rates, fleet state and observed electricity prices; no realized current requests in pricing. Learned prices also govern simulated future periods.',
        uncertainty='Paired scenario bootstrap conditional on the three fixed trained/planning seeds; excludes retraining uncertainty.',
        limitations='Simulated same-distribution cases; unequal planning computation. Bounded inventory is not exhaustive. The small exact reference retains historical settings without retuning.',
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_files()})
    validate_splits(p, old_seeds)
    reference = Path(control_root)/'outputs/general_adp_study_20261025/reference_inventory/protocol.json'
    origin = read(reference)
    if any(origin.get(k) != v for k, v in dict(price_index=4, early_shadow=.3, late_shadow=.15,
            energy_fractions=[.5, 1.], max_joint_actions=200000).items()):
        raise ValueError('Historical exhaustive inventory settings differ')
    study.mkdir(parents=True)
    for filename in p['source_sha256']:
        target = study/'source_snapshot'/filename; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/filename, target)
    write(study/'provenance/historical_json_sha256.json', old_hashes)
    write(study/'provenance/exact_inventory_origin.json', origin)
    np.savez_compressed(study/'provenance/historical_seeds.npz', seeds=np.asarray(sorted(old_seeds), dtype=np.int64))
    p['historical_seed_sha256'] = digest(study/'provenance/historical_seeds.npz')
    p['historical_seed_count'] = len(old_seeds)
    p['origin_sha256'] = {name: digest(study/'provenance'/name) for name in
        ('historical_json_sha256.json', 'exact_inventory_origin.json')}
    seal(study/'protocol.json', p)
    verify(study)


def verify(study):
    p = sealed(study/'protocol.json')
    if (p['schema'] != SCHEMA or p['seeds'] != list(SEEDS) or p['price_indices'] != list(PRICE_INDICES)
            or p['action_seed'] != ACTION_SEED or p['final_policy_episodes'] != 6912 or p['exact_inventory'] != EXACT
            or set(p['cases']) != set(CASES)):
        raise ValueError('Study protocol differs from the declared comparison')
    for case, data in p['cases'].items():
        scenario = Scenario.from_json(ROOT/CASES[case])
        if (data['scenario'] != normalized(asdict(scenario)) or data['validation_seeds'] != VALIDATION[case]
                or data['test_seeds'] != TEST[case] or data['rules'] != rule_grid(scenario)):
            raise ValueError('Case configuration or scene bank changed')
        for phase in PHASES:
            for seed in SEEDS:
                if data['training'][phase][str(seed)] != normalized(asdict(training_config(case, phase, seed))):
                    raise ValueError('Training budgets changed')
    for name, sha in p['source_sha256'].items():
        if digest(ROOT/name) != sha or digest(study/'source_snapshot'/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    for name, sha in p['origin_sha256'].items():
        if digest(study/'provenance'/name) != sha: raise ValueError('Historical provenance changed')
    if digest(study/'provenance/historical_seeds.npz') != p['historical_seed_sha256']:
        raise ValueError('Historical seed registry changed')
    validate_splits(p, history_bank(str(study/'provenance/historical_seeds.npz'), p['historical_seed_sha256']))
    if (study/'provenance/replay.json').exists():
        replay = sealed(study/'provenance/replay.json')
        if replay['source_protocol_sha256'] != digest(study/'protocol.json') or replay['fresh_independent_study'] is not False:
            raise ValueError('Replay provenance differs from the copied protocol')
    return p


def replay_init(study, source):
    """Faithfully reproduce a frozen study; deliberately reuse its scene banks."""
    if study.exists(): raise FileExistsError('Use a new replay directory')
    verify(source)  # Includes exact current source/config and original init evidence.
    study.mkdir(parents=True)
    shutil.copytree(source/'source_snapshot', study/'source_snapshot')
    (study/'provenance').mkdir()
    for name in ('protocol.json', 'protocol.sha256'):
        shutil.copy2(source/name, study/name)
    for name in ('historical_seeds.npz', 'historical_json_sha256.json', 'exact_inventory_origin.json'):
        shutil.copy2(source/'provenance'/name, study/'provenance'/name)
    seal(study/'provenance/replay.json', dict(source_study=str(source.resolve()),
        source_protocol_sha256=digest(source/'protocol.json'), fresh_independent_study=False,
        purpose='Faithful rerun of the same frozen algorithm and scenario banks; no prior training, selection or test results copied.'))
    verify(study)


def inspect_training(study, p, case, phase, seed):
    directory = train_path(study, case, phase, seed)
    summary, protocol, history = (read(directory/name) for name in ('summary.json', 'protocol.json', 'training_history.json'))
    config = training_config(case, phase, seed); scenario = scenario_from(p['cases'][case]['scenario'])
    model = LearnedPricingPolicy.load_checkpoint(directory/'checkpoint.npz')
    if sealed(directory.with_suffix('.binding.json')) != training_binding(study, p, case, phase, seed):
        raise ValueError('Training backend binding differs from declared settings')
    expected = config.generations*config.population
    if (summary['config'] != normalized(asdict(config)) or protocol['config'] != summary['config']
            or model.scenario != scenario or summary['generation_seeds'] != config.generation_seeds()
            or protocol['generation_seeds'] != config.generation_seeds() or len(history) != config.generations
            or summary['candidate_evaluations'] != expected
            or summary['environment_episodes'] != expected*config.episodes_per_generation
            or summary['environment_periods'] != expected*config.episodes_per_generation*scenario.horizon
            or summary['invalid_actions'] or summary['all_returns_finite'] is not True
            or summary['checkpoint_reload_matches'] is not True
            or summary['policy_sha256'] != model.provenance()['policy_sha256']):
        raise ValueError('Training model, budget or scenario evidence differs')
    for filename, key in [('checkpoint.npz', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                          ('training_history.json', 'training_history_sha256')]:
        if digest(directory/filename) != summary[key]: raise ValueError('Training evidence changed')
    expected_sources = learner_sources()
    if (summary['source_sha256'] != expected_sources or protocol['source_sha256'] != expected_sources
            or protocol['scenario'] != normalized(asdict(scenario))
            or any(p['source_sha256'].get(name) != sha for name, sha in expected_sources.items())):
        raise ValueError('Training source differs from frozen study')
    factory = summary['backend_factory']
    if (factory != protocol['backend_factory']
            or factory['source_sha256'] != p['source_sha256']['experiments/run_priced_adp_study.py']):
        raise ValueError('Training backend factory differs from frozen runner')
    counters = {}
    for g, row in enumerate(history):
        if (row['generation'] != g or row['planning_seed'] != seed
                or row['environment_seeds'] != config.generation_seeds()[g] or len(row['candidates']) != config.population):
            raise ValueError('Training generation bank or population differs')
        scores = []
        for candidate in row['candidates']:
            profits = np.asarray(candidate['episode_discounted_profits'])
            if (profits.shape != (config.episodes_per_generation,) or not np.isfinite(profits).all()
                    or candidate['invalid_actions'] or candidate['mean_discounted_profit'] != float(profits.mean())):
                raise ValueError('Training candidate accounting differs')
            scores.append(float(profits.mean()))
            for key, value in candidate['backend_counters'].items(): counters[key] = counters.get(key, 0)+value
            if phase != 'warm' and candidate['kind'].startswith('fixed_price_'):
                raise ValueError('Refinement must not add fixed-price anchors')
        winner = int(np.argmax(scores))
        if row['winner_index'] != winner or row['winner_score'] != scores[winner]:
            raise ValueError('Training winner is not the declared first argmax')
    if counters != summary['backend_counters'] or not np.array_equal(
            model.coefficients, history[-1]['candidates'][history[-1]['winner_index']]['coefficients']):
        raise ValueError('Training checkpoint or counters differ from complete history')
    if phase != 'warm':
        warm = LearnedPricingPolicy.load_checkpoint(train_path(study, case, 'warm', seed)/'checkpoint.npz')
        if protocol['initial_policy_sha256'] != warm.provenance()['policy_sha256']:
            raise ValueError('Backend refinements must start from the same warm pricing policy')
    if counters['decisions'] != summary['environment_periods']:
        raise ValueError('Training backend decision count differs from episode budget')
    if phase == 'rollout':
        exogenous = summary['environment_episodes']*8*sum(range(scenario.horizon))
        if (counters['exogenous_draws'] != exogenous or not exogenous <= counters['future_model_queries'] <= 5*exogenous
                or not counters['decisions'] <= counters['candidate_plans'] <= 5*counters['decisions']):
            raise ValueError('Training rollout query budget differs')
        own = planning_query_seeds(scenario.horizon, seed, tuple(sum(config.generation_seeds(), [])))
        forbidden = history_bank(str(study/'provenance/historical_seeds.npz'), p['historical_seed_sha256']) | all_environment_seeds(p)
        if own & forbidden: raise ValueError('Training planning queries overlap historical or environment streams')
    return model


def training_binding(study, p, case, phase, seed):
    setting = verify_rules(study, p, case)['winner']['specification']
    return dict(protocol_sha256=digest(study/'protocol.json'), case=case, phase=phase, seed=seed,
        backend=phase, config=normalized(asdict(training_config(case, phase, seed))),
        inventory_setting=setting if phase == 'inventory' else None,
        rollout_config=normalized(asdict(RolloutADPConfig(seed=seed, rollout_samples=8))) if phase == 'rollout' else None,
        initial_checkpoint_sha256=None if phase == 'warm' else digest(train_path(study, case, 'warm', seed)/'checkpoint.npz'))


def validation_path(study, case, backend, kind, seed=None, price=None):
    suffix = f'price_{price}_seed_{seed}' if kind == 'fixed' else f'{kind}_seed_{seed}'
    return study/'validation'/case/backend/(suffix+'.json')


def evaluate_saved(study, p, case, seeds, path, specification):
    if path.exists(): return inspect_evaluation(study, p, case, seeds, path, specification)
    scenario = scenario_from(p['cases'][case]['scenario'])
    spec = specification
    if spec['pricing'] == 'fixed':
        pricer = constant_pricer(scenario, spec['price_index'])
    else:
        pricer = inspect_training(study, p, case, spec['phase'], spec['seed'])
    backend = make_backend(scenario, spec['backend'], pricer, spec.get('seed', 7),
        spec.get('setting'), fixed_price=spec.get('price_index') if spec['pricing'] == 'fixed' else None)
    pricing_records = getattr(backend, 'pricing_decisions', [])
    def pricing(obs, rng):
        levels = backend.prices(obs, rng)
        if not hasattr(backend, 'pricing_decisions'):
            active = pricing_observation(obs, scenario)['pricing_mask'].any(axis=1)
            width = scenario.battery_capacity_kwh-scenario.reserve_kwh
            pricing_records.append(dict(episode_key=backend.episode_key, period=int(obs['period'][0]),
                price_levels=levels.tolist(), mobile_available=active.tolist(),
                fleet_usable_battery_fraction=(float(np.mean(obs['energy_kwh']))-scenario.reserve_kwh)/width if width else 0.))
        return levels
    result = evaluate_policy(scenario, seeds, pricing, backend.dispatch,
        action_seed=ACTION_SEED, label=spec['label'], on_episode_start=backend.reset_episode)
    check_result(result, seeds)
    result.update(specification=spec, backend_statistics=backend.statistics)
    if spec['pricing'] != 'fixed':
        result['pricing_checkpoint_sha256'] = digest(train_path(study, case, spec['phase'], spec['seed'])/'checkpoint.npz')
    evidence = {}
    if spec['backend'] == 'rollout':
        trace_path = path.with_suffix('.planning.json'); write(trace_path, backend.decisions)
        evidence[trace_path.name] = digest(trace_path)
        result['rollout_config'] = normalized(asdict(backend.config))
        provenance_path = path.with_suffix('.provenance.json'); write(provenance_path, backend.provenance())
        evidence[provenance_path.name] = digest(provenance_path)
    price_path = path.with_suffix('.pricing.json'); write(price_path, pricing_records)
    evidence[price_path.name] = digest(price_path)
    result['active_pricing'] = summarize_pricing(pricing_records, scenario)
    result['evidence_sha256'] = evidence
    seal(path, result)
    inspect_evaluation(study, p, case, seeds, path, spec)
    print(f"{case} {spec['label']}: {result['mean_discounted_profit']:.3f}", flush=True)
    return result


def inspect_evaluation(study, p, case, seeds, path, specification):
    result = sealed(path); check_result(result, seeds)
    if result['specification'] != specification or result['action_seed'] != ACTION_SEED:
        raise ValueError('Evaluation specification or action seed changed')
    for name, sha in result['evidence_sha256'].items():
        if digest(path.parent/name) != sha: raise ValueError('Evaluation trace changed')
    spec = specification
    scenario = scenario_from(p['cases'][case]['scenario'])
    prices = read(path.with_suffix('.pricing.json'))
    if (len(prices) != len(seeds)*scenario.horizon or [(r['episode_key'], r['period']) for r in prices]
            != [(seed, t) for seed in seeds for t in range(scenario.horizon)]
            or result['active_pricing'] != summarize_pricing(prices, scenario)):
        raise ValueError('Current pricing trace or summary differs')
    if spec['pricing'] != 'fixed' and result['pricing_checkpoint_sha256'] != digest(
            train_path(study, case, spec['phase'], spec['seed'])/'checkpoint.npz'):
        raise ValueError('Evaluated pricing checkpoint changed')
    if spec['backend'] == 'rollout':
        config = previous.config_from_data(result['rollout_config'])
        if config != RolloutADPConfig(seed=spec['seed'], rollout_samples=8,
                price_index=spec.get('price_index') if spec['pricing'] == 'fixed' else None,
                planner_sweeps=2, energy_fractions=(1.,)):
            raise ValueError('Rollout budget changed')
        trace = read(path.with_suffix('.planning.json'))
        pricer = (constant_pricer(scenario, spec['price_index']) if spec['pricing'] == 'fixed'
            else LearnedPricingPolicy.load_checkpoint(train_path(study, case, spec['phase'], spec['seed'])/'checkpoint.npz'))
        expected_backend = make_backend(scenario, 'rollout', pricer, spec['seed'], spec.get('setting'),
            fixed_price=spec.get('price_index') if spec['pricing'] == 'fixed' else None)
        expected_provenance = normalized(expected_backend.provenance())
        expected_provenance['statistics'] = result['backend_statistics']
        if read(path.with_suffix('.provenance.json')) != expected_provenance:
            raise ValueError('Rollout provenance differs from its source, actor or configuration')
        used = previous.check_planning_trace(trace, result['backend_statistics'], scenario, config, seeds)
        prohibited = history_bank(str(study/'provenance/historical_seeds.npz'), p['historical_seed_sha256']) | all_environment_seeds(p) | training_queries(p)
        if list(seeds) == p['cases'][case]['test_seeds']: prohibited |= validation_queries(p)
        if used & prohibited: raise ValueError('Future planning seeds overlap an environment or historical stream')
        if spec['pricing'] != 'fixed':
            if (result['backend_statistics']['future_pricing_calls'] != result['backend_statistics']['future_model_queries']
                    or result['backend_statistics']['current_pricing_calls'] != len(seeds)*scenario.horizon
                    or result['backend_statistics']['current_active_price_counts'] != result['active_pricing']['counts']):
                raise ValueError('Every future model query must use the pricing actor')
            if any(row['future_pricing_calls'] != row['future_model_queries'] for row in trace):
                raise ValueError('Missing future pricing decisions')
    return result


def summarize_pricing(records, scenario):
    counts = np.zeros((scenario.horizon, len(scenario.price_levels)), dtype=np.int64)
    for row in records:
        levels, active = np.asarray(row['price_levels']), np.asarray(row['mobile_available'])
        if (levels.shape != (scenario.n_zones,) or levels.dtype.kind not in 'iu'
                or active.shape != levels.shape or active.dtype.kind != 'b'
                or (levels < 0).any() or (levels >= len(scenario.price_levels)).any()
                or (levels[~active] != 0).any()
                or not 0 <= row['period'] < scenario.horizon
                or not -1e-8 <= row['fleet_usable_battery_fraction'] <= 1+1e-8):
            raise ValueError('Malformed current pricing trace')
        counts[row['period']] += np.bincount(levels[active], minlength=len(scenario.price_levels))
    total = counts.sum(axis=0); levels = np.asarray(scenario.price_levels)
    return dict(counts=total.tolist(), by_period=counts.tolist(),
        mean_active_price=float(total@levels/total.sum()) if total.sum() else None)


def rule_spec(row):
    return dict(label=row['id'], backend='inventory', pricing='fixed', price_index=row['price_index'], setting=row)


def verify_rules(study, p, case, require_selection=True):
    rows = []
    for setting in p['cases'][case]['rules']:
        path = study/'rules'/case/(setting['id']+'.json')
        result = inspect_evaluation(study, p, case, p['cases'][case]['validation_seeds'], path, rule_spec(setting))
        rows.append(dict(specification=setting, mean_discounted_profit=result['mean_discounted_profit']))
    winner = max(rows, key=lambda r: r['mean_discounted_profit'])
    expected = dict(rows=rows, winner=winner, selection='First argmax over all 49 declared validation settings', test_accessed=False)
    if require_selection and sealed(study/'rules'/case/'selection.json') != expected:
        raise ValueError('Inventory selection differs from complete validation evidence')
    return expected


def job(payload):
    study_s, phase, case, seed, item = payload
    study = Path(study_s); p = verify(study); data = p['cases'][case]
    if phase == 'rule':
        return evaluate_saved(study, p, case, data['validation_seeds'],
            study/'rules'/case/(item['id']+'.json'), rule_spec(item))['mean_discounted_profit']
    setting = verify_rules(study, p, case)['winner']['specification']
    if phase == 'train':
        phases = PHASES if item == 'all' else (item,)
        scenario = scenario_from(data['scenario'])
        for train_phase in phases:
            directory = train_path(study, case, train_phase, seed)
            if directory.exists():
                inspect_training(study, p, case, train_phase, seed); continue
            initial = None if train_phase == 'warm' else inspect_training(study, p, case, 'warm', seed)
            binding_path = directory.with_suffix('.binding.json')
            binding = training_binding(study, p, case, train_phase, seed)
            if binding_path.exists() and sealed(binding_path) != binding:
                raise ValueError('Existing training backend binding changed')
            if not binding_path.exists(): seal(binding_path, binding)
            def factory(pricer, planning_seed):
                return make_backend(scenario, train_phase, pricer, planning_seed, setting)
            fit_price_policy(scenario, training_config(case, train_phase, seed), factory, directory, initial)
            inspect_training(study, p, case, train_phase, seed)
        return None
    if phase == 'validation':
        backend, kind, price = item
        spec = dict(label=f'{backend}/{kind}/{seed}/{price}', backend=backend, pricing=kind,
            seed=seed, setting=setting)
        if kind == 'fixed': spec['price_index'] = price
        else: spec['phase'] = 'warm' if kind == 'warm' else backend
        return evaluate_saved(study, p, case, data['validation_seeds'],
            validation_path(study, case, backend, kind, seed, price), spec)['mean_discounted_profit']
    if phase == 'test':
        frozen = sealed(study/'test/frozen_inputs.json')
        if frozen['protocol_sha256'] != digest(study/'protocol.json'):
            raise ValueError('Final protocol changed')
        name, spec = item
        if frozen['final_jobs'][case][name] != spec: raise ValueError('Final policy selection changed')
        return evaluate_saved(study, p, case, data['test_seeds'], study/'test'/case/(name+'.json'), spec)['mean_discounted_profit']
    raise ValueError('Unknown worker phase')


def run_jobs(jobs, workers):
    if isinstance(workers, bool) or workers not in (1, 2, 3): raise ValueError('workers must be 1, 2, or 3')
    if workers == 1:
        for item in jobs: job(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(job, jobs): pass


def cases_for(case):
    if case is not None and case not in CASES: raise ValueError('Unknown case')
    return list(CASES) if case is None else [case]


def rules(study, case=None, workers=1):
    p = verify(study)
    jobs = [(str(study), 'rule', name, None, row) for name in cases_for(case) for row in p['cases'][name]['rules']]
    run_jobs(jobs, workers)
    for name in cases_for(case):
        selection = verify_rules(study, p, name, False)
        path = study/'rules'/name/'selection.json'
        if path.exists() and sealed(path) != selection: raise ValueError('Existing rule selection changed')
        if not path.exists(): seal(path, selection)


def train(study, case=None, seed=None, phase='all', workers=1):
    verify(study)
    if seed not in (*SEEDS, None) or phase not in (*PHASES, 'all'): raise ValueError('Unknown training job')
    run_jobs([(str(study), 'train', name, s, phase) for name in cases_for(case)
        for s in (SEEDS if seed is None else (seed,))], workers)


def validate(study, case=None, seed=None, workers=1):
    p = verify(study)
    if seed not in (*SEEDS, None): raise ValueError('Unknown seed')
    jobs = []
    for name in cases_for(case):
        for s in (SEEDS if seed is None else (seed,)):
            for phase in PHASES: inspect_training(study, p, name, phase, s)
            jobs += [(str(study), 'validation', name, s, ('rollout', 'fixed', price)) for price in PRICE_INDICES]
            jobs += [(str(study), 'validation', name, s, (backend, kind, None))
                for backend in BACKENDS for kind in ('warm', 'refined')]
    run_jobs(jobs, workers)


def selection_from_scores(fixed_prices, learned_scores, inventory):
    """Separate global fixed-price choice from each seed's declared fallback."""
    rows = [dict(price_index=price, by_seed=fixed_prices[str(price)],
        mean_discounted_profit=float(np.mean([fixed_prices[str(price)][str(seed)] for seed in SEEDS]))) for price in PRICE_INDICES]
    fixed = max(rows, key=lambda row: row['mean_discounted_profit'])
    selected = {}
    for backend in BACKENDS:
        selected[backend] = {}
        for seed in SEEDS:
            fallback = fixed['by_seed'][str(seed)] if backend == 'rollout' else inventory['mean_discounted_profit']
            scores = dict(fixed=fallback, warm=learned_scores[backend][str(seed)]['warm'],
                          refined=learned_scores[backend][str(seed)]['refined'])
            selected[backend][str(seed)] = dict(candidate=max(scores, key=scores.get), scores=scores)
    return dict(fixed_rollout_rows=rows, fixed_rollout=fixed, fixed_inventory=inventory,
        selected=selected, test_accessed=False, fallback_order=['fixed', 'warm', 'refined'])


def compute_selection(study, p, case):
    data = p['cases'][case]; setting = verify_rules(study, p, case)['winner']
    prices, learned = {}, {b: {} for b in BACKENDS}
    for price in PRICE_INDICES:
        prices[str(price)] = {}
        for seed in SEEDS:
            spec = dict(label=f'rollout/fixed/{seed}/{price}', backend='rollout', pricing='fixed',
                seed=seed, setting=setting['specification'], price_index=price)
            result = inspect_evaluation(study, p, case, data['validation_seeds'],
                validation_path(study, case, 'rollout', 'fixed', seed, price), spec)
            prices[str(price)][str(seed)] = result['mean_discounted_profit']
    for backend in BACKENDS:
        for seed in SEEDS:
            learned[backend][str(seed)] = {}
            for kind in ('warm', 'refined'):
                spec = dict(label=f'{backend}/{kind}/{seed}/None', backend=backend, pricing=kind,
                    seed=seed, setting=setting['specification'], phase='warm' if kind == 'warm' else backend)
                result = inspect_evaluation(study, p, case, data['validation_seeds'],
                    validation_path(study, case, backend, kind, seed), spec)
                learned[backend][str(seed)][kind] = result['mean_discounted_profit']
    return selection_from_scores(prices, learned, setting)


def select(study, case=None):
    p = verify(study)
    for name in cases_for(case):
        selection = compute_selection(study, p, name); path = study/'selection'/f'{name}.json'
        if path.exists() and sealed(path) != selection: raise ValueError('Existing pricing selection changed')
        if not path.exists(): seal(path, selection)


def final_specs(case, selection):
    specs = {}; setting = selection['fixed_inventory']['specification']
    def fixed(backend, price, seed=None):
        result = dict(backend=backend, pricing='fixed', price_index=price, setting=setting)
        if seed is not None: result['seed'] = seed
        return result
    for seed in SEEDS:
        specs[f'fixed_rollout_{seed}'] = fixed('rollout', selection['fixed_rollout']['price_index'], seed)
        for backend in BACKENDS:
            candidate = selection['selected'][backend][str(seed)]['candidate']
            if candidate == 'fixed':
                price = selection['fixed_rollout']['price_index'] if backend == 'rollout' else setting['price_index']
                spec = fixed(backend, price, seed)
            else:
                spec = dict(backend=backend, pricing=candidate, phase='warm' if candidate == 'warm' else backend,
                            seed=seed, setting=setting)
            specs[f'selected_{backend}_{seed}'] = spec
        specs[f'rollout_price_inventory_{seed}'] = dict(specs[f'selected_rollout_{seed}'], backend='inventory')
    specs['fixed_inventory'] = fixed('inventory', setting['price_index'])
    if case == 'small':
        specs['exact_inventory'] = dict(backend='exact', pricing='fixed', price_index=4, setting=EXACT)
    return {name: dict(spec, label=name) for name, spec in specs.items()}


def prepare_final(study, p):
    selections, jobs = {}, {}
    for case in CASES:
        for phase in PHASES:
            for seed in SEEDS: inspect_training(study, p, case, phase, seed)
        actual = compute_selection(study, p, case)
        if sealed(study/'selection'/f'{case}.json') != actual: raise ValueError('Pricing selection changed')
        selections[case] = actual; jobs[case] = final_specs(case, actual)
    evidence = {str(path.relative_to(study)): digest(path) for folder in ('training', 'rules', 'validation', 'selection')
        for path in (study/folder).rglob('*') if path.is_file() and path.suffix in ('.json', '.npz', '.sha256')}
    return dict(protocol_sha256=digest(study/'protocol.json'), selections=selections,
        final_jobs=jobs, evidence_sha256=evidence)


def aggregate(items, label):
    adapted = [dict(item, training_seed=item['specification']['seed']) for item in items]
    result = previous.aggregate_models(adapted, label)
    result['seed_role'] = dict(fixed_rollout='planning_seed', selected_rollout='pricing_training_and_planning_seed',
        selected_inventory='pricing_training_seed', rollout_price_inventory='source_pricing_training_seed')[label]
    result['mean_mobile_price'] = float(np.mean([item['mean_mobile_price'] for item in items]))
    counts = np.sum([item['active_pricing']['counts'] for item in items], axis=0)
    weighted = sum(item['active_pricing']['mean_active_price']*sum(item['active_pricing']['counts'])
        for item in items if item['active_pricing']['mean_active_price'] is not None)
    result['active_pricing'] = dict(counts=counts.tolist(),
        by_period=np.sum([item['active_pricing']['by_period'] for item in items], axis=0).tolist(),
        mean_active_price=float(weighted/counts.sum()) if counts.sum() else None)
    return result


def analysis_case(results, selection):
    groups = {name: aggregate([results[f'{name}_{seed}'] for seed in SEEDS], name)
        for name in ('fixed_rollout', 'selected_rollout', 'selected_inventory', 'rollout_price_inventory')}
    groups.update({name: results[name] for name in ('fixed_inventory', 'exact_inventory') if name in results})
    comparisons = {name: paired_comparison(groups['selected_rollout'], result)
        for name, result in groups.items() if name != 'selected_rollout'}
    per_seed = {str(seed): {name: paired_comparison(results[f'selected_rollout_{seed}'],
        results[f'{name}_{seed}'] if f'{name}_{seed}' in results else results[name])
        for name in groups if name != 'selected_rollout'} for seed in SEEDS}
    comparisons['inventory_pricing_gain'] = paired_comparison(groups['selected_inventory'], groups['fixed_inventory'])
    return dict(selection=selection, groups=groups, comparisons=comparisons, per_seed=per_seed,
        by_seed={name: {str(seed): results[f'{name}_{seed}']['mean_discounted_profit'] for seed in SEEDS}
            for name in ('fixed_rollout', 'selected_rollout', 'selected_inventory', 'rollout_price_inventory')},
        backend_statistics={name: row['backend_statistics'] for name, row in results.items()},
        consistent_advantage={name: all(per_seed[str(seed)][name]['paired_bootstrap_confidence_interval_95'][0] > 0
            for seed in SEEDS) for name in groups if name != 'selected_rollout'})


def evaluate(study, workers=1):
    p = verify(study)
    frozen = prepare_final(study, p)  # No final directory before every prerequisite passes.
    destination = study/'test'
    if destination.exists():
        if sealed(destination/'frozen_inputs.json') != frozen: raise ValueError('Frozen final inputs changed')
    else:
        destination.mkdir(); seal(destination/'frozen_inputs.json', frozen)
    if (study/'analysis.json').exists():
        existing = sealed(study/'analysis.json')
        if (existing['protocol_sha256'] != digest(study/'protocol.json')
                or any(digest(study/name) != sha for name, sha in existing['result_sha256'].items())):
            raise ValueError('Completed final results changed')
        show(study)
        return
    started = time.perf_counter()
    jobs = [(str(study), 'test', case, None, (name, spec)) for case, specs in frozen['final_jobs'].items()
        for name, spec in specs.items()]
    run_jobs(jobs, workers)
    analyses, count = {}, 0
    for case, specs in frozen['final_jobs'].items():
        results = {name: inspect_evaluation(study, p, case, p['cases'][case]['test_seeds'],
            destination/case/(name+'.json'), spec) for name, spec in specs.items()}
        count += sum(row['n_episodes'] for row in results.values())
        analyses[case] = analysis_case(results, frozen['selections'][case])
    if count != p['final_policy_episodes']: raise ValueError('Final policy episode count differs')
    verify(study)
    if any(digest(study/name) != sha for name, sha in frozen['evidence_sha256'].items()):
        raise ValueError('Frozen training or validation evidence changed during final evaluation')
    if sealed(destination/'frozen_inputs.json') != frozen: raise ValueError('Frozen final inputs changed')
    seal(study/'analysis.json', dict(schema=SCHEMA, cases=analyses, policy_episodes=count,
        run_kind='faithful_replay' if (study/'provenance/replay.json').exists() else 'fresh_study',
        protocol_sha256=digest(study/'protocol.json'), elapsed_this_invocation_seconds=time.perf_counter()-started,
        result_sha256={str(path.relative_to(study)): digest(path) for case in CASES for path in (destination/case).glob('*') if path.is_file()},
        uncertainty=p['uncertainty'], fallback=p['pricing_fallback']))
    show(study)


def show(study):
    analysis = sealed(study/'analysis.json')
    for case, data in analysis['cases'].items():
        print('\n'+case)
        for name, row in data['groups'].items(): print(f"  {name:28s} {row['mean_discounted_profit']:10.3f}")
        for backend in BACKENDS:
            print(f"  {backend} choices: "+', '.join(f"{seed}: {data['selection']['selected'][backend][str(seed)]['candidate']}" for seed in SEEDS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('init', 'rules', 'train', 'validate', 'select', 'evaluate', 'show'), required=True)
    parser.add_argument('--study-dir', type=Path, default=ROOT/'outputs/priced_adp_study_20261025')
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--replay-of', type=Path, help='Reproduce a frozen identical-source study, explicitly reusing its banks')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--training-phase', choices=(*PHASES, 'all'), default='all')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(argv); study = args.study_dir.resolve()
    if args.phase == 'init':
        if args.replay_of is not None:
            if args.control_root is not None: parser.error('Use --replay-of or --control-root, not both')
            replay_init(study, args.replay_of.resolve())
        else:
            if args.control_root is None: parser.error('--control-root is required for historical stream auditing')
            init(study, args.control_root.resolve())
    elif args.phase == 'rules': rules(study, args.case, args.workers)
    elif args.phase == 'train': train(study, args.case, args.seed, args.training_phase, args.workers)
    elif args.phase == 'validate': validate(study, args.case, args.seed, args.workers)
    elif args.phase == 'select': select(study, args.case)
    elif args.phase == 'evaluate': evaluate(study, args.workers)
    else: show(study)


if __name__ == '__main__': main()
