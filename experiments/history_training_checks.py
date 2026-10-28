"""Check saved training internals without running any policy or test scenario.

The study runner owns completion seals and the frozen source snapshot. These
checks bind the learner's own records to that completion binding, reconstruct
CEM arithmetic, and check PPO checkpoint/monitor/demonstration consistency.
CEM histories contain episode profits but no component ledgers; independent
sampled episode replays remain necessary to audit those economic calculations.
"""
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from charging_env.rl.adaptive_pricing import (LearnedPricingPolicy, PriceSearchConfig,
    FEATURE_SCHEMA, FEATURE_NAMES, PARAMETER_COUNT, _COUNTERS, _sources)


REVENUE = ('mobile_service_revenue', 'fixed_service_revenue', 'discharge_revenue', 'terminal_inventory_value')
COST = ('procurement_cost', 'travel_cost', 'operating_cost', 'unmet_demand_cost')
MODES = ('WAIT', 'REPOSITION', 'SERVE', 'RECHARGE', 'DISCHARGE')


def _require(condition, message):
    if not condition: raise ValueError(message)


def _read(path):
    def reject(value): raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _close(actual, expected, name, tolerance=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    _require(a.shape == b.shape and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu'
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a, b, rtol=0., atol=tolerance), name+' differs')


def _integer(value, name, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        name+' must be an integer >= '+str(minimum))


def _finite(value):
    if isinstance(value, dict):
        for child in value.values(): _finite(child)
    elif isinstance(value, list):
        for child in value: _finite(child)
    elif isinstance(value, (float, int)):
        _require(np.isfinite(value), 'Nonfinite saved metric')


def _episode_ledgers(episodes, scenario, expected_seeds):
    _require([e['seed'] for e in episodes] == list(expected_seeds), 'Episode bank or order differs')
    for episode in episodes:
        _finite(episode)
        _require(episode['periods'] == scenario.horizon and episode['invalid_actions'] == 0,
            'Incomplete episode or invalid action')
        for name, profit in (('reward_terms', 'profit'), ('discounted_reward_terms', 'discounted_profit')):
            ledger = episode[name]
            _require(set(ledger) == set(REVENUE+COST), 'Reward ledger fields differ')
            accounted = sum(ledger[k] for k in REVENUE)-sum(ledger[k] for k in COST)
            _close(episode[profit], accounted, profit+' reward ledger', 1e-7)
        _require(set(episode['action_counts']) == set(MODES)
            and sum(episode['action_counts'].values()) == scenario.horizon*scenario.n_mcv,
            'Fleet action count differs')
        for value in episode['action_counts'].values(): _integer(value, 'Action count')
        expected_energy = (episode['initial_inventory_kwh']-episode['travel_kwh']
            -episode['mobile_delivered_kwh']/scenario.mobile_service_efficiency
            +scenario.charge_efficiency*episode['charge_grid_kwh']
            -episode['export_grid_kwh']/scenario.discharge_efficiency)
        _close(episode['final_inventory_kwh'], expected_energy, 'Battery ledger', 1e-7)
        _close(episode['reward_terms']['terminal_inventory_value'],
            scenario.terminal_value_per_stored_kwh*episode['final_inventory_kwh'], 'Terminal reward', 1e-7)
        _close(episode['discounted_reward_terms']['terminal_inventory_value'],
            scenario.discount_per_period**(scenario.horizon-1)*episode['reward_terms']['terminal_inventory_value'],
            'Discounted terminal reward', 1e-7)


def _evaluation(result, scenario, config):
    _finite(result)
    seeds = [config.eval_seed_base+2*i for i in range(config.eval_episodes)]
    _require(result['evaluation_seeds'] == seeds and result['n_episodes'] == len(seeds)
        and result['invalid_actions'] == 0 and result['action_seed'] == config.eval_action_seed
        and result['deterministic'] is (config.eval_policy == 'deterministic'), 'PPO monitor metadata differs')
    _episode_ledgers(result['episodes'], scenario, seeds)
    _close(result['mean_discounted_profit'], np.mean([e['discounted_profit'] for e in result['episodes']]),
        'PPO monitor profit mean')
    for ledger in ('reward_terms', 'discounted_reward_terms'):
        for term in REVENUE+COST:
            _close(result['mean_'+ledger][term], np.mean([e[ledger][term] for e in result['episodes']]),
                'PPO aggregate reward ledger')


def _check_cem(directory, scenario, phase, config, factory_sha, warm_checkpoint):
    _require(isinstance(config, PriceSearchConfig), 'Pricing phase requires PriceSearchConfig')
    config.validate(scenario)
    summary, protocol, history = (_read(directory/name) for name in
        ('summary.json', 'protocol.json', 'training_history.json'))
    _finite(summary); _finite(history)
    expected_config = _normalize(asdict(config))
    _require(protocol['schema'] == 'charging_env.price_search.v1'
        and summary['schema'] == 'charging_env.price_search_summary.v1'
        and protocol['scenario'] == _normalize(asdict(scenario))
        and protocol['config'] == summary['config'] == expected_config, 'CEM schema, configuration or scenario differs')
    source = _sources()
    _require(protocol['source_sha256'] == summary['source_sha256'] == source, 'CEM source identity differs')
    _require(protocol['backend_factory'] == summary['backend_factory']
        and protocol['backend_factory']['source_sha256'] == factory_sha, 'CEM backend factory differs')
    plan_seed = config.seed if config.planning_seed is None else config.planning_seed
    bank = config.generation_seeds()
    for item in (protocol, summary):
        _require(item['feature_schema'] == FEATURE_SCHEMA and item['parameter_count'] == PARAMETER_COUNT
            and item['generation_seeds'] == bank and item['planning_seed'] == plan_seed, 'CEM feature or seed metadata differs')
    _require(protocol['feature_names'] == list(FEATURE_NAMES), 'CEM feature names differ')
    expected_candidates = config.generations*config.population
    expected_episodes = expected_candidates*config.episodes_per_generation
    _require(protocol['expected_candidate_evaluations'] == expected_candidates
        and protocol['expected_environment_episodes'] == expected_episodes
        and summary['candidate_evaluations'] == expected_candidates
        and summary['environment_episodes'] == expected_episodes
        and summary['environment_periods'] == expected_episodes*scenario.horizon
        and summary['generations'] == config.generations and summary['seed'] == config.seed
        and summary['invalid_actions'] == 0 and summary['all_returns_finite'] is True
        and summary['checkpoint_reload_matches'] is True, 'CEM budgets or completion claims differ')
    for filename, key in (('checkpoint.npz', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                          ('training_history.json', 'training_history_sha256')):
        _require(_hash(directory/filename) == summary[key], 'CEM internal file hash differs: '+filename)
    levels = np.asarray(scenario.price_levels)
    if phase == 'warm':
        _require(warm_checkpoint is None, 'Initial pricing phase must not import a warm checkpoint')
        initial = np.zeros(PARAMETER_COUNT); initial[0] = levels[len(levels)//2]
        actor = LearnedPricingPolicy(scenario, initial)
    else:
        _require(warm_checkpoint is not None, 'Pricing refinement requires its warm checkpoint')
        actor = LearnedPricingPolicy.load_checkpoint(warm_checkpoint)
        _require(actor.scenario == scenario, 'Warm checkpoint uses a different fitted Scenario')
        initial = actor.coefficients.copy()
    _require(protocol['initial_policy_sha256'] == actor.provenance()['policy_sha256'], 'CEM initialization binding differs')
    span = max(float(np.ptp(levels)), 1e-6)
    low = np.full(PARAMETER_COUNT, -config.coefficient_bound*span); high = -low
    low[0], high[0] = levels.min(), levels.max()
    mean = initial.copy(); incumbent = initial.copy(); sd = np.full(PARAMETER_COUNT, config.initial_std*span)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, config.training_seed_base, 0x50524943]))
    total_counters = {name: 0 for name in _COUNTERS}
    _require(len(history) == config.generations, 'CEM history is incomplete')
    for generation, row in enumerate(history):
        _require(row['generation'] == generation and row['environment_seeds'] == bank[generation]
            and row['planning_seed'] == plan_seed, 'CEM generation bank differs')
        proposals, kinds, seen = [], [], set()
        def add(value, kind):
            value = np.clip(value, low, high); identity = tuple(value)
            if identity not in seen:
                proposals.append(value.copy()); kinds.append(kind); seen.add(identity)
        add(incumbent, 'incumbent'); add(mean, 'search_mean')
        for index in config.anchors(scenario):
            fixed = np.zeros(PARAMETER_COUNT); fixed[0] = levels[index]
            add(fixed, 'fixed_price_'+str(index))
        while len(proposals) < config.population:
            noise = rng.standard_normal(PARAMETER_COUNT)
            add(mean+sd*noise, 'sample')
            if len(proposals) < config.population: add(mean-sd*noise, 'antithetic_sample')
        _require(len(row['candidates']) == len(proposals) == config.population, 'CEM population differs')
        scores = []
        for i, candidate in enumerate(row['candidates']):
            _require(candidate['candidate'] == i and candidate['kind'] == kinds[i]
                and candidate['invalid_actions'] == 0, 'CEM proposal order, kind or invalid actions differ')
            _close(candidate['coefficients'], proposals[i], 'CEM exact proposal', 0.)
            profits = np.asarray(candidate['episode_discounted_profits'])
            _require(profits.shape == (config.episodes_per_generation,) and np.isfinite(profits).all(), 'CEM episode profits invalid')
            scores.append(float(profits.mean()))
            _close(candidate['mean_discounted_profit'], scores[-1], 'CEM candidate mean', 0.)
            counters = candidate['backend_counters']
            _require(set(counters) == set(_COUNTERS), 'CEM counter fields differ')
            for name, value in counters.items():
                _integer(value, 'CEM '+name); total_counters[name] += value
            _require(counters['decisions'] == config.episodes_per_generation*scenario.horizon, 'CEM candidate decision budget differs')
            if phase == 'rollout':
                draws = config.episodes_per_generation*8*sum(range(scenario.horizon))
                _require(counters['exogenous_draws'] == draws
                    and draws <= counters['future_model_queries'] <= 5*draws
                    and counters['decisions'] <= counters['candidate_plans'] <= 5*counters['decisions']
                    and counters['current_plan_outcomes'] == counters['candidate_plans'], 'Rollout CEM query budget differs')
            else:
                _require(counters['exogenous_draws'] == counters['future_model_queries'] == 0,
                    'Rule pricing fit unexpectedly used future rollout queries')
        order = np.argsort(-np.asarray(scores), kind='stable'); winner = int(order[0])
        _require(row['winner_index'] == winner, 'CEM selected winner differs')
        _close(row['winner_score'], scores[winner], 'CEM winner score', 0.)
        _close(row['incumbent_score'], scores[0], 'CEM incumbent score', 0.)
        incumbent = proposals[winner].copy()
        elite = np.stack([proposals[int(i)] for i in order[:config.elite_count]])
        mean = np.clip((1-config.smoothing)*mean+config.smoothing*elite.mean(axis=0), low, high)
        sd = np.maximum(config.minimum_std*span, (1-config.smoothing)*sd+config.smoothing*elite.std(axis=0))
        _close(row['search_mean'], mean, 'CEM elite mean', 0.)
        _close(row['search_std'], sd, 'CEM elite standard deviation', 0.)
    model = LearnedPricingPolicy.load_checkpoint(directory/'checkpoint.npz')
    _require(model.scenario == scenario and model.provenance()['policy_sha256'] == summary['policy_sha256']
        and summary['backend_counters'] == total_counters
        and summary['final_candidate_index'] == history[-1]['winner_index'], 'CEM final model or counters differ')
    _close(model.coefficients, incumbent, 'CEM final checkpoint coefficients', 0.)
    _close(summary['final_mean_discounted_profit'], history[-1]['winner_score'], 'CEM final training score', 0.)


def _check_ppo(directory, scenario, config):
    from charging_env.rl.paper_ppo import ALGORITHM, PaperPPOConfig, PaperPPOTrainer
    _require(isinstance(config, PaperPPOConfig), 'PPO phase requires PaperPPOConfig')
    config.validate()
    summary, protocol = _read(directory/'summary.json'), _read(directory/'protocol.json')
    _finite(summary)
    expected = _normalize(asdict(config))
    _require(_read(directory/'scenario.json') == protocol['scenario'] == _normalize(asdict(scenario))
        and _read(directory/'training_config.json') == protocol['training_config'] == expected
        and summary['algorithm'] == protocol['algorithm'] == ALGORITHM, 'PPO scenario/configuration/algorithm differs')
    _require(protocol['resumed'] is False and protocol['warmstart_completed_before_run'] is False
        and protocol['raw_rewards'] is True and protocol['reward_shaping'] is False
        and protocol['reward_scale'] == config.reward_scale
        and protocol['training_seed_rule'] == '2*(training_seed_base+seed+episode_count)', 'PPO protocol is not a fresh original-reward run')
    models = {name: PaperPPOTrainer.load_checkpoint(directory/file, device=config.device) for name, file in
        (('final', 'checkpoint.pt'), ('best', 'best_checkpoint.pt'), ('warmstart', 'warmstart_checkpoint.pt'))}
    source = models['final']._source_hashes()
    _require(protocol['source_sha256'] == source, 'PPO training source differs')
    for name, model in models.items():
        _require(model.scenario == model.training_scenario == scenario and model.config == config,
            'PPO '+name+' checkpoint uses different fitted scenario/configuration')
        _require(model.episode_count == model.update_count*config.episodes_per_update
            and model.environment_periods == model.episode_count*scenario.horizon
            and 0 <= model.update_count <= config.updates, 'PPO '+name+' checkpoint counters differ')
    final = models['final']; episodes = config.updates*config.episodes_per_update
    _require(final.update_count == config.updates and final.episode_count == episodes
        and summary['update_count'] == summary['updates'] == config.updates
        and summary['episode_count'] == summary['episodes'] == episodes
        and summary['environment_periods'] == episodes*scenario.horizon and summary['seed'] == config.seed
        and summary['all_metrics_finite'] is True and summary['checkpoint_reload_matches'] is True
        and summary['invalid_evaluation_actions'] == 0
        and summary['checkpoint_sha256'] == _hash(directory/'checkpoint.pt')
        and summary['protocol_sha256'] == _hash(directory/'protocol.json'), 'PPO final completion evidence differs')
    with (directory/'training_metrics.csv').open(newline='') as stream:
        rows = [{name: float(value) for name, value in row.items()} for row in csv.DictReader(stream)]
    _require(len(rows) == config.updates, 'PPO update metrics are incomplete')
    for i, row in enumerate(rows, 1):
        _finite(row)
        _require(row['update'] == i and row['episodes'] == i*config.episodes_per_update
            and row['environment_periods'] == i*config.episodes_per_update*scenario.horizon
            and row['optimizer_steps'] >= 1 and row['early_stopped'] in (0., 1.), 'PPO training metric counters differ')
    warm = _read(directory/'warmstart_summary.json'); _finite(warm)
    demonstration = _read(directory/'demonstration_summary.json')['episodes']
    demo_seeds = [2*(config.warmstart_seed_base+config.seed+i) for i in range(config.warmstart_episodes)]
    demo_periods = config.warmstart_episodes*scenario.horizon
    _require(config.warmstart_episodes > 0 and warm['completed'] is True
        and warm['demonstration_episodes'] == summary['demonstration_episodes'] == config.warmstart_episodes
        and warm['demonstration_periods'] == summary['demonstration_periods'] == demo_periods
        and warm['teacher_pricing_queries'] == warm['teacher_dispatch_queries'] == demo_periods
        and warm['seeds'] == demo_seeds and warm['epochs'] == config.warmstart_epochs
        and warm['shuffle_seed'] == 909000+config.seed
        and warm['minibatch_size'] == config.warmstart_minibatch_size
        and warm['learning_rate'] == config.warmstart_learning_rate
        and warm['reward_scale'] == config.reward_scale and warm['extra_training_reward'] is False
        and warm['dispatch_solver'] == config.warmstart_dispatch_solver
        and warm['coordinate_sweeps'] == (config.warmstart_coordinate_sweeps if config.warmstart_dispatch_solver == 'coordinate' else None)
        and warm['source_sha256'] == source and summary['warmstart'] == warm
        and summary['warmstart_completed'] is True, 'PPO warm-start provenance or budget differs')
    steps = config.warmstart_epochs*math.ceil(demo_periods/config.warmstart_minibatch_size)
    _require(warm['optimizer_steps'] == steps and warm['network_optimizer_steps'] == 4*steps,
        'PPO demonstration optimizer budget differs')
    _episode_ledgers(demonstration, scenario, demo_seeds)
    _close(warm['mean_demonstration_discounted_profit'], np.mean([e['discounted_profit'] for e in demonstration]), 'Demonstration profit mean')
    metrics = _read(directory/'pretraining_metrics.json'); _finite(metrics)
    _require(len(metrics) == config.warmstart_epochs and [r['epoch'] for r in metrics] == list(range(1, config.warmstart_epochs+1))
        and metrics[-1] == warm['final_metrics'], 'PPO pretraining metrics differ')
    for name in ('final', 'warmstart'):
        _require(models[name].warmstart_metadata == dict(completed=True, summary=warm), 'PPO '+name+' initialization metadata differs')
    _require(models['warmstart'].update_count == 0, 'Warm-start checkpoint contains PPO updates')
    initial_eval, warm_eval, final_eval = (_read(directory/name) for name in
        ('initial_evaluation.json', 'warmstart_evaluation.json', 'final_evaluation.json'))
    periodic = _read(directory/'evaluations.json') if config.eval_interval and config.updates >= config.eval_interval else []
    expected_periodic = [i for i in range(1, config.updates+1) if config.eval_interval and i % config.eval_interval == 0]
    _require([r['update'] for r in periodic] == expected_periodic, 'PPO periodic monitor schedule differs')
    for result in (initial_eval, warm_eval, final_eval, *periodic): _evaluation(result, scenario, config)
    expected_history = [dict(update=0, phase='initial', **initial_eval), dict(update=0, phase='warmstart', **warm_eval)]
    expected_history += [dict(phase='periodic', **row) for row in periodic]
    if expected_history[-1]['update'] != config.updates:
        expected_history.append(dict(update=config.updates, phase='final', **final_eval))
    history = _read(directory/'validation_history.json')
    _require(history == expected_history and summary['final_evaluation'] == final_eval, 'PPO monitoring records disagree')
    winner = max(history, key=lambda row: row['mean_discounted_profit'])
    _require(models['best'].update_count == winner['update'], 'PPO monitor-best checkpoint update differs')
    expected_warm = dict(completed=False, summary=None) if winner['phase'] == 'initial' else dict(completed=True, summary=warm)
    _require(models['best'].warmstart_metadata == expected_warm, 'PPO monitor-best initialization status differs')
    for key, value in (('initial_mean_discounted_profit', initial_eval['mean_discounted_profit']),
        ('final_mean_discounted_profit', final_eval['mean_discounted_profit']),
        ('best_validation_profit', winner['mean_discounted_profit'])):
        _close(summary[key], value, 'PPO summary '+key)
    _require(summary['monitoring_evaluation_episodes'] == (3+len(periodic))*config.eval_episodes
        and summary['checkpoint_replay_episodes'] == config.eval_episodes
        and summary['total_evaluation_policy_episodes'] == (4+len(periodic))*config.eval_episodes,
        'PPO monitoring/replay episode budget differs')


def check_training_artifacts(directory, scenario, phase, seed, config, model_sha256,
                             setting, expected_factory_sha256, warm_checkpoint=None):
    """Fail closed on inconsistent saved training; do not run environments.

    ``config`` is the expected learner dataclass, so this also supports small
    development fixtures. The frozen study supplies its 500-update/8,000-episode
    PPO config. The caller separately verifies the completion file's seal and
    every contained file hash, plus the frozen study source snapshot.
    """
    directory = Path(directory); scenario.validate()
    _require(phase in ('warm', 'rollout', 'inventory', 'ppo') and config.seed == seed,
        'Unknown training phase or wrong optimization seed')
    _integer(seed, 'Optimization seed')
    _require(isinstance(model_sha256, str) and len(model_sha256) == 64
        and all(c in '0123456789abcdef' for c in model_sha256), 'Invalid historical model fingerprint')
    expected_binding = dict(model_sha256=model_sha256, phase=phase, seed=seed,
        config=_normalize(asdict(config)), setting=_normalize(setting) if phase == 'inventory' else None)
    _require(_read(directory/'completed.json')['binding'] == expected_binding, 'Historical model/backend binding differs')
    if phase == 'ppo':
        _require(warm_checkpoint is None, 'PPO does not import a CEM checkpoint')
        _check_ppo(directory, scenario, config)
    else:
        _require(isinstance(expected_factory_sha256, str) and len(expected_factory_sha256) == 64,
            'Missing expected backend factory fingerprint')
        _check_cem(directory, scenario, phase, config, expected_factory_sha256, warm_checkpoint)
