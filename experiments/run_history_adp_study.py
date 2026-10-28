"""Fit a historical market model, then compare frozen controllers privately.

The learner receives quote logs and public operational settings. Policy training
and selection use only the fitted simulator. Private environments enter only
history collection and final evaluation, never policy construction.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
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
import run_general_adp_study as general
import run_priced_adp_study as priced
from history_training_checks import check_training_artifacts
from charging_env.config import Scenario
from charging_env.benchmarks.evaluation import paired_comparison
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PriceSearchConfig, fit_price_policy
from charging_env.rl.history_model import HistoricalLog, HistoricalModel, HistoryFitConfig, fit_history_model
from charging_env.rl.history_control import collect_historical_log, evaluate_history_policy
from charging_env.rl.history_control import model_scenario_sha256
from charging_env.rl.rollout_adp import RolloutADPConfig
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer

SCHEMA = 'charging_env.history_adp_study.v1'
SEEDS = (7, 19, 37)
REGIMES = ('baseline', 'volume_up', 'spatiotemporal')
PHASES = ('warm', 'rollout', 'inventory')
SCENARIO_FILE = 'configs/synthetic_25_zones.json'
ACTION_SEED = 917991
LOGGING_SEED = 91731
DEVELOPMENT_SEEDS = (tuple(4_100_000_000+2*i for i in range(4))
    + tuple(4_100_001_001+2*i for i in range(2)) + (4_210_000_001,4_210_000_003)
    + (4_500_000_000,4_500_000_002,4_500_001_001,4_500_004_014,4_500_006_014)
    + tuple(sorted(priced.planning_query_seeds(2,7,(4_210_000_001,4_210_000_003)))))
FIT_SEEDS = tuple(3_100_000_001+2*i for i in range(120))
HOLDOUT_SEEDS = tuple(3_100_100_001+2*i for i in range(32))
POLICY_VALIDATION = tuple(3_200_000_001+2*i for i in range(16))
RULE_VALIDATION = tuple(3_201_000_001+2*i for i in range(32))
MONITOR = tuple(3_202_000_001+2*i for i in range(32))
TEST = {name: tuple(3_500_000_001+j*1_000_000+2*i for i in range(128))
        for j, name in enumerate(REGIMES)}
PPO_CANDIDATES = ('final', 'best', 'warmstart')
PPO_FILES = dict(final='checkpoint.pt', best='best_checkpoint.pt', warmstart='warmstart_checkpoint.pt')
PRICE_CANDIDATES = ('fixed', 'warm', 'refined')
read, write, digest, seal, sealed = general.read, general.write, general.digest, general.seal, general.sealed
normalized, scenario_from, check_result = general.normalized, general._scenario_data, general.check_result


def price_config(phase, seed):
    if phase not in PHASES or seed not in SEEDS:
        raise ValueError('Unknown pricing phase or seed')
    warm = phase == 'warm'
    return PriceSearchConfig(seed=seed, generations=12 if warm else 4,
        population=24 if warm else 8, elite_count=6 if warm else 3,
        episodes_per_generation=16 if warm else 4,
        training_seed_base=3_300_000_000+PHASES.index(phase)*1_000_000+SEEDS.index(seed)*100_000,
        planning_seed=seed, action_seed=ACTION_SEED, fixed_anchor_indices=None if warm else ())


def ppo_config(seed):
    if seed not in SEEDS:
        raise ValueError('Unknown PPO seed')
    offset = SEEDS.index(seed)*100_000
    return PaperPPOConfig(seed=seed, updates=500, episodes_per_update=16,
        epochs=6, minibatch_size=192, hidden_size=64,
        actor_learning_rate=1e-4, critic_learning_rate=1e-3,
        gae_lambda=1., normalize_advantages=True, reward_scale=20.,
        dispatch_entropy_coef=.005, target_kl=.03,
        warmstart_episodes=256, warmstart_epochs=100,
        warmstart_seed_base=1_710_000_000+offset,
        warmstart_dispatch_solver='coordinate', warmstart_coordinate_sweeps=2,
        training_seed_base=1_700_000_000+offset,
        eval_episodes=len(MONITOR), eval_interval=100,
        eval_seed_base=MONITOR[0], eval_action_seed=ACTION_SEED,
        eval_policy='stochastic', torch_threads=1).validate()


def shifted_scenarios(base):
    """Private fixtures: stable, higher counts, and relocated/delayed demand."""
    rates = base.resolved_arrival_rates
    xy = np.asarray(base.zone_xy_km)
    mirrored = xy.min(axis=0)+xy.max(axis=0)-xy
    permutation = []
    for point in mirrored:
        matches = np.flatnonzero(np.all(np.isclose(xy, point, atol=1e-12, rtol=0), axis=1))
        if len(matches) != 1:
            raise ValueError('Declared spatial shift requires a centrally symmetric zone layout')
        permutation.append(int(matches[0]))
    relocated = np.roll(rates, 3, axis=0)[:, permutation]
    if not np.isclose(relocated.sum(), rates.sum(), rtol=0, atol=1e-8):
        raise ValueError('Spatial/time shift must preserve expected daily demand')
    def at(values):
        return replace(base, arrival_rates=tuple(map(tuple, values))).validate()
    return dict(baseline=base, volume_up=at(1.4*rates), spatiotemporal=at(relocated))


def environment_banks():
    banks = dict(history_fit=set(FIT_SEEDS), history_holdout=set(HOLDOUT_SEEDS),
        policy_validation=set(POLICY_VALIDATION), rule_validation=set(RULE_VALIDATION), monitor=set(MONITOR))
    banks.update({f'test/{name}': set(seeds) for name, seeds in TEST.items()})
    for seed in SEEDS:
        for phase in PHASES:
            banks[f'pricing/{phase}/{seed}'] = set(sum(price_config(phase, seed).generation_seeds(), []))
        c = ppo_config(seed)
        banks[f'ppo/{seed}'] = {2*(c.training_seed_base+seed+i) for i in range(c.updates*c.episodes_per_update)}
        banks[f'demonstration/{seed}'] = {2*(c.warmstart_seed_base+seed+i) for i in range(c.warmstart_episodes)}
    seen = set()
    for name, values in banks.items():
        if seen & values:
            raise ValueError('Overlapping environment bank: '+name)
        seen |= values
    return banks


def source_files():
    return sorted([*ROOT.glob('src/charging_env/**/*.py'),
        *(ROOT/'experiments'/name for name in ('run_history_adp_study.py', 'history_training_checks.py', 'run_priced_adp_study.py',
          'run_general_adp_study.py', 'run_fitted_value_study.py')), ROOT/SCENARIO_FILE])


def init(study, control_root):
    if study.exists():
        raise FileExistsError('Choose an unused study directory')
    base = Scenario.from_json(ROOT/SCENARIO_FILE)
    if (base.n_zones, base.n_mcv) != (25, 10):
        raise ValueError('This declared study requires 25 zones and ten vehicles')
    historical, origin = priced.historical_seeds(control_root)
    historical.update(DEVELOPMENT_SEEDS)
    # Registries preserve streams even when an old bundle no longer has raw JSON.
    for path in sorted((control_root/'outputs').rglob('historical_seeds.npz')):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                values = archive[key]
                if values.dtype.kind not in 'iu' or values.ndim != 1:
                    raise ValueError('Malformed historical seed registry')
                historical.update(map(int, values))
        origin[str(path.relative_to(control_root))] = digest(path)
    banks = environment_banks()
    if set.union(*banks.values()) & historical:
        raise ValueError('A declared environment bank was used in an earlier study')
    queries=set()
    for seed in SEEDS:
        episodes=tuple(sum(price_config('rollout',seed).generation_seeds(),[]))+POLICY_VALIDATION+tuple(sum(TEST.values(),()))
        used=priced.planning_query_seeds(base.horizon,seed,episodes)
        if used & (historical | set.union(*banks.values()) | queries):
            raise ValueError('Planning queries overlap historical, environment or other planning streams')
        queries |= used
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files()}
    study.mkdir(parents=True)
    (study/'provenance').mkdir()
    np.savez_compressed(study/'provenance/historical_seeds.npz', seeds=np.asarray(sorted(historical), dtype=np.int64))
    np.savez_compressed(study/'provenance/planned_query_seeds.npz',seeds=np.asarray(sorted(queries),dtype=np.int64))
    protocol = dict(schema=SCHEMA, seeds=list(SEEDS), action_seed=ACTION_SEED, logging_seed=LOGGING_SEED,
        private_environments={k: normalized(asdict(v)) for k, v in shifted_scenarios(base).items()},
        model_fit_config=normalized(asdict(HistoryFitConfig())),
        history_fit_seeds=list(FIT_SEEDS), history_holdout_seeds=list(HOLDOUT_SEEDS),
        policy_validation_seeds=list(POLICY_VALIDATION), rule_validation_seeds=list(RULE_VALIDATION),
        monitor_seeds=list(MONITOR), test_seeds={k: list(v) for k, v in TEST.items()},
        price_configs={f'{phase}/{seed}': normalized(asdict(price_config(phase, seed))) for phase in PHASES for seed in SEEDS},
        ppo_configs={str(seed): normalized(asdict(ppo_config(seed))) for seed in SEEDS},
        environment_bank_sizes={k: len(v) for k, v in banks.items()},
        source_sha256=hashes, historical_source_sha256=origin,
        historical_seed_sha256=digest(study/'provenance/historical_seeds.npz'),
        historical_seed_count=len(historical), final_policy_episodes=3840,
        planned_query_seed_count=len(queries), planned_query_seed_sha256=digest(study/'provenance/planned_query_seeds.npz'),
        information='All controllers use a history-fitted Scenario. True rates and behavioral parameters are never supplied to a controller. Current accepted requests are observed at dispatch.',
        data='Synthetic complete quote logs, including outside choices. Whole-day fit/holdout split; randomized price offers. No real-city observations.',
        fitting='Poisson time-zone rates, uniform request sizes, and common-intercept multinomial-logit customer choice; families assumed, parameters estimated.',
        policy_learning='PPO and pricing train in the same fitted simulator. Inventory parameters are selected there. Historical holdout logs provide prediction diagnostics only.',
        pricing_selection='Fixed inventory-selected price, initial learned and refined policies; first maximum on 16 fitted-model days, separately per dispatcher/seed.',
        ppo_selection='Final, monitor-best, warmstart checkpoints in that tie order; fixed stochastic evaluation on 16 fitted-model days. Monitor-best may be initial/untrained.',
        testing='128 fresh paired days in each private regime; no per-regime tuning or online model adaptation; identical frozen checkpoints.',
        shifts='Volume multiplies arrivals by1.4. Spatiotemporal rotates zones180degrees and delays hourly intensities by3 periods circularly; daily expected total unchanged.',
        uncertainty='5000 paired scenario bootstrap resamples, conditional on one historical fit and three optimization seeds; not uncertainty over new historical datasets.',
        limitations='Known physical/network/tariff settings and assumed Poisson/uniform/logit families. Full quote traffic and outside choices must be observable. Simulator-trained PPO; unequal online model-query costs.')
    seal(study/'protocol.json', protocol)
    for name in hashes:
        target=study/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    verify(study)
    print('Initialized '+str(study), flush=True)


def verify(study):
    p = sealed(study/'protocol.json')
    expected = dict(action_seed=ACTION_SEED, logging_seed=LOGGING_SEED, final_policy_episodes=3840,
        model_fit_config=normalized(asdict(HistoryFitConfig())),
        private_environments={k: normalized(asdict(v)) for k,v in shifted_scenarios(Scenario.from_json(ROOT/SCENARIO_FILE)).items()},
        environment_bank_sizes={k:len(v) for k,v in environment_banks().items()},
        history_fit_seeds=list(FIT_SEEDS), history_holdout_seeds=list(HOLDOUT_SEEDS),
        policy_validation_seeds=list(POLICY_VALIDATION), rule_validation_seeds=list(RULE_VALIDATION),
        monitor_seeds=list(MONITOR), test_seeds={k:list(v) for k,v in TEST.items()},
        price_configs={f'{phase}/{seed}':normalized(asdict(price_config(phase,seed))) for phase in PHASES for seed in SEEDS},
        ppo_configs={str(seed):normalized(asdict(ppo_config(seed))) for seed in SEEDS})
    if p['schema']!=SCHEMA or p['seeds']!=list(SEEDS) or any(p[k]!=v for k,v in expected.items()):
        raise ValueError('Declared study settings changed')
    if p['source_sha256'] != {str(f.relative_to(ROOT)):digest(f) for f in source_files()}:
        raise ValueError('Frozen source changed; do not mix implementations in a study')
    if any(digest(study/'source_snapshot'/name)!=sha for name,sha in p['source_sha256'].items()):
        raise ValueError('Frozen source snapshot changed')
    if digest(study/'provenance/historical_seeds.npz') != p['historical_seed_sha256']:
        raise ValueError('Historical seed registry changed')
    if (p['planned_query_seed_count']!=658944
            or digest(study/'provenance/planned_query_seeds.npz')!=p['planned_query_seed_sha256']):
        raise ValueError('Declared planning seed registry changed')
    historical=priced.history_bank(str(study/'provenance/historical_seeds.npz'),p['historical_seed_sha256'])
    if len(historical)!=p['historical_seed_count'] or set.union(*environment_banks().values()) & historical:
        raise ValueError('Historical seed registry or independent environment banks differ')
    return p


def model_path(study):
    return study/'model/checkpoint.npz'


def fitted_model(study):
    binding = sealed(study/'model/binding.json')
    if (binding['protocol_sha256'] != digest(study/'protocol.json')
            or binding['checkpoint_sha256'] != digest(model_path(study))
            or binding['fit_log_sha256'] != digest(study/'history/fit.npz')
            or binding['holdout_log_sha256'] != digest(study/'history/holdout.npz')
            or any(digest(study/'model'/name)!=sha for name,sha in binding['evidence_sha256'].items())):
        raise ValueError('Fitted-model binding changed')
    evidence={'checkpoint.npz','scenario.json','fit_report.json','holdout_diagnostics.json'}
    if set(binding['evidence_sha256'])!=evidence: raise ValueError('Incomplete model evidence')
    model=HistoricalModel.load_checkpoint(model_path(study)); report=read(study/'model/fit_report.json')
    if (model.scenario!=Scenario.from_json(study/'model/scenario.json') or model.provenance()['fit']!=report
            or report['config']!=normalized(asdict(HistoryFitConfig()))
            or report['source_sha256']!=digest(ROOT/'src/charging_env/rl/history_model.py')
            or report['historical_log']!=binding['fit_log_provenance']
            or report['fit_days']!=len(FIT_SEEDS) or not report['choice']['converged']
            or not report['no_hidden_behavior_parameters'] or report['held_out_data_used']):
        raise ValueError('Historical fit provenance differs')
    return model


def history(study):
    p=verify(study); dest=study/'history'
    if (dest/'binding.json').exists():
        b=sealed(dest/'binding.json')
        if b['protocol_sha256']!=digest(study/'protocol.json') or any(digest(dest/k)!=v for k,v in b['files'].items()):
            raise ValueError('Historical logs changed')
        return
    if dest.exists(): raise FileExistsError('Incomplete history is preserved; use a new study')
    dest.mkdir()
    truth=scenario_from(p['private_environments']['baseline'])
    for name,seeds,start in [('fit',FIT_SEEDS,0),('holdout',HOLDOUT_SEEDS,len(FIT_SEEDS))]:
        log=collect_historical_log(truth,seeds,logging_seed=LOGGING_SEED,start_day=start)
        log.save(dest/(name+'.npz'))
        print('Collected '+name+' historical days: '+str(len(seeds)),flush=True)
    seal(dest/'binding.json',dict(protocol_sha256=digest(study/'protocol.json'),
        files={name+'.npz':digest(dest/(name+'.npz')) for name in ('fit','holdout')}))


def fit(study):
    p=verify(study); history(study)
    if (study/'model/binding.json').exists(): fitted_model(study); return
    dest=study/'model'
    if dest.exists(): raise FileExistsError('Incomplete model fit is preserved')
    dest.mkdir()
    log=HistoricalLog.load(study/'history/fit.npz')
    holdout=HistoricalLog.load(study/'history/holdout.npz')
    if (log.day_ids.tolist()!=list(range(len(FIT_SEEDS)))
            or holdout.day_ids.tolist()!=list(range(len(FIT_SEEDS),len(FIT_SEEDS)+len(HOLDOUT_SEEDS)))
            or log.public_config!=holdout.public_config):
        raise ValueError('Historical day split or public metadata differs')
    model,report=fit_history_model(log,HistoryFitConfig(**p['model_fit_config']))
    if not report['choice']['converged']: raise ValueError('Choice fit did not converge; training has not started')
    model.save_checkpoint(model_path(study)); model.scenario.to_json(dest/'scenario.json')
    write(dest/'fit_report.json',report)
    write(dest/'holdout_diagnostics.json',model.diagnostics(holdout))
    seal(dest/'binding.json',dict(protocol_sha256=digest(study/'protocol.json'),
        checkpoint_sha256=digest(model_path(study)),fit_log_sha256=digest(study/'history/fit.npz'),
        holdout_log_sha256=digest(study/'history/holdout.npz'),
        fit_log_provenance=log.provenance(),
        evidence_sha256={f.name:digest(f) for f in dest.iterdir() if f.is_file()}))
    fitted_model(study)
    print('Historical market model fitted; held-out diagnostics saved.',flush=True)


def train_path(study,phase,seed):
    return study/'training'/phase/f'seed_{seed}'


def rule_setting(study):
    scenario=fitted_model(study).scenario; rows=[]
    for setting in priced.rule_grid(scenario):
        spec=dict(backend='inventory',kind='fixed',seed=7,setting=setting,label=setting['id'])
        result=inspect_evaluation(study,RULE_VALIDATION,study/'rules'/f"{setting['id']}.json",spec,scenario=scenario)
        rows.append(dict(specification=setting,mean_discounted_profit=result['mean_discounted_profit']))
    winner=rows[int(np.argmax([r['mean_discounted_profit'] for r in rows]))]
    expected=dict(model_sha256=digest(model_path(study)),candidates=rows,winner=winner)
    if sealed(study/'rules/selection.json')!=expected: raise ValueError('Inventory selection differs from full grid')
    return winner['specification']


def pricing_actor(study,scenario,backend,kind,seed,setting):
    if kind=='fixed': return priced.constant_pricer(scenario,setting['price_index'])
    phase='warm' if kind=='warm' else backend
    model=LearnedPricingPolicy.load_checkpoint(train_path(study,phase,seed)/'checkpoint.npz')
    if model.scenario != scenario: raise ValueError('Pricing checkpoint uses a different market model')
    return model


def evaluation(study,truth,scenario,seeds,path,spec):
    """An evaluator owns truth; controller factories receive only fitted state."""
    if path.exists(): return inspect_evaluation(study,seeds,path,spec,truth=truth,scenario=scenario)
    if any(path.with_suffix(suffix).exists() for suffix in ('.planning.json','.pricing.json')):
        raise FileExistsError('Incomplete evaluation traces are preserved: '+str(path))
    captured=[]
    def factory(fitted):
        if spec['backend']=='ppo':
            obj=PaperPPOTrainer.load_checkpoint(train_path(study,'ppo',spec['seed'])/PPO_FILES[spec['kind']])
            if obj.scenario != fitted or obj.training_scenario != fitted:
                raise ValueError('PPO checkpoint was not trained in the historical model')
            obj.set_evaluation_mode(deterministic=False)
        else:
            actor=pricing_actor(study,fitted,spec['backend'],spec['kind'],spec['seed'],spec['setting'])
            obj=priced.make_backend(fitted,spec['backend'],actor,spec['seed'],spec['setting'])
        captured.append(obj)
        return obj
    result=evaluate_history_policy(truth,scenario,seeds,factory,action_seed=ACTION_SEED,label=spec['label'])
    check_result(result,seeds)
    if len(captured)!=1: raise ValueError('Controller construction must happen exactly once')
    obj=captured[0]
    result.update(specification=spec,training_seed=spec['seed'],historical_model_sha256=digest(model_path(study)),
        evaluation_environment_sha256=model_scenario_sha256(truth))
    evidence={}
    for attribute,suffix in [('decisions','.planning.json'),('pricing_decisions','.pricing.json')]:
        if hasattr(obj,attribute):
            target=path.with_suffix(suffix);write(target,getattr(obj,attribute));evidence[target.name]=digest(target)
    result['backend_statistics']=getattr(obj,'statistics',None)
    if hasattr(obj,'pricing_decisions'):
        result['active_pricing']=priced.summarize_pricing(obj.pricing_decisions,scenario)
    result['evidence_sha256']=evidence
    if spec['backend']=='ppo':
        result['checkpoint_sha256']=digest(train_path(study,'ppo',spec['seed'])/PPO_FILES[spec['kind']])
        result['checkpoint_update']=obj.update_count
        result['checkpoint_warmstart_completed']=obj.warmstart_metadata['completed']
    elif spec['kind']!='fixed':
        phase='warm' if spec['kind']=='warm' else spec['backend']
        result['checkpoint_sha256']=digest(train_path(study,phase,spec['seed'])/'checkpoint.npz')
    seal(path,result)
    return inspect_evaluation(study,seeds,path,spec,truth=truth,scenario=scenario)


def inspect_evaluation(study,seeds,path,spec,*,truth=None,scenario=None):
    r=sealed(path);check_result(r,seeds)
    if scenario is None: scenario=fitted_model(study).scenario
    if truth is None: truth=scenario
    if (r['specification']!=spec or r['historical_model_sha256']!=digest(model_path(study))
            or r['action_seed']!=ACTION_SEED
            or r['evaluation_environment_sha256']!=model_scenario_sha256(truth)):
        raise ValueError('Evaluation model or settings changed')
    b=r['history_boundary']; expected_order=[(seed,t) for seed in seeds for t in range(scenario.horizon)]
    model_sha=model_scenario_sha256(scenario)
    if (b['model_scenario_sha256']!=model_sha or b['forecast_source']!='fitted_historical_model'
            or b['true_model_passed_to_controller'] is not False
            or b['controller_constructed_by_fitted_model_factory'] is not True
            or [(v['episode_key'],v['period']) for v in b['decisions']]!=expected_order
            or any(v['model_scenario_sha256']!=model_sha or v['future_model_queries']<0 for v in b['decisions'])
            or b['model_queries']!=sum(v['future_model_queries'] for v in b['decisions'])):
        raise ValueError('Historical observation or controller boundary differs')
    for name,sha in r['evidence_sha256'].items():
        if digest(path.parent/name)!=sha: raise ValueError('Evaluation trace changed')
    if spec['backend']=='ppo':
        checkpoint=train_path(study,'ppo',spec['seed'])/PPO_FILES[spec['kind']]
    elif spec['kind']!='fixed':
        checkpoint=train_path(study,'warm' if spec['kind']=='warm' else spec['backend'],spec['seed'])/'checkpoint.npz'
    else: checkpoint=None
    if checkpoint is not None and r['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('Evaluated checkpoint changed')
    if spec['backend']!='ppo':
        prices=read(path.with_suffix('.pricing.json'))
        if ([(v['episode_key'],v['period']) for v in prices]!=expected_order
                or r['active_pricing']!=priced.summarize_pricing(prices,scenario)):
            raise ValueError('Current pricing trace differs')
    if spec['backend']=='rollout':
        trace=read(path.with_suffix('.planning.json')); stats=r['backend_statistics']
        used=general.check_planning_trace(trace,stats,scenario,RolloutADPConfig(seed=spec['seed'],rollout_samples=8),seeds)
        protocol=sealed(study/'protocol.json')
        forbidden=priced.history_bank(str(study/'provenance/historical_seeds.npz'),protocol['historical_seed_sha256'])
        if used & (forbidden | set.union(*environment_banks().values())):
            raise ValueError('Planning streams overlap historical or environment streams')
        if (b['model_queries']!=stats['future_model_queries']
                or stats['future_pricing_calls']!=stats['future_model_queries']
                or stats['current_pricing_calls']!=len(expected_order)
                or stats['current_active_price_counts']!=r['active_pricing']['counts']
                or any(v['future_pricing_calls']!=v['future_model_queries'] for v in trace)
                or [v['future_model_queries'] for v in b['decisions']]!=[v['future_model_queries'] for v in trace]):
            raise ValueError('Fitted future query or pricing count differs')
    elif b['model_queries']!=0: raise ValueError('A nonplanning policy made future model queries')
    return r


def rule_job(job):
    directory,setting=job;study=Path(directory)
    verify(study);scenario=fitted_model(study).scenario
    spec=dict(backend='inventory',kind='fixed',seed=7,setting=setting,label=setting['id'])
    result=evaluation(study,scenario,scenario,RULE_VALIDATION,study/'rules'/f"{setting['id']}.json",spec)
    print('Inventory model-validation '+setting['id']+f": {result['mean_discounted_profit']:.3f}",flush=True)
    return dict(specification=setting,mean_discounted_profit=result['mean_discounted_profit'])


def run_jobs(function,jobs,workers):
    if workers==1: return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function,jobs))


def rules(study,workers):
    verify(study);scenario=fitted_model(study).scenario
    rows=run_jobs(rule_job,[(str(study),s) for s in priced.rule_grid(scenario)],workers)
    winner=rows[int(np.argmax([r['mean_discounted_profit'] for r in rows]))]
    value=dict(model_sha256=digest(model_path(study)),candidates=rows,winner=winner)
    path=study/'rules/selection.json'
    if path.exists():
        if sealed(path)!=value: raise ValueError('Inventory selection changed')
    else: seal(path,value)


def train_seed(job):
    directory,seed=job;study=Path(directory);verify(study)
    scenario=fitted_model(study).scenario;setting=rule_setting(study)
    for phase in PHASES:
        destination=train_path(study,phase,seed)
        binding=dict(model_sha256=digest(model_path(study)),phase=phase,seed=seed,
            config=normalized(asdict(price_config(phase,seed))),setting=setting if phase=='inventory' else None)
        if (destination/'completed.json').exists():
            check_training(study,phase,seed);continue
        if destination.exists(): raise FileExistsError('Incomplete training is preserved: '+str(destination))
        initial=None if phase=='warm' else LearnedPricingPolicy.load_checkpoint(train_path(study,'warm',seed)/'checkpoint.npz')
        factory=lambda actor,planning_seed:priced.make_backend(scenario,phase,actor,planning_seed,setting)
        fit_price_policy(scenario,price_config(phase,seed),factory,destination,initial_policy=initial)
        seal(destination/'completed.json',dict(binding=binding,
            files={f.name:digest(f) for f in destination.iterdir() if f.is_file()}))
        check_training(study,phase,seed)
    destination=train_path(study,'ppo',seed)
    if (destination/'completed.json').exists(): check_training(study,'ppo',seed);return
    if destination.exists(): raise FileExistsError('Incomplete PPO training is preserved')
    trainer=PaperPPOTrainer(scenario,ppo_config(seed));trainer.train(destination)
    seal(destination/'completed.json',dict(binding=dict(model_sha256=digest(model_path(study)),phase='ppo',seed=seed,
        config=normalized(asdict(ppo_config(seed))),setting=None),
        files={f.name:digest(f) for f in destination.iterdir() if f.is_file()}))
    check_training(study,'ppo',seed)
    print('Completed all training for seed '+str(seed),flush=True)


def check_training(study,phase,seed):
    directory=train_path(study,phase,seed);done=sealed(directory/'completed.json')
    expected=dict(model_sha256=digest(model_path(study)),phase=phase,seed=seed,
        config=normalized(asdict(ppo_config(seed) if phase=='ppo' else price_config(phase,seed))),
        setting=rule_setting(study) if phase=='inventory' else None)
    if (done['binding']!=expected or any(digest(directory/name)!=sha for name,sha in done['files'].items())
            or set(done['files'])!={f.name for f in directory.iterdir() if f.is_file() and f.name not in ('completed.json','completed.sha256')}):
        raise ValueError('Training evidence changed')
    scenario=fitted_model(study).scenario
    check_training_artifacts(directory,scenario,phase,seed,
        ppo_config(seed) if phase=='ppo' else price_config(phase,seed),digest(model_path(study)),
        rule_setting(study) if phase=='inventory' else None,digest(Path(__file__)),
        train_path(study,'warm',seed)/'checkpoint.npz' if phase in ('rollout','inventory') else None)
    return done


def selection_job(job):
    directory,backend,seed=job;study=Path(directory);verify(study)
    scenario=fitted_model(study).scenario;setting=rule_setting(study)
    for phase in (('ppo',) if backend=='ppo' else ('warm',backend)):
        check_training(study,phase,seed)
    kinds=PPO_CANDIDATES if backend=='ppo' else PRICE_CANDIDATES
    rows=[]
    for kind in kinds:
        spec=dict(backend=backend,kind=kind,seed=seed,setting=setting,label=f'{backend}_{kind}_{seed}')
        path=study/'validation'/f'{backend}_{kind}_{seed}.json'
        result=evaluation(study,scenario,scenario,POLICY_VALIDATION,path,spec)
        rows.append(dict(specification=spec,mean_discounted_profit=result['mean_discounted_profit'],
            result_sha256=digest(path),checkpoint_update=result.get('checkpoint_update'),
            checkpoint_warmstart_completed=result.get('checkpoint_warmstart_completed')))
    winner=rows[int(np.argmax([r['mean_discounted_profit'] for r in rows]))]
    result=dict(candidates=rows,winner=winner,model_sha256=digest(model_path(study)))
    path=study/'selection'/f'{backend}_{seed}.json'
    if path.exists():
        if sealed(path)!=result: raise ValueError('Controller selection changed')
    else: seal(path,result)
    print(f"Selected {backend} seed {seed}: {winner['specification']['kind']}",flush=True)
    return result


def final_specs(study):
    result={}
    for backend in ('rollout','ppo','inventory'):
        for seed in SEEDS:
            row=sealed(study/'selection'/f'{backend}_{seed}.json')
            spec=dict(row['winner']['specification'],label=f'{backend}_{seed}')
            result[spec['label']]=spec
    result['fixed_inventory']=dict(backend='inventory',kind='fixed',seed=7,
        setting=rule_setting(study),label='fixed_inventory')
    return result


def evidence_hashes(study):
    return {str(p.relative_to(study)):digest(p) for folder in ('history','model','rules','training','validation','selection')
        for p in sorted((study/folder).rglob('*')) if p.is_file()}


def final_job(job):
    directory,regime,spec=job;study=Path(directory);p=verify(study)
    scenario=fitted_model(study).scenario
    truth=scenario_from(p['private_environments'][regime])
    result=evaluation(study,truth,scenario,TEST[regime],study/'test'/regime/(spec['label']+'.json'),spec)
    print(f"{regime} {spec['label']}: {result['mean_discounted_profit']:.3f}",flush=True)
    return result


def evaluate(study,workers):
    p=verify(study)
    for seed in SEEDS:
        for phase in (*PHASES,'ppo'): check_training(study,phase,seed)
    # Reconstruct selections from stored validation results, without private tests.
    for backend in ('rollout','ppo','inventory'):
        for seed in SEEDS: selection_job((str(study),backend,seed))
    specs=final_specs(study)
    frozen=dict(protocol_sha256=digest(study/'protocol.json'),model_sha256=digest(model_path(study)),
        specifications=specs,evidence_sha256=evidence_hashes(study))
    target=study/'test/frozen_inputs.json'
    if target.exists():
        if sealed(target)!=frozen: raise ValueError('Final inputs changed')
    else: seal(target,frozen)
    if (study/'analysis.json').exists():
        analysis=sealed(study/'analysis.json')
        if any(digest(study/n)!=h for n,h in analysis['result_sha256'].items()):
            raise ValueError('Final result changed')
        show(study);return
    started=time.perf_counter()
    run_jobs(final_job,[(str(study),regime,spec) for regime in REGIMES for spec in specs.values()],workers)
    cases={};count=0
    for regime in REGIMES:
        records={name:inspect_evaluation(study,TEST[regime],study/'test'/regime/(name+'.json'),spec,
            truth=scenario_from(p['private_environments'][regime]))
            for name,spec in specs.items()}
        count+=sum(r['n_episodes'] for r in records.values())
        groups={backend:general.aggregate_models([records[f'{backend}_{seed}'] for seed in SEEDS],backend)
            for backend in ('rollout','ppo','inventory')}
        for backend,role in [('rollout','pricing_training_and_planning_seed'),('ppo','training_seed'),('inventory','pricing_training_seed')]:
            groups[backend]['seed_role']=role
        groups['fixed_inventory']=records['fixed_inventory']
        comparisons={name:paired_comparison(groups['rollout'],r) for name,r in groups.items() if name!='rollout'}
        per_seed={str(seed):{name:paired_comparison(records[f'rollout_{seed}'],
            records[f'{name}_{seed}'] if name!='fixed_inventory' else records[name])
            for name in ('ppo','inventory','fixed_inventory')} for seed in SEEDS}
        cases[regime]=dict(groups=groups,comparisons=comparisons,per_seed=per_seed,
            by_seed={backend:{str(seed):records[f'{backend}_{seed}']['mean_discounted_profit'] for seed in SEEDS}
                for backend in ('rollout','ppo','inventory')},
            consistent_advantage={name:all(per_seed[str(seed)][name]['paired_bootstrap_confidence_interval_95'][0]>0
                for seed in SEEDS) for name in comparisons})
    if count!=p['final_policy_episodes']: raise ValueError('Incomplete final evaluation')
    verify(study)
    if evidence_hashes(study)!=frozen['evidence_sha256']: raise ValueError('Frozen inputs changed during final evaluation')
    seal(study/'analysis.json',dict(schema=SCHEMA,regimes=cases,policy_episodes=count,
        protocol_sha256=digest(study/'protocol.json'),model_sha256=digest(model_path(study)),
        elapsed_seconds=time.perf_counter()-started,
        result_sha256={str(f.relative_to(study)):digest(f) for regime in REGIMES
            for f in (study/'test'/regime).iterdir() if f.is_file()},
        selection=specs,uncertainty=p['uncertainty']))
    show(study)


def show(study):
    a=sealed(study/'analysis.json')
    for regime,c in a['regimes'].items():
        print('\n'+regime)
        for name,r in c['groups'].items(): print(f"  {name:20s} {r['mean_discounted_profit']:10.3f}")
        for name,r in c['comparisons'].items():
            lo,hi=r['paired_bootstrap_confidence_interval_95']
            print(f"  rollout minus {name}: {r['mean_difference']:.3f} [{lo:.3f}, {hi:.3f}]")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('init','history','fit','rules','train','select','evaluate','show'))
    parser.add_argument('--study-dir',type=Path,default=ROOT/'outputs/history_adp_study_20261027')
    parser.add_argument('--control-root',type=Path)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args(argv);study=args.study_dir.resolve()
    if args.phase=='init':
        if args.control_root is None: parser.error('--control-root is required for historical seed checks')
        init(study,args.control_root.resolve())
    elif args.phase=='history': history(study)
    elif args.phase=='fit': fit(study)
    elif args.phase=='rules': rules(study,args.workers)
    elif args.phase=='train': run_jobs(train_seed,[(str(study),s) for s in (SEEDS if args.seed is None else (args.seed,))],args.workers)
    elif args.phase=='select': run_jobs(selection_job,[(str(study),b,s) for b in ('rollout','ppo','inventory')
        for s in (SEEDS if args.seed is None else (args.seed,))],args.workers)
    elif args.phase=='evaluate': evaluate(study,args.workers)
    else: show(study)


if __name__=='__main__': main()
