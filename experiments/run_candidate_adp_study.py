"""Controlled same-price comparison: rollout, candidate DDQN and hybrid ADP.

Preparation/selection never opens test days. Evaluation is restartable by
immutable chunk files; latency is measured separately after parallel work ends.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np

from charging_env.config import Scenario
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.benchmarks.policies import service_first_dispatch
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PriceSearchConfig, fit_price_policy
from charging_env.rl.history_model import HistoricalModel, fit_history_model
from charging_env.rl.history_control import collect_historical_log, evaluate_history_policy
from charging_env.rl.priced_rollout import PricedRolloutADPPolicy, _price_levels
from charging_env.rl.rollout_adp import RolloutADPConfig

SCHEMA = 'charging_env.candidate_replacement_study.v1'
SEEDS = (7, 19, 37)
METHODS = ('rollout', 'q', 'hybrid')
REGIMES = ('baseline', 'volume_up', 'spatiotemporal')
FIT = tuple(3_100_000_001+2*i for i in range(120))
HOLDOUT = tuple(3_100_100_001+2*i for i in range(32))
VALIDATION = tuple(3_200_000_001+2*i for i in range(16))
TEST = {name: tuple(3_500_000_001+j*1_000_000+2*i for i in range(128))
        for j, name in enumerate(REGIMES)}
ACTION_SEED = 917991


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.01 * 2**attempt, .25))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [*ROOT.glob('src/charging_env/**/*.py'), Path(__file__)]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(paths)}


def model_and_pricer(study):
    scenario = HistoricalModel.load_checkpoint(study/'shared/history_model.npz').scenario
    pricer = LearnedPricingPolicy.load_checkpoint(study/'shared/pricing_checkpoint.npz')
    if scenario != pricer.scenario:
        raise ValueError('Shared model and pricing checkpoint differ')
    return scenario, pricer


class WarmBackend:
    def __init__(self, pricer):
        self.scenario, self.pricer = pricer.scenario, deepcopy(pricer)
        self.statistics = {}

    def reset_episode(self, episode_key=0, action_seed=None):
        pass

    def prices(self, obs, rng=None):
        return _price_levels(obs, self.scenario, self.pricer)[0]

    def dispatch(self, obs, rng=None):
        return service_first_dispatch(obs, self.scenario, .2)


def warm_factory(pricer, seed):
    return WarmBackend(pricer)


def shifted_scenarios(base):
    rates = base.resolved_arrival_rates
    xy = np.asarray(base.zone_xy_km)
    mirrored = xy.min(axis=0)+xy.max(axis=0)-xy
    permutation = []
    for point in mirrored:
        matches = np.flatnonzero(np.all(np.isclose(xy, point, atol=1e-12, rtol=0), axis=1))
        if len(matches) != 1:
            raise ValueError('Spatial shift needs a centrally symmetric layout')
        permutation.append(int(matches[0]))
    relocated = np.roll(rates, 3, axis=0)[:, permutation]
    def at(values):
        return replace(base, arrival_rates=tuple(map(tuple, values))).validate()
    return dict(baseline=base, volume_up=at(1.4*rates), spatiotemporal=at(relocated))


def prepare(study, scenario_file, episodes):
    if (study/'protocol.json').exists():
        p = read(study/'protocol.json')
        if p['training_episodes'] != episodes or p['scenario_sha256'] != digest(scenario_file):
            raise ValueError('Existing study settings differ')
        model_and_pricer(study)
        print('Shared preparation already complete', flush=True); return
    shared = study/'shared'; shared.mkdir(parents=True, exist_ok=True)
    truth = Scenario.from_json(scenario_file)
    if not (shared/'history_model.npz').exists():
        started = time.perf_counter()
        log = collect_historical_log(truth, FIT, logging_seed=91731)
        log.save(shared/'fit_quotes.npz')
        model, fit = fit_history_model(log)
        if not fit['choice']['converged']:
            raise ValueError('History choice fit did not converge')
        model.save_checkpoint(shared/'history_model.npz')
        holdout = collect_historical_log(truth, HOLDOUT, logging_seed=91731, start_day=len(FIT))
        holdout.save(shared/'holdout_quotes.npz')
        write(shared/'historical_fit.json', dict(fit=fit, holdout=model.diagnostics(holdout),
            elapsed_seconds=time.perf_counter()-started, fit_seeds=FIT, holdout_seeds=HOLDOUT))
        print('Historical fit completed', flush=True)
    model = HistoricalModel.load_checkpoint(shared/'history_model.npz'); fitted = model.scenario
    price_dir = shared/'price_search'
    if not (shared/'pricing_checkpoint.npz').exists():
        if (price_dir/'summary.json').exists():
            pricer = LearnedPricingPolicy.load_checkpoint(price_dir/'checkpoint.npz')
        else:
            price_config = PriceSearchConfig(seed=7, generations=12, population=24, elite_count=6,
                episodes_per_generation=16, training_seed_base=3_300_000_000,
                planning_seed=7, action_seed=ACTION_SEED)
            pricer, _ = fit_price_policy(fitted, price_config, warm_factory, price_dir)
        # The CEM incumbent includes fixed anchors. Use this one frozen function
        # across all dispatch methods; do not optimize prices for a contender.
        pricer.save_checkpoint(shared/'pricing_checkpoint.npz')
    pricing = LearnedPricingPolicy.load_checkpoint(shared/'pricing_checkpoint.npz')
    all_test = set(sum((list(x) for x in TEST.values()), []))
    training = {str(s): [4_600_000_000+SEEDS.index(s)*100_000+2*i for i in range(episodes)] for s in SEEDS}
    if any(all_test & set(x) for x in [FIT, HOLDOUT, VALIDATION, *training.values()]):
        raise ValueError('Training/development/test bank overlap')
    p = dict(schema=SCHEMA, scenario_file=str(Path(scenario_file).resolve()), scenario_sha256=digest(scenario_file),
        scenario=asdict(truth), seeds=SEEDS, methods=METHODS, regimes=REGIMES,
        training_episodes=episodes, training_seeds=training, validation_seeds=VALIDATION, test_seeds=TEST,
        timing_seeds=list(TEST['baseline'][:4]), rollout_config=asdict(RolloutADPConfig()),
        model_sha256=digest(shared/'history_model.npz'), pricing_sha256=digest(shared/'pricing_checkpoint.npz'),
        pricing_policy_sha256=pricing.provenance()['policy_sha256'],
        pricing_scope='One shared reconstructed warm CEM actor; no dispatch-specific pricing refinement.',
        original_artifacts_available=False, test_days='Exact banks from historical study; not a newly collected unseen study.',
        information='Training/validation use fitted model only; private truth only in history collection and final evaluation.',
        selection='Each method/seed uses maximum fitted-model validation profit; exact ties keep earlier snapshot.',
        comparison='Same pricing function and candidate generator, not forced identical realized prices or plans across diverging states.',
        decision_rule='Profit-first: lower paired 95% CI > 0 and faster supports replacement; upper < 0 shows profit cost; crossing 0 is inconclusive. No post-hoc noninferiority margin.',
        uncertainty='Average three policy seeds within each test day, then paired day bootstrap; conditional on shared history and trained checkpoints.',
        source_sha256=source_hashes())
    write(study/'protocol.json', p)
    print('Shared protocol frozen', flush=True)


def verify_shared(study):
    p = read(study/'protocol.json')
    if digest(study/'shared/history_model.npz') != p['model_sha256'] or digest(study/'shared/pricing_checkpoint.npz') != p['pricing_sha256']:
        raise ValueError('Shared frozen checkpoint changed')
    if source_hashes() != p['source_sha256']:
        raise ValueError('Study source changed after protocol freeze; use a new study')
    return p


def train_job(job):
    from charging_env.rl.candidate_training import CandidateTrainingConfig, fit_candidate_value
    study, method, seed = Path(job[0]), job[1], job[2]
    p = verify_shared(study); scenario, pricer = model_and_pricer(study)
    directory = study/'training'/method/f'seed_{seed}'
    if (directory/'summary.json').exists():
        return method, seed, read(directory/'summary.json')
    config = CandidateTrainingConfig(seed=seed, episodes=p['training_episodes'],
        training_seed_base=p['training_seeds'][str(seed)][0], eval_interval=128)
    _, summary = fit_candidate_value(scenario, pricer, method, config, directory,
        p['validation_seeds'], RolloutADPConfig(seed=seed))
    return method, seed, summary


def pool_jobs(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print('Completed', futures[future], flush=True)
    return results


def train(study, workers):
    verify_shared(study)
    if (study/'training_seal.json').exists():
        verify_selection(study)
        print('All selected training checkpoints already sealed', flush=True)
        return
    jobs = [(str(study), method, seed) for method in ('q', 'hybrid') for seed in SEEDS]
    started = time.perf_counter(); pool_jobs(train_job, jobs, workers)
    checkpoints = {f'{method}/{seed}': digest(study/'training'/method/f'seed_{seed}'/'best_checkpoint.npz')
                   for method in ('q', 'hybrid') for seed in SEEDS}
    write(study/'training_seal.json', dict(checkpoints=checkpoints, parallel_wall_seconds=time.perf_counter()-started,
        source_sha256=source_hashes(), test_data_accessed=False))


def verify_selection(study):
    seal = read(study/'training_seal.json')
    if seal['source_sha256'] != source_hashes():
        raise ValueError('Selected-checkpoint source identity changed')
    for key, expected in seal['checkpoints'].items():
        family, train_seed = key.split('/')
        if digest(study/'training'/family/f'seed_{train_seed}'/'best_checkpoint.npz') != expected:
            raise ValueError('Validation-selected checkpoint changed before testing')
    return digest(study/'training_seal.json')


def verify_result(result, protocol, seal_sha, method, seed, regime):
    expected = dict(selection_seal_sha256=seal_sha, source_sha256=protocol['source_sha256'],
        historical_model_sha256=protocol['model_sha256'], pricing_checkpoint_sha256=protocol['pricing_sha256'],
        pricing_policy_sha256=protocol['pricing_policy_sha256'], policy_seed=seed, regime=regime, label=method)
    if any(result.get(k) != v for k,v in expected.items()):
        raise ValueError('Cached evaluation identity differs from frozen study')


def controller(study, method, seed, scenario, pricer):
    if method == 'rollout':
        return PricedRolloutADPPolicy(scenario, pricer, RolloutADPConfig(seed=seed))
    from charging_env.rl.candidate_value import CandidateValuePolicy
    return CandidateValuePolicy.load_checkpoint(study/'training'/method/f'seed_{seed}'/'best_checkpoint.npz', pricer)


def evaluation_job(job):
    # Each process has one Torch/BLAS thread; timing here is explicitly excluded
    # from final latency claims because workers compete for CPU resources.
    import torch
    torch.set_num_threads(1)
    study, method, seed, regime, days, output = Path(job[0]), *job[1:]
    output = Path(output)
    p = verify_shared(study)
    seal_sha = verify_selection(study)
    if output.exists():
        result = read(output)
        verify_result(result, p, seal_sha, method, seed, regime)
        if result['evaluation_seeds'] != list(days): raise ValueError('Cached test days differ')
        return str(output)
    scenario, pricer = model_and_pricer(study)
    truth = shifted_scenarios(Scenario(**_tuples(p['scenario'])))[regime]
    holder = []
    def factory(model):
        policy = controller(study, method, seed, model, pricer); holder.append(policy); return policy
    result = evaluate_history_policy(truth, scenario, days, factory, action_seed=ACTION_SEED, label=method)
    result['policy_seed'] = seed
    result['regime'] = regime
    result['selection_seal_sha256'] = seal_sha
    result['source_sha256'] = p['source_sha256']
    result['historical_model_sha256'] = p['model_sha256']
    result['pricing_checkpoint_sha256'] = p['pricing_sha256']
    result['controller_statistics'] = deepcopy(holder[0].statistics)
    result['pricing_policy_sha256'] = pricer.provenance()['policy_sha256']
    result['parallel_timing_not_for_comparison'] = True
    if result['invalid_actions']: raise RuntimeError('Invalid test actions')
    write(output, result)
    return str(output)


def _tuples(value):
    if isinstance(value, dict): return {k: _tuples(v) for k, v in value.items()}
    if isinstance(value, list): return tuple(_tuples(v) for v in value)
    return value


def evaluate(study, workers, chunk_size):
    p = verify_shared(study)
    seal_sha = verify_selection(study)
    if not (study/'training_seal.json').exists(): raise ValueError('Train and seal all six policies first')
    jobs = []
    for method in METHODS:
        for seed in SEEDS:
            for regime in REGIMES:
                bank = p['test_seeds'][regime]
                for start in range(0, len(bank), chunk_size):
                    output = study/'evaluation'/regime/method/f'seed_{seed}'/f'days_{start:03d}.json'
                    jobs.append((str(study), method, seed, regime, bank[start:start+chunk_size], str(output)))
    started = time.perf_counter(); pool_jobs(evaluation_job, jobs, workers)
    write(study/'evaluation_complete.json', dict(jobs=len(jobs), wall_seconds=time.perf_counter()-started,
        chunk_size=chunk_size, workers=workers, source_sha256=source_hashes()))


def timing(study):
    import torch
    torch.set_num_threads(1)
    p = verify_shared(study)
    # Serial, interleaved methods, same four predeclared baseline days.
    scenario, pricer = model_and_pricer(study)
    truth = Scenario(**_tuples(p['scenario']))
    rows = []
    for seed in SEEDS:
        for i, day in enumerate(p['timing_seeds']):
            order = METHODS[i % 3:]+METHODS[:i % 3]
            for method in order:
                policy = controller(study, method, seed, scenario, pricer)
                result = evaluate_history_policy(truth, scenario, [day], lambda model: policy,
                    action_seed=ACTION_SEED, label=method)
                rows.append(dict(method=method, policy_seed=seed, day=day,
                    dispatch_ms=result['mean_policy_decision_ms']['dispatch'],
                    pricing_ms=result['mean_policy_decision_ms']['pricing'],
                    future_model_queries=policy.statistics['future_model_queries']))
                print('Serial timing', seed, day, method, rows[-1]['dispatch_ms'], flush=True)
                write(study/'timing_progress.json', rows)
    write(study/'timing.json', dict(scope='Serial actual-trajectory timings on 4 predeclared baseline days x 3 seeds; parallel work finished.',
        selection_seal_sha256=verify_selection(study),
        rows=rows, mean_dispatch_ms={m:float(np.mean([r['dispatch_ms'] for r in rows if r['method']==m])) for m in METHODS}))


def report(study):
    p = verify_shared(study)
    seal_sha = verify_selection(study)
    if not (study/'evaluation_complete.json').exists(): raise ValueError('Complete evaluation first')
    statistics, comparisons = {}, {}
    for regime in REGIMES:
        statistics[regime], groups = {}, {}
        for method in METHODS:
            per_seed, grouped, queries, invalid = {}, {}, 0, 0
            for seed in SEEDS:
                files = sorted((study/'evaluation'/regime/method/f'seed_{seed}').glob('days_*.json'))
                episodes = []
                for path in files:
                    result = read(path)
                    verify_result(result, p, seal_sha, method, seed, regime)
                    episodes.extend(result['episodes'])
                    if result['pricing_policy_sha256'] != p['pricing_policy_sha256']: raise ValueError('Different test pricing policy')
                    queries += result['controller_statistics']['future_model_queries']; invalid += result['invalid_actions']
                days = [e['seed'] for e in episodes]
                if len(days) != len(set(days)) or set(days) != set(p['test_seeds'][regime]):
                    raise ValueError('Missing/duplicated/unexpected test days')
                per_seed[str(seed)] = float(np.mean([e['discounted_profit'] for e in episodes]))
                for episode in episodes: grouped.setdefault(episode['seed'], []).append(episode['discounted_profit'])
            averaged = [dict(seed=day, discounted_profit=float(np.mean(grouped[day]))) for day in p['test_seeds'][regime]]
            if any(len(x)!=len(SEEDS) for x in grouped.values()): raise ValueError('Incomplete policy seed ensemble')
            groups[method] = dict(label=method, episodes=averaged)
            statistics[regime][method] = dict(mean_profit=float(np.mean([x['discounted_profit'] for x in averaged])),
                per_seed_mean_profit=per_seed, invalid_actions=invalid, future_model_queries=queries)
        comparisons[regime] = {m:paired_comparison(groups[m], groups['rollout']) for m in ('q','hybrid')}
    train_costs = {m:{str(s):read(study/'training'/m/f'seed_{s}'/'summary.json') for s in SEEDS} for m in ('q','hybrid')}
    latency = read(study/'timing.json') if (study/'timing.json').exists() else None
    if latency and latency.get('selection_seal_sha256') != seal_sha:
        raise ValueError('Latency record has a different selected-checkpoint identity')
    verdicts = {}
    for method in ('q','hybrid'):
        ci = comparisons['baseline'][method]['paired_bootstrap_confidence_interval_95']
        verdicts[method] = ('profit_improved' if ci[0]>0 else 'profit_cost_detected' if ci[1]<0 else 'profit_inconclusive')
    result = dict(schema=SCHEMA, statistics=statistics, comparisons=comparisons, training_costs=train_costs,
        timing=latency, baseline_profit_verdict=verdicts,
        scope=p['pricing_scope'], uncertainty=p['uncertainty'], source_sha256=source_hashes())
    write(study/'comparison.json', result)
    lines = ['# 候选 Q-learning 与混合 ADP 对照结果', '',
        '三种调度器共用同一冻结价格函数、历史拟合模型、候选生成规则和测试日。使用原历史实验的测试银行；本轮不使用测试结果挑模型。', '',
        '原目录缺少模型检查点，本轮按原 120 天记录流程重建模型，并以原完整暖定价预算重建一套共享价格；这不是 README 各种已选定价模型的原样重跑。', '',
        '| 测试条件 | Rollout | 候选 Q | 混合 ADP |', '|---|---:|---:|---:|']
    labels = dict(baseline='原需求',volume_up='到达增加40%',spatiotemporal='时空转移')
    for regime in REGIMES:
        lines.append('| '+labels[regime]+' | '+' | '.join(f"{statistics[regime][m]['mean_profit']:.2f}" for m in METHODS)+' |')
    lines += ['', '利润单位为美元/日，按原单期折扣累计；先对同一天的三个策略种子平均，再以128天做5000次配对bootstrap。区间条件于本次历史拟合和训练模型，不覆盖重采历史或重训的不确定性。', '',
        '| 相对 Rollout 的利润变化 | 原需求（95%区间） | 增量需求（95%区间） | 时空转移（95%区间） |', '|---|---:|---:|---:|']
    for method,label in [('q','候选 Q'),('hybrid','混合 ADP')]:
        cells=[]
        for regime in REGIMES:
            c=comparisons[regime][method];lo,hi=c['paired_bootstrap_confidence_interval_95'];cells.append(f"{c['mean_difference']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    if latency:
        lines += ['', '| 方法 | 串行平均调度耗时（ms） | 相对 Rollout 加速 |', '|---|---:|---:|']
        for method in METHODS:
            ms=latency['mean_dispatch_ms'][method];lines.append(f"| {method} | {ms:.2f} | {latency['mean_dispatch_ms']['rollout']/ms:.2f}× |")
        lines += ['', '耗时在并行训练和收益评估完成后，按预先固定的4个原需求测试日、3个种子串行交错测量。不同调度导致后续车队状态不同，因此这是实际轨迹耗时；不是固定状态微基准。']
    lines += ['', '| 方法 | 每种子训练天数 | 三种子训练环境期数 | 三种子训练CPU任务累计秒数 |', '|---|---:|---:|---:|']
    for method in ('q','hybrid'):
        costs=list(train_costs[method].values());lines.append(f"| {method} | {p['training_episodes']} | {sum(x['environment_periods'] for x in costs)} | {sum(x['training_seconds'] for x in costs):.1f} |")
    lines += ['', '训练与验证时间均保存在 comparison.json；并行任务累计耗时不等于整个实验墙钟时间。共享历史拟合和定价训练成本三种方法相同。新方法在线未来模型查询为零，但仍要生成候选和调用精确当前物理模型。', '',
        '判定预先采用利润优先：配对区间下界大于0且更快，支持替换；上界小于0，存在利润代价；跨0只表示证据不足，不能声称利润无损。没有事后选择非劣容忍界。', '',
        '所有结论限于此次固定训练预算和共享定价。Q和混合ADP均在同一候选受限动作空间训练，不能据此宣称全联合动作最优或算法已收敛。', '',
        '可复核记录：protocol.json、shared/、training/、training_seal.json、evaluation/、comparison.json、timing.json。']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(statistics=statistics, comparisons=comparisons, latency=latency['mean_dispatch_ms'] if latency else None),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','train','evaluate','timing','report','all'),required=True)
    parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--scenario',type=Path,default=ROOT/'configs/synthetic_25_zones.json')
    parser.add_argument('--episodes',type=int,default=512)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--chunk-size',type=int,default=16)
    args=parser.parse_args();study=args.study_dir.resolve()
    if args.episodes<1 or args.workers<1 or args.chunk_size<1: parser.error('Budgets must be positive')
    for phase in (('prepare','train','evaluate','timing','report') if args.phase=='all' else (args.phase,)):
        if phase=='prepare': prepare(study,args.scenario,args.episodes)
        elif phase=='train': train(study,args.workers)
        elif phase=='evaluate': evaluate(study,args.workers,args.chunk_size)
        elif phase=='timing': timing(study)
        else: report(study)


if __name__=='__main__':
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    main()
