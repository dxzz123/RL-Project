"""Fit candidate NN ADP to frozen, explicitly synthetic dispatch logs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / 'outputs/candidate_adp_study'
DEFAULT_OUTPUT = PROJECT / 'outputs/logged_nn_adp_study'
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'experiments')]

import numpy as np
import torch

from charging_env.benchmarks.compact import inventory_shadow
from charging_env.benchmarks.evaluation import paired_comparison
from charging_env.benchmarks.policies import dispatch_outcome, service_first_dispatch
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from charging_env.config import Scenario
from charging_env.rl.candidate_value import CandidateValuePolicy, ValueNetwork, encode_afterstate
from charging_env.rl.history_control import (
    evaluate_history_policy, history_dispatch_observation, history_pricing_observation,
)
from charging_env.rl.priced_rollout import _price_levels
from charging_env.rl.rollout_adp import RolloutADPConfig
from charging_env.simulator import ChargingSimulator
from run_candidate_adp_study import (
    ACTION_SEED, _tuples, digest, model_and_pricer, read, shifted_scenarios, write,
)

FAMILIES = ('service_first', 'constant_inventory', 'time_inventory')
GROUPS = (*FAMILIES, 'mixed')
REGIMES = ('baseline', 'volume_up', 'spatiotemporal')
SEEDS = (7, 19, 37)
TRAIN_DAYS = 768
VALIDATION_DAYS = 96
SCALE = 100.
EPOCHS = 80
BATCH_SIZE = 256
LR = 3e-4


def single_thread():
    torch.set_num_threads(1)


def run_jobs(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=single_thread) as executor:
        futures = [executor.submit(function, job) for job in jobs]
        return [future.result() for future in as_completed(futures)]


def settings():
    previous = read(REFERENCE / 'protocol.json')
    fitted, _ = model_and_pricer(REFERENCE)
    config = RolloutADPConfig(**_tuples(previous['rollout_config']))
    logging = {
        'service_first': dict(family='service_first', recharge_threshold=config.base_recharge_threshold),
        'constant_inventory': dict(family='constant_inventory', shadow_price=config.inventory_early_shadow),
        'time_inventory': dict(family='time_inventory', early_shadow_price=config.inventory_early_shadow,
                               late_shadow_price=config.inventory_late_shadow,
                               switch_period=fitted.horizon // 2),
    }
    train = [5_120_000_001 + 2*i for i in range(TRAIN_DAYS)]
    validation = [5_120_100_001 + 2*i for i in range(VALIDATION_DAYS)]
    test = {r: previous['test_seeds'][r] for r in REGIMES}
    seen = {d for days in test.values() for d in days}
    previous_training_days = {d for days in previous['training_seeds'].values() for d in days}
    if (seen | previous_training_days) & set(train + validation) or set(train) & set(validation):
        raise ValueError('Logging day banks overlap')
    for file, key in (('history_model.npz', 'model_sha256'), ('pricing_checkpoint.npz', 'pricing_sha256')):
        if digest(REFERENCE / 'shared' / file) != previous[key]:
            raise ValueError(f'Shared artifact changed: {file}')
    return dict(
        data_origin='SYNTHETIC logging trajectories generated once in original baseline truth scenario',
        training='Fixed log only; actual logged post-dispatch observations and realized Monte Carlo returns',
        selection='Minimum MSE on separate logged validation days within each group; earlier epoch breaks ties',
        mixing='Same chronological days and same count per group; mixed takes one policy per day in rotation',
        logging_policies=logging, train_days=train, validation_days=validation, test_days=test,
        gamma=fitted.discount_per_period, reward_scale=SCALE, policy_seeds=SEEDS, epochs=EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LR, rollout_config=asdict(config),
        pricing_sha256=previous['pricing_sha256'], model_sha256=previous['model_sha256'],
        study_code_sha256=digest(Path(__file__)), reference_protocol_sha256=digest(REFERENCE / 'protocol.json'),
    )


class LoggingPolicy:
    def __init__(self, model, pricer, spec, config):
        self.scenario, self.pricer, self.spec, self.config = model, pricer, spec, config

    def prices(self, obs, rng=None):
        return _price_levels(obs, self.scenario, self.pricer)[0]

    def dispatch(self, obs, rng=None):
        if self.spec['family'] == 'service_first':
            return service_first_dispatch(obs, self.scenario, self.spec['recharge_threshold'])
        shadow = inventory_shadow(obs, self.scenario, self.spec)
        return coordinate_inventory_dispatch(obs, self.scenario, shadow, sweeps=self.config.planner_sweeps,
                                             energy_fractions=self.config.energy_fractions)


def collect_block(job):
    family, days, spec, raw_config = job
    truth = Scenario(**_tuples(read(REFERENCE / 'protocol.json')['scenario'])).validate()
    model, pricer = model_and_pricer(REFERENCE)
    policy = LoggingPolicy(model, pricer, spec, RolloutADPConfig(**_tuples(raw_config)))
    simulator = ChargingSimulator(truth)
    episodes = []
    for day in days:
        before, _ = simulator.reset(seed=day)
        fields = {name: [] for name in ('locations', 'energy_kwh', 'modes', 'target_zone', 'action_kwh',
                                         'price_levels', 'profit', 'features')}
        for period in range(model.horizon):
            levels = policy.prices(history_pricing_observation(before, model))
            observation, quote_profit, done, truncated, _ = simulator.quote_price_levels(levels)
            if quote_profit or done or truncated:
                raise RuntimeError('Unexpected quote-stage transition')
            current = history_dispatch_observation(observation, model)
            actions = policy.dispatch(current)
            before, profit, done, truncated, info = simulator.dispatch(actions)
            if truncated or done != (period + 1 == model.horizon) or info['invalid_actions']:
                raise RuntimeError('Invalid logged dispatch')
            locations, energy = before['locations'], before['energy_kwh']
            if day == days[0]:
                exact = dispatch_outcome(current, model, actions)
                np.testing.assert_array_equal(exact['next_locations'], locations)
                np.testing.assert_allclose(exact['next_energy_kwh'], energy, rtol=0., atol=1e-8)
                np.testing.assert_allclose(exact['reward'], profit, rtol=0., atol=1e-8)
            fields['locations'].append(locations)
            fields['energy_kwh'].append(energy)
            fields['modes'].append([int(a.mode) for a in actions])
            fields['target_zone'].append([int(a.target_zone) for a in actions])
            fields['action_kwh'].append([float(a.energy_kwh) for a in actions])
            fields['price_levels'].append(levels)
            fields['profit'].append(profit)
            fields['features'].append(encode_afterstate(model, period + 1, locations, energy)[0])
        episodes.append(fields)
    return family, list(days), {k: np.asarray([episode[k] for episode in episodes]) for k in episodes[0]}


def collect(output, workers):
    output.mkdir(parents=True, exist_ok=True)
    protocol = settings()
    manifest_path = output / 'log_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    write(output / 'protocol.json', protocol)
    manifest = {}
    for split in ('train', 'validation'):
        days = protocol[f'{split}_days']
        jobs = [(family, days[start:start+48], protocol['logging_policies'][family], protocol['rollout_config'])
                for family in FAMILIES for start in range(0, len(days), 48)]
        results = run_jobs(collect_block, jobs, workers)
        for family in FAMILIES:
            ordered = sorted((part_days[0], part_days, part) for name, part_days, part in results if name == family)
            log_days = [day for _, block, _ in ordered for day in block]
            if days != log_days:
                raise RuntimeError('Incomplete log coverage')
            arrays = {key: np.concatenate([part[key] for _, _, part in ordered]) for key in ordered[0][2]}
            path = output / f'{split}_{family}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, day=np.asarray(days), **arrays)
            manifest[path.name] = dict(sha256=digest(path), days=len(days), periods=arrays['profit'].shape[1])
        print(f'Collected {len(days)} {split} days for each of {len(FAMILIES)} synthetic policies', flush=True)
    write(manifest_path, manifest)


def continuation_labels(rewards, gamma):
    labels = np.zeros_like(rewards, dtype=np.float64)
    for period in reversed(range(rewards.shape[1] - 1)):
        labels[:, period] = rewards[:, period + 1] + gamma * labels[:, period + 1]
    return labels


def training_data(output, protocol, manifest, split, group):
    days = protocol[f'{split}_days']
    families = FAMILIES if group == 'mixed' else (group,)
    features, rewards = None, None
    for index, family in enumerate(families):
        path = output / f'{split}_{family}.npz'
        if digest(path) != manifest[path.name]['sha256']:
            raise ValueError(f'Log changed after collection: {path}')
        with np.load(path, allow_pickle=False) as log:
            np.testing.assert_array_equal(log['day'], days)
            mask = np.arange(len(days)) % len(families) == index
            if features is None:
                features, rewards = np.empty_like(log['features']), np.empty_like(log['profit'])
            features[mask], rewards[mask] = log['features'][mask], log['profit'][mask]
    labels = continuation_labels(rewards, protocol['gamma'])
    # The deployed terminal scorer is exactly current profit; exclude this known zero from fitting.
    return features[:, :-1].reshape(-1, features.shape[-1]), labels[:, :-1].ravel()


def fit_job(job):
    output, group, seed = job
    protocol, manifest = read(output / 'protocol.json'), read(output / 'log_manifest.json')
    x, y = training_data(output, protocol, manifest, 'train', group)
    val_x, val_y = training_data(output, protocol, manifest, 'validation', group)
    scale = protocol['reward_scale']
    train_x, train_y = torch.from_numpy(x), torch.from_numpy((y / scale).astype(np.float32))
    valid_x, valid_y = torch.from_numpy(val_x), torch.from_numpy((val_y / scale).astype(np.float32))
    torch.manual_seed(seed)
    network = ValueNetwork(x.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=protocol['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    best_mse, best = float('inf'), None
    history = []
    for epoch in range(1, protocol['epochs'] + 1):
        network.train()
        order = torch.randperm(len(x), generator=generator)
        for rows in order.split(protocol['batch_size']):
            loss = torch.nn.functional.mse_loss(network(train_x[rows]), train_y[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            residual = (network(valid_x) - valid_y) * scale
            mse = float(residual.square().mean())
            mae = float(residual.abs().mean())
        if not np.isfinite(mse):
            raise RuntimeError('Nonfinite logged validation loss')
        history.append(dict(epoch=epoch, mse_usd2=mse, mae_usd=mae))
        if mse < best_mse:
            best_mse = mse
            best = {key: value.detach().numpy().copy() for key, value in network.state_dict().items()}
    folder = output / 'training' / group / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'weights.npz').open('xb') as stream:
        np.savez_compressed(stream, **best)
    time_mean = y.reshape(len(protocol['train_days']), -1).mean(axis=0)
    time_rmse = np.sqrt(np.mean((val_y.reshape(-1, len(time_mean)) - time_mean)**2))
    selected = min(history, key=lambda item: item['mse_usd2'])
    write(folder / 'summary.json', dict(selected=selected, train_rows=len(y), validation_rows=len(val_y),
                                      validation_time_only_rmse_usd=float(time_rmse), history=history,
                                      weights_sha256=digest(folder / 'weights.npz')))
    print(f'Fit {group} seed {seed}, selected epoch {selected["epoch"]}', flush=True)


def fit(output, workers):
    if not (output / 'log_manifest.json').is_file():
        raise ValueError('Freeze logs before fitting')
    run_jobs(fit_job, [(output, group, seed) for group in GROUPS for seed in SEEDS], workers)
    checkpoints = {f'{group}/{seed}': read(output / 'training' / group / str(seed) / 'summary.json')['weights_sha256']
                   for group in GROUPS for seed in SEEDS}
    write(output / 'training_seal.json', dict(log_manifest_sha256=digest(output / 'log_manifest.json'),
                                             protocol_sha256=digest(output / 'protocol.json'), weights=checkpoints))


def make_eval_controller(output, method, seed, model, pricer, protocol, seal):
    config = RolloutADPConfig(**_tuples(protocol['rollout_config']))
    if method in FAMILIES:
        return LoggingPolicy(model, pricer, protocol['logging_policies'][method], config)
    from charging_env.rl.candidate_value import afterstate_feature_size
    network = ValueNetwork(afterstate_feature_size(model))
    if method == 'current_only':
        with torch.no_grad():
            for param in network.parameters():
                param.zero_()
    else:
        group = method.removeprefix('nn_')
        path = output / 'training' / group / str(seed) / 'weights.npz'
        if digest(path) != seal['weights'][f'{group}/{seed}']:
            raise ValueError('Selected checkpoint changed')
        with np.load(path, allow_pickle=False) as archive:
            network.load_state_dict({key: torch.from_numpy(archive[key]) for key in archive.files})
    return CandidateValuePolicy(model, pricer, network, 'hybrid', config, protocol['reward_scale'])


def evaluate_job(job):
    output, method, seed, regime = job
    path = output / 'evaluation' / regime / f'{method}_{seed}.json'
    if path.is_file():
        return
    protocol, seal = read(output / 'protocol.json'), read(output / 'training_seal.json')
    if (digest(output / 'protocol.json') != seal['protocol_sha256']
            or digest(output / 'log_manifest.json') != seal['log_manifest_sha256']):
        raise ValueError('Evaluation requires the sealed training run')
    model, pricer = model_and_pricer(REFERENCE)
    truth = shifted_scenarios(Scenario(**_tuples(read(REFERENCE / 'protocol.json')['scenario'])))[regime]
    def factory(fitted):
        return make_eval_controller(output, method, seed, fitted, pricer, protocol, seal)
    result = evaluate_history_policy(truth, model, protocol['test_days'][regime], factory,
                                     action_seed=ACTION_SEED, label=method)
    boundary = result.pop('history_boundary')
    if result['invalid_actions'] or boundary['model_queries']:
        raise RuntimeError('Logged-value or logging policy violated evaluation contract')
    compact = [dict(seed=row['seed'], discounted_profit=row['discounted_profit'],
                    served_requests=row['served_requests'], total_requests=row['total_requests'])
               for row in result['episodes']]
    write(path, dict(label=method, seed=seed, episodes=compact, invalid_actions=result['invalid_actions'],
                     future_model_queries=boundary['model_queries'],
                     training_seal_sha256=digest(output / 'training_seal.json')))
    print(f'Evaluated {regime} / {method} / {seed}: {result["mean_discounted_profit"]:.2f}', flush=True)


def evaluate(output, workers):
    jobs = [(output, method, seed, regime) for regime in REGIMES
            for method in ('current_only', *FAMILIES, *(f'nn_{g}' for g in GROUPS))
            for seed in (SEEDS if method.startswith('nn_') else (0,))]
    run_jobs(evaluate_job, jobs, workers)


def daily_group(label, results, days):
    by_run = []
    for rows in results:
        by_day = {row['seed']: row['discounted_profit'] for row in rows}
        if set(by_day) != set(days) or len(rows) != len(days):
            raise ValueError(f'Wrong test days for {label}')
        by_run.append([by_day[day] for day in days])
    return dict(label=label, episodes=[dict(seed=day, discounted_profit=float(value))
                                      for day, value in zip(days, np.mean(by_run, axis=0))],
                seed_means=np.mean(by_run, axis=1).tolist())


def reference_group(method, regime, days, protocol):
    label = 'reference_' + method
    runs = []
    previous = read(REFERENCE / 'protocol.json')
    for seed in SEEDS:
        folder = REFERENCE / 'evaluation' / regime / method / f'seed_{seed}'
        rows = []
        for path in sorted(folder.glob('days_*.json')):
            result = read(path)
            if (result['historical_model_sha256'] != protocol['model_sha256']
                    or result['pricing_checkpoint_sha256'] != protocol['pricing_sha256']
                    or result['pricing_policy_sha256'] != previous['pricing_policy_sha256']
                    or result['invalid_actions']):
                raise ValueError('Mismatched reference result')
            rows.extend(result['episodes'])
        runs.append(rows)
    return daily_group(label, runs, days)


def summarize(output):
    protocol = read(output / 'protocol.json')
    seal_hash = digest(output / 'training_seal.json')
    statistics, comparisons = {}, {}
    for regime in REGIMES:
        days = protocol['test_days'][regime]
        groups = {f'reference_{m}': reference_group(m, regime, days, protocol) for m in ('rollout', 'hybrid')}
        for method in ('current_only', *FAMILIES, *(f'nn_{g}' for g in GROUPS)):
            runs = [read(output / 'evaluation' / regime / f'{method}_{seed}.json')
                    for seed in (SEEDS if method.startswith('nn_') else (0,))]
            if any(run['training_seal_sha256'] != seal_hash or run['invalid_actions']
                   or run['future_model_queries'] for run in runs):
                raise ValueError('Unverified test results')
            groups[method] = daily_group(method, [run['episodes'] for run in runs], days)
        statistics[regime] = {name: dict(mean=float(np.mean([row['discounted_profit'] for row in data['episodes']])),
                                         seed_means=data['seed_means']) for name, data in groups.items()}
        pairs = [(f'nn_{g}', base) for g in GROUPS for base in ('reference_rollout', 'reference_hybrid', 'current_only')]
        pairs += [(f'nn_{family}', family) for family in FAMILIES]
        pairs += [('nn_mixed', f'nn_{family}') for family in FAMILIES]
        comparisons[regime] = {f'{a}_vs_{b}': paired_comparison(groups[a], groups[b]) for a, b in pairs}
    write(output / 'comparison.json', dict(statistics=statistics, comparisons=comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'fit', 'evaluate', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    single_thread()
    if args.phase == 'summarize':
        summarize(args.output)
    else:
        {'collect': collect, 'fit': fit, 'evaluate': evaluate}[args.phase](args.output, args.workers)


if __name__ == '__main__':
    main()
