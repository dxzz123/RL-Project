"""Reproducible, simulator-only policy benchmarks with paired uncertainty.

Callbacks receive only the observation available at their decision stage.  They
may close over a validated Scenario, but are never given a simulator or its RNG.
This module deliberately does not import PyTorch, so heuristic benchmarks work
with the base environment installation.
"""
from collections.abc import Callable, Iterable
import math
import time

import numpy as np

from ..config import Scenario
from ..simulator import ChargingSimulator, Mode


def _seed_list(seeds: Iterable[int]) -> list[int]:
    result = list(seeds)
    if not result:
        raise ValueError('At least one evaluation seed is required')
    if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))
           or s < 0 for s in result):
        raise ValueError('Evaluation seeds must be nonnegative integers')
    result = [int(s) for s in result]
    if len(set(result)) != len(result):
        raise ValueError('Evaluation seeds must be unique')
    return result


def _mean_defined(values):
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def episode_summary(seed, infos, rewards, gamma, *, n_mcv):
    """Summarize one complete episode; fulfillment is undefined with no selection.

    Costs in reward_terms remain positive expenses, as in the simulator. Outside
    requests are excluded from selected-service fulfillment but included in the
    total-arrival served share. Energy quantities are sums over the episode,
    except initial/final inventory. Utilization denominators include all periods.
    """
    if not infos or len(infos) != len(rewards):
        raise ValueError('A nonempty episode needs one info record per reward')
    total = sum(int(i['total_requests']) for i in infos)
    outside = sum(int(i['outside_requests']) for i in infos)
    selected = total - outside
    mobile_selected = sum(int(np.count_nonzero(i['request_choices'] == 0)) for i in infos)
    mobile_served = sum(int(np.sum(i['served_mobile'])) for i in infos)
    fixed_served = sum(int(np.sum(i['served_fixed'])) for i in infos)
    served = mobile_served + fixed_served
    actions = {mode.name: sum(int(np.count_nonzero(np.asarray(i['executed_modes']) == mode))
                             for i in infos) for mode in Mode}
    ledger_keys = ('mobile_delivered_kwh', 'fixed_delivered_kwh', 'fixed_grid_kwh',
                   'travel_kwh', 'charge_grid_kwh', 'export_grid_kwh')
    result = {
        'seed': int(seed), 'periods': len(infos),
        'profit': float(sum(rewards)),
        'discounted_profit': float(sum(gamma**t * r for t, r in enumerate(rewards))),
        'total_requests': total, 'outside_requests': outside,
        'selected_requests': selected, 'mobile_selected_requests': mobile_selected,
        'fixed_selected_requests': selected - mobile_selected,
        'served_requests': served, 'mobile_served_requests': mobile_served,
        'fixed_served_requests': fixed_served,
        'outside_option_share': outside / total if total else 0.,
        'served_request_share': served / total if total else 0.,
        'selected_service_fulfillment': served / selected if selected else None,
        'mobile_service_fulfillment': mobile_served / mobile_selected if mobile_selected else None,
        'invalid_actions': sum(len(i['invalid_actions']) for i in infos),
        'reward_terms': {key: float(sum(i['reward_terms'][key] for i in infos))
                         for key in infos[0]['reward_terms']},
        'discounted_reward_terms': {
            key: float(sum(gamma**t * i['reward_terms'][key] for t, i in enumerate(infos)))
            for key in infos[0]['reward_terms']},
        'action_counts': actions,
        'fleet_active_fraction': 1. - actions['WAIT'] / (n_mcv * len(infos)),
        'fleet_service_action_fraction': actions['SERVE'] / (n_mcv * len(infos)),
        'mobile_service_utilization': float(np.mean([i['metrics']['mcv_service_utilization']
                                                    for i in infos])),
        'fixed_service_utilization': np.mean([i['metrics']['fcs_service_utilization']
                                              for i in infos], axis=0).tolist(),
        'unmet_mobile_kwh': float(sum(i['unmet_mobile_kwh'] for i in infos)),
        'unmet_fixed_kwh': float(sum(i['unmet_fixed_kwh'] for i in infos)),
        'initial_inventory_kwh': float(np.sum(infos[0]['energy_ledger']['before_kwh'])),
        'final_inventory_kwh': float(np.sum(infos[-1]['energy_ledger']['after_kwh'])),
    }
    result.update({key: float(sum(np.sum(i['energy_ledger'][key]) for i in infos))
                   for key in ledger_keys})
    return result


def evaluate_policy(config: Scenario, seeds: Iterable[int], pricing_fn: Callable,
                    dispatch_fn: Callable, action_seed: int = 321, label: str = '',
                    on_episode_start: Callable | None = None):
    """Evaluate a fixed policy on independent, paired environment seeds.

    pricing_fn(obs, rng) returns an integer price-level vector; dispatch_fn(obs,
    rng) returns one DispatchAction per vehicle. Both see copies from the public
    observation API. An episode-local policy RNG is seeded by action_seed + the
    environment seed, so evaluation order does not change action randomness and
    policy draws cannot consume environmental randomness or global NumPy state.
    An optional on_episode_start(env_seed, action_seed) lets an adapter initialize
    its own non-NumPy RNG before any timed decisions; the adapter is responsible
    for restoring any global random state it changes.

    Profit tail metrics refer to *discounted* profit. Normal confidence intervals
    describe environment/action sampling uncertainty conditional on this fixed
    policy, not uncertainty across independently trained policies. Runtime values
    include callback execution only and are descriptive, machine-dependent data.
    """
    config = config.validate()
    seeds = _seed_list(seeds)
    if (isinstance(action_seed, (bool, np.bool_))
            or not isinstance(action_seed, (int, np.integer)) or action_seed < 0):
        raise ValueError('action_seed must be a nonnegative integer')
    episodes = []
    pricing_ms, dispatch_ms = [], []
    for seed in seeds:
        sim = ChargingSimulator(config)
        obs, _ = sim.reset(seed=seed)
        policy_rng = np.random.default_rng(int(action_seed) + seed)
        if on_episode_start is not None:
            on_episode_start(seed, int(action_seed))
        infos, rewards, prices = [], [], []
        episode_pricing_ms, episode_dispatch_ms = [], []
        done = False
        while not done:
            start = time.perf_counter()
            levels = pricing_fn(obs, policy_rng)
            episode_pricing_ms.append((time.perf_counter() - start) * 1000.)
            post, pricing_reward, pricing_done, pricing_truncated, _ = sim.quote_price_levels(levels)
            if pricing_reward != 0. or pricing_done or pricing_truncated:
                raise RuntimeError('Unexpected simulator pricing transition')
            prices.append(float(np.mean(post['mobile_prices'])))
            start = time.perf_counter()
            commands = dispatch_fn(post, policy_rng)
            episode_dispatch_ms.append((time.perf_counter() - start) * 1000.)
            obs, reward, done, truncated, info = sim.dispatch(commands)
            if truncated:
                raise RuntimeError('Unexpected truncated benchmark episode')
            infos.append(info)
            rewards.append(reward)
        row = episode_summary(seed, infos, rewards, config.discount_per_period,
                              n_mcv=config.n_mcv)
        row['mean_mobile_price'] = float(np.mean(prices))
        row['mean_policy_decision_ms'] = {
            'pricing': float(np.mean(episode_pricing_ms)),
            'dispatch': float(np.mean(episode_dispatch_ms)),
        }
        episodes.append(row)
        pricing_ms.extend(episode_pricing_ms)
        dispatch_ms.extend(episode_dispatch_ms)
    values = np.asarray([e['discounted_profit'] for e in episodes], dtype=float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    total = sum(e['total_requests'] for e in episodes)
    selected = sum(e['selected_requests'] for e in episodes)
    served = sum(e['served_requests'] for e in episodes)
    aggregate_fields = (
        'profit', 'total_requests', 'outside_requests', 'selected_requests',
        'mobile_selected_requests', 'fixed_selected_requests', 'served_requests',
        'mobile_served_requests', 'fixed_served_requests', 'outside_option_share',
        'served_request_share', 'selected_service_fulfillment', 'mobile_service_fulfillment',
        'fleet_active_fraction', 'fleet_service_action_fraction', 'mobile_service_utilization',
        'mobile_delivered_kwh', 'fixed_delivered_kwh', 'fixed_grid_kwh', 'travel_kwh',
        'charge_grid_kwh', 'export_grid_kwh', 'unmet_mobile_kwh', 'unmet_fixed_kwh',
        'initial_inventory_kwh', 'final_inventory_kwh',
    )
    result = {
        'label': str(label), 'action_seed': int(action_seed), 'evaluation_seeds': seeds,
        'n_episodes': len(episodes), 'episodes': episodes,
        'mean_discounted_profit': mean,
        'standard_error_discounted_profit': sem,
        'confidence_interval_95': [mean - 1.96 * sem, mean + 1.96 * sem] if sem is not None else None,
        'profit_p10': float(np.quantile(values, .1)),
        'lower_tail_mean_10pct': float(np.sort(values)[:math.ceil(.1 * len(values))].mean()),
        'loss_probability': float(np.mean(values < 0.)),
        'mean_reward_terms': {key: float(np.mean([e['reward_terms'][key] for e in episodes]))
                              for key in episodes[0]['reward_terms']},
        'mean_discounted_reward_terms': {
            key: float(np.mean([e['discounted_reward_terms'][key] for e in episodes]))
            for key in episodes[0]['discounted_reward_terms']},
        'action_counts': {mode.name: sum(e['action_counts'][mode.name] for e in episodes)
                         for mode in Mode},
        'invalid_actions': sum(e['invalid_actions'] for e in episodes),
        'mean_mobile_price': float(np.mean([e['mean_mobile_price'] for e in episodes])),
        'mean_policy_decision_ms': {'pricing': float(np.mean(pricing_ms)),
                                    'dispatch': float(np.mean(dispatch_ms))},
        'pooled_served_request_share': served / total if total else 0.,
        'pooled_outside_option_share': sum(e['outside_requests'] for e in episodes) / total if total else 0.,
        'pooled_selected_service_fulfillment': served / selected if selected else None,
        'episodes_with_selected_requests': sum(e['selected_requests'] > 0 for e in episodes),
        'mean_fixed_service_utilization': np.mean([e['fixed_service_utilization'] for e in episodes],
                                                  axis=0).tolist(),
    }
    result.update({'mean_' + key: _mean_defined([e[key] for e in episodes])
                   for key in aggregate_fields})
    return result


def paired_comparison(candidate, baseline, bootstrap_samples: int = 5000, seed: int = 271828):
    """Paired bootstrap of candidate minus baseline on identical environment seeds.

    Inputs are evaluate_policy results or compatible mappings with `episodes`.
    Reordering is allowed, duplicated/missing/different seeds are not. The CI is
    conditional on the evaluated trained models and does not capture training
    seed uncertainty. For one paired episode the CI is undefined, rather than
    spuriously reporting zero uncertainty. No global random state is modified.
    """
    if (isinstance(bootstrap_samples, (bool, np.bool_))
            or not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 1):
        raise ValueError('bootstrap_samples must be a positive integer')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('bootstrap seed must be a nonnegative integer')
    candidate_seeds = _seed_list(e['seed'] for e in candidate['episodes'])
    baseline_seeds = _seed_list(e['seed'] for e in baseline['episodes'])
    if set(candidate_seeds) != set(baseline_seeds):
        raise ValueError('Paired comparisons require identical environment seed sets')
    baseline_values = {int(e['seed']): float(e['discounted_profit']) for e in baseline['episodes']}
    differences = np.asarray([float(e['discounted_profit']) - baseline_values[int(e['seed'])]
                              for e in candidate['episodes']], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError('Paired episode profits must be finite')
    rng = np.random.default_rng(int(seed))
    interval = None
    if len(differences) > 1:
        # Limit peak memory even for a large evaluation set.
        bootstrap = []
        for offset in range(0, int(bootstrap_samples), 256):
            count = min(256, int(bootstrap_samples) - offset)
            indices = rng.integers(0, len(differences), size=(count, len(differences)))
            bootstrap.extend(differences[indices].mean(axis=1).tolist())
        interval = np.quantile(bootstrap, [.025, .975]).tolist()
    return {
        'candidate': candidate.get('label', ''), 'baseline': baseline.get('label', ''),
        'n_episodes': len(differences), 'mean_difference': float(differences.mean()),
        'paired_bootstrap_confidence_interval_95': interval,
        'bootstrap_samples': int(bootstrap_samples), 'bootstrap_seed': int(seed),
        'candidate_win_fraction': float(np.mean(differences > 0.)),
        'tie_fraction': float(np.mean(differences == 0.)),
        'uncertainty_scope': 'Environment/action sampling, conditional on the fixed evaluated policies; excludes training seed uncertainty.',
    }
