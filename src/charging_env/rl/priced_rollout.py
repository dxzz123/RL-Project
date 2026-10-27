"""Learned node prices with online full-fleet rollout dispatch.

The supplied deterministic pricing actor is frozen when the controller is
constructed. Both actual pricing decisions and every simulated future period
use that actor through the same before-arrival information restriction as Joint
PPO. Dispatch retains the original rollout candidates, reward, and physics.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..benchmarks.policies import dispatch_outcome, service_first_dispatch
from ..simulator import ChargingSimulator
from .adaptive_pricing import pricing_observation
from .fitted_train import _new_run_directory
from .rollout_adp import (
    MAX_FUTURE_QUERIES_PER_DECISION, RolloutADPConfig, RolloutADPPolicy,
    _integer, evaluate_rollout_policy,
)

PRICE_TRACE_ENCODING = (
    'Concatenated price-index vectors as little-endian signed int64, in '
    'candidate/draw/future-period/zone order; no shape header.'
)


def _price_levels(obs, scenario, pricing_policy):
    """Copy only allowed pricing information and validate the actor's output."""
    allowed = pricing_observation(obs, scenario)
    active = np.asarray(allowed['pricing_mask'], dtype=bool).any(axis=1)
    # Deterministic inference cannot consume the live environment or planning
    # random stream. This is the same inference mode in real and future states.
    levels = np.asarray(pricing_policy.prices(allowed, rng=None))
    if (levels.shape != (scenario.n_zones,) or levels.dtype.kind not in 'iu'
            or (levels < 0).any() or (levels >= len(scenario.price_levels)).any()):
        raise ValueError('Pricing policy must return one valid integer price index per zone')
    levels = levels.astype(np.int64, copy=True)
    levels[~active] = 0
    return levels, active


def sample_priced_dispatch_observation(scenario, period, locations, energies, seed, pricing_policy):
    """Generate a future period, price it through the actor, then sample choices.

    This is a model query at a specified fleet state, not access to a live
    simulator. It uses the original period preparation and customer-choice
    logic. The seed couples arrivals, request sizes, station shocks, and choice
    uniforms across candidates. The actor sees observed station prices and
    prescribed arrival rates, but never the generated latent request details.

    A constant-price actor produces exactly the same observation as
    ``sample_dispatch_observation`` with the same price index and seed.
    """
    if not callable(getattr(pricing_policy, 'prices', None)):
        raise ValueError('Pricing policy must expose a prices method')
    if getattr(pricing_policy, 'scenario', None) != scenario:
        raise ValueError('Pricing policy and rollout scenario must match')
    return _sample_priced_dispatch_observation(scenario, period, locations, energies, seed, pricing_policy)


def _sample_priced_dispatch_observation(scenario, period, locations, energies, seed, pricing_policy):
    """Internal query after the public wrapper/constructor checks actor identity.

    Avoid reconstructing and validating the actor's immutable scenario on every
    model query. Fleet state and physical bounds are still checked every time.
    """
    _integer(period, 'period')
    _integer(seed, 'seed')
    if period >= scenario.horizon:
        raise ValueError('Synthetic dispatch period must precede the horizon')
    locations = np.asarray(locations)
    energies = np.asarray(energies, dtype=float)
    if (locations.shape != (scenario.n_mcv,) or locations.dtype.kind not in 'iu'
            or (locations < 0).any() or (locations >= scenario.n_zones).any()):
        raise ValueError('Synthetic locations must be one valid integer node per vehicle')
    if (energies.shape != (scenario.n_mcv,) or not np.isfinite(energies).all()
            or (energies < scenario.reserve_kwh).any()
            or (energies > scenario.battery_capacity_kwh).any()):
        raise ValueError('Synthetic energy must lie between reserve and capacity')
    sim = ChargingSimulator(scenario)
    sim.rng = np.random.default_rng(int(seed))
    sim.period = int(period)
    sim.stage = 0
    sim.locations = locations.astype(np.int64, copy=True)
    sim.energy = energies.copy()
    sim.cumulative_profit = 0.
    sim.cumulative_discounted_profit = 0.
    sim._prepare_period()
    sim._ready = True
    levels, _ = _price_levels(sim.observe(), scenario, pricing_policy)
    post, reward, done, truncated, _ = sim.quote_price_levels(levels)
    if reward != 0. or done or truncated:
        raise RuntimeError('Unexpected synthetic pricing transition')
    return post


class PricedRolloutADPPolicy(RolloutADPPolicy):
    """Frozen learned pricing and rollout dispatch under that same pricing rule.

    ``rollout_config.price_index`` is retained only for compatibility with the
    original rollout configuration; it is not used to price any period. The
    supplied actor, not that field, determines all current and future prices.
    """
    def __init__(self, scenario, pricing_policy, rollout_config=None, *, extra_proposal=None):
        scenario.validate()
        if (not callable(getattr(pricing_policy, 'prices', None))
                or not callable(getattr(pricing_policy, 'provenance', None))
                or getattr(pricing_policy, 'scenario', None) != scenario):
            raise ValueError('A matching deterministic pricing policy with provenance is required')
        # CEM can reuse candidate buffers outside this controller. A private
        # snapshot prevents those later changes from changing its continuation.
        self.pricing_policy = deepcopy(pricing_policy)
        try:
            self._pricing_provenance = json.loads(json.dumps(
                self.pricing_policy.provenance(), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError('Pricing provenance must be finite JSON data') from exc
        if not isinstance(self._pricing_provenance, dict):
            raise ValueError('Pricing provenance must be a mapping')
        super().__init__(scenario, rollout_config, extra_proposal=extra_proposal)
        self.pricing_decisions = []
        self.statistics.update(current_pricing_calls=0, future_pricing_calls=0,
            current_active_price_counts=[0]*len(scenario.price_levels),
            future_active_price_counts=[0]*len(scenario.price_levels))

    def prices(self, obs, rng=None):
        levels, active = _price_levels(obs, self.scenario, self.pricing_policy)
        counts = np.bincount(levels[active], minlength=len(self.scenario.price_levels))
        self.statistics['current_pricing_calls'] += 1
        self.statistics['current_active_price_counts'] = (
            np.asarray(self.statistics['current_active_price_counts'])+counts).tolist()
        width = self.scenario.battery_capacity_kwh-self.scenario.reserve_kwh
        usable_fraction = ((float(np.mean(obs['energy_kwh']))-self.scenario.reserve_kwh)/width
                           if width else 0.)
        self.pricing_decisions.append(dict(episode_key=self.episode_key,
            period=int(np.asarray(obs['period'])[0]), price_levels=levels.tolist(),
            mobile_available=active.tolist(), fleet_usable_battery_fraction=usable_fraction))
        return levels

    def dispatch(self, obs, rng=None):
        started = time.perf_counter()
        scenario, config = self.scenario, self.config
        period = np.asarray(obs['period'])
        if (int(obs['stage']) != 1 or period.shape != (1,) or period.dtype.kind not in 'iu'
                or not 0 <= period[0] < scenario.horizon):
            raise ValueError('Rollout dispatch needs a valid current dispatch observation')
        period = int(period[0])
        names, plans, coordinate_stats = self._candidate_plans(obs)
        remaining = scenario.horizon-period-1
        query_budget = len(plans)*config.rollout_samples*remaining
        if query_budget > MAX_FUTURE_QUERIES_PER_DECISION:
            raise ValueError('Actual rollout query budget exceeded')
        # Keep the old draw schedule exactly: the learned actor itself draws no
        # random numbers, and all candidates share this future seed table.
        seed_shape = (remaining, config.rollout_samples)
        seeds = 2*self._rng.integers(0, np.iinfo(np.int64).max//2, size=seed_shape, dtype=np.int64)
        while len(np.unique(seeds)) != seeds.size:
            seeds = 2*self._rng.integers(0, np.iinfo(np.int64).max//2, size=seed_shape, dtype=np.int64)
        outcomes = [dispatch_outcome(obs, scenario, actions) for actions in plans]
        scores = np.repeat(np.asarray([o['reward'] for o in outcomes])[:, None], config.rollout_samples, axis=1)
        price_hash = hashlib.sha256()
        counts = np.zeros(len(scenario.price_levels), dtype=np.int64)
        first_prices, completed = None, 0
        for candidate, outcome in enumerate(outcomes):
            for draw in range(config.rollout_samples):
                locations, energies = outcome['next_locations'], outcome['next_energy_kwh']
                discount = scenario.discount_per_period
                for offset, future_period in enumerate(range(period+1, scenario.horizon)):
                    energies = self._clip_roundoff(energies)
                    future = _sample_priced_dispatch_observation(scenario, future_period, locations,
                        energies, int(seeds[offset, draw]), self.pricing_policy)
                    levels = np.searchsorted(scenario.price_levels, future['mobile_prices']).astype(np.int64)
                    price_hash.update(levels.astype('<i8', copy=False).tobytes(order='C'))
                    if first_prices is None:
                        first_prices = levels.tolist()
                    active = np.asarray(future['mobile_available'], dtype=bool)
                    counts += np.bincount(levels[active], minlength=len(scenario.price_levels))
                    actions = self._validated_actions(future, service_first_dispatch(future, scenario,
                        config.base_recharge_threshold))
                    step = dispatch_outcome(future, scenario, actions)
                    scores[candidate, draw] += discount*step['reward']
                    discount *= scenario.discount_per_period
                    locations, energies = step['next_locations'], step['next_energy_kwh']
                    completed += 1
        if completed != query_budget or not np.isfinite(scores).all():
            raise FloatingPointError('Rollout query count or value is invalid')
        means = scores.mean(axis=1)
        winner = int(np.argmax(means))
        self.statistics['decisions'] += 1
        self.statistics['candidate_plans'] += len(plans)
        self.statistics['current_plan_outcomes'] += len(plans)
        self.statistics['future_model_queries'] += completed
        self.statistics['exogenous_draws'] += int(seeds.size)
        self.statistics['coordinate_candidate_outcomes'] += coordinate_stats['candidate_outcomes']
        self.statistics['future_pricing_calls'] += completed
        self.statistics['future_active_price_counts'] = (
            np.asarray(self.statistics['future_active_price_counts'])+counts).tolist()
        self.decisions.append(dict(episode_key=self.episode_key, episode_decision=self.episode_decision,
            period=period, candidates=names, selected=names[winner], current_profit=[o['reward'] for o in outcomes],
            mean_total_scores=means.tolist(), sampled_total_scores=scores.tolist(),
            future_periods=list(range(period+1, scenario.horizon)), future_draw_seeds=seeds.tolist(),
            future_model_queries=completed, query_budget=query_budget,
            coordinate_candidate_outcomes=coordinate_stats['candidate_outcomes'],
            future_pricing_calls=completed, future_active_price_counts=counts.tolist(),
            future_price_levels_sha256=price_hash.hexdigest(), first_future_price_levels=first_prices,
            seconds=time.perf_counter()-started))
        self.episode_decision += 1
        return plans[winner]

    def provenance(self):
        result = super().provenance()
        package = Path(__file__).resolve().parents[1]
        sources = dict(result['source_sha256'])
        for name in ('rl/priced_rollout.py', 'rl/adaptive_pricing.py'):
            sources['src/charging_env/'+name] = hashlib.sha256((package/name).read_bytes()).hexdigest()
        result.update(schema='charging_env.priced_rollout.v1',
            algorithm='learned_pricing_with_online_rollout_adp',
            offline_training=True, offline_training_scope='Supplied pricing policy only; no fitted dispatch value.',
            fixed_price_index=None, pricing=deepcopy(self._pricing_provenance),
            pricing_information='Period, fleet, prescribed arrival rates, observed station prices and grid buy price.',
            pricing_inference='Same frozen deterministic actor in actual pricing and all future rollouts.',
            unused_rollout_setting='config.price_index is ignored by this learned-pricing controller.',
            future_price_trace_encoding=PRICE_TRACE_ENCODING, source_sha256=sources)
        return result


def main(argv=None):
    """Evaluate a saved pricing actor coupled to the rollout dispatcher."""
    from .adaptive_pricing import LearnedPricingPolicy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pricing-checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7, help='Planning randomness, not a training seed')
    parser.add_argument('--rollout-samples', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--eval-seed-base', type=int, default=1_000_001)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/priced_rollout'))
    args = parser.parse_args(argv)
    try:
        _integer(args.episodes, 'episodes', 1)
        _integer(args.eval_seed_base, 'eval_seed_base')
        pricer = LearnedPricingPolicy.load_checkpoint(args.pricing_checkpoint)
        policy = PricedRolloutADPPolicy(pricer.scenario, pricer,
            RolloutADPConfig(seed=args.seed, rollout_samples=args.rollout_samples))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    directory = _new_run_directory(args.output_dir)
    seeds = [args.eval_seed_base+2*i for i in range(args.episodes)]
    checkpoint_hash = hashlib.sha256(args.pricing_checkpoint.read_bytes()).hexdigest()
    protocol = policy.provenance() | dict(evaluation_seeds=seeds, action_seed=6999,
        pricing_checkpoint_sha256=checkpoint_hash)
    (directory/'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    print(f'Learned pricing with rollout ADP. Outputs: {directory}', flush=True)
    started = time.perf_counter()
    result = evaluate_rollout_policy(policy, seeds, label='priced_rollout_adp')
    if (policy.provenance()['source_sha256'] != protocol['source_sha256']
            or hashlib.sha256(args.pricing_checkpoint.read_bytes()).hexdigest() != checkpoint_hash):
        raise RuntimeError('Pricing checkpoint or rollout source changed during evaluation')
    for filename, value in (('evaluation.json', result), ('planning_trace.json', policy.decisions),
                            ('pricing_trace.json', policy.pricing_decisions),
                            ('provenance.json', policy.provenance())):
        (directory/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    summary = {key: result[key] for key in ('n_episodes', 'mean_discounted_profit', 'confidence_interval_95',
        'mean_unmet_mobile_kwh', 'mean_final_inventory_kwh', 'mean_policy_decision_ms',
        'invalid_actions', 'action_counts', 'planning_statistics')}
    summary.update(wall_seconds=time.perf_counter()-started, pricing_checkpoint_sha256=checkpoint_hash,
        evaluation_sha256=hashlib.sha256((directory/'evaluation.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest())
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
