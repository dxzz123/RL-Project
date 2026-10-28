"""Observation-only history collection and a fitted-model control boundary.

Only this evaluator owns the true Scenario. Controller factories receive a
separate fitted Scenario, and both decision stages receive fitted arrival
forecasts. Current accepted requests remain observable at dispatch. This is an
explicit experimental information boundary, not a sandbox against malicious
Python callbacks that deliberately capture external objects.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from numbers import Integral

import numpy as np

from ..benchmarks.evaluation import evaluate_policy, _seed_list
from ..benchmarks.policies import service_first_dispatch
from ..config import Scenario
from ..simulator import ChargingSimulator
from .adaptive_pricing import pricing_observation


BOUNDARY_SCHEMA = 'charging_env.history_control.v1'
# These are current public simulator fields, not arbitrary extra caller data.
DISPATCH_FIELDS = (
    'stage', 'period', 'locations', 'energy_kwh', 'arrival_rates',
    'requests_by_zone', 'potential_energy_kwh', 'request_details', 'mobile_prices',
    'mobile_demand', 'fixed_demand', 'mobile_requested_kwh', 'fixed_requested_kwh',
    'outside_requests', 'mobile_wait_hours', 'mobile_available', 'eligible_mcvs',
    'fcs_wait_hours', 'fcs_retail_prices', 'fcs_discharge_prices', 'grid_buy_price',
    'pricing_mask', 'action_mask',
)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name+' must be a nonnegative integer')


def _scenario_data(scenario):
    if not isinstance(scenario, Scenario):
        raise ValueError('Controllers must expose a Scenario')
    return json.loads(json.dumps(asdict(scenario), allow_nan=False))


def model_scenario_sha256(scenario):
    """Identity of the complete fitted generative model, including preferences."""
    data = json.dumps(_scenario_data(scenario), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def _period(obs, model, stage):
    period = np.asarray(obs['period'])
    if (int(obs['stage']) != stage or period.shape != (1,)
            or period.dtype.kind not in 'iu' or not 0 <= period[0] < model.horizon):
        raise ValueError('Expected a valid current '+('pricing' if stage == 0 else 'dispatch')+' observation')
    return int(period[0])


def history_pricing_observation(obs, fitted_scenario):
    """Substitute the fitted forecast before removing all latent quote data."""
    period = _period(obs, fitted_scenario, 0)
    # Do not inspect or validate the environment's true rate field at all.
    replaced = dict(obs)
    replaced['arrival_rates'] = fitted_scenario.resolved_arrival_rates[period].copy()
    return pricing_observation(replaced, fitted_scenario)


def history_dispatch_observation(obs, fitted_scenario):
    """Keep actual current requests, choices and physics; replace forecasts.

    Unlike pricing, dispatch occurs after the current quotes have been chosen.
    Its true realized request sizes, request identities, choices, prices and
    feasible-action mask are operational observations and must not be simulated
    again under the fitted demand model. Unknown additional fields are dropped.
    """
    period = _period(obs, fitted_scenario, 1)
    result = {name: deepcopy(obs[name]) for name in DISPATCH_FIELDS
              if name != 'arrival_rates'}
    result['arrival_rates'] = fitted_scenario.resolved_arrival_rates[period].copy()
    return result


def _compatible_public_physics(truth, model):
    from .history_model import public_model_config
    truth.validate()
    model.validate()
    if public_model_config(truth) != public_model_config(model):
        raise ValueError('Truth and fitted model must share known public physics and tariffs')


def _assert_controller_model(controller, expected):
    """Check the explicit adapters and PPO encoders used by this project."""
    objects = [('controller', controller)]
    for name in ('policy', 'pricing_policy', 'pricer', 'pricing_actor', 'dispatch_actor',
                 'pricing_critic', 'dispatch_critic'):
        item = getattr(controller, name, None)
        if item is not None:
            objects.append((name, item))
            encoder = getattr(item, 'encoder', None)
            if encoder is not None:
                objects.append((name+'.encoder', encoder))
    if not hasattr(controller, 'scenario'):
        raise ValueError('Controller must declare its fitted scenario')
    for name, item in objects:
        for field in ('scenario', 'training_scenario'):
            value = getattr(item, field, None)
            if value is not None and _scenario_data(value) != expected:
                raise ValueError(name+'.'+field+' differs from the fitted historical model')


def _query_count(controller):
    value = getattr(controller, 'statistics', {}).get('future_model_queries', 0)
    _integer(value, 'future_model_queries')
    return int(value)


def evaluate_history_policy(true_scenario, fitted_scenario, seeds, controller_factory,
                            *, action_seed=321, label='', on_decision=None):
    """Evaluate fitted-model control in a private, possibly shifted environment.

    ``controller_factory(fitted_scenario_copy)`` must return an object exposing
    ``scenario``, ``prices(obs, rng)`` and ``dispatch(obs, rng)``. Optional
    ``reset_episode(seed, action_seed)`` or PPO ``on_episode_start`` is honored.
    A PPO factory should return a policy loaded from a checkpoint trained on the
    fitted model; its training scenario and encoder scenarios are checked. No
    ``with_scenario`` or ``evaluate_scenario`` adaptation is performed here.

    The original evaluator computes every physical transition and profit ledger
    using truth. The factory, controller and optional ``on_decision(record)``
    hook never receive truth or a live simulator. The hook gets only period,
    episode ID, fitted-model identity and actual planner-query counter delta.
    The planner's existing decision traces retain its individual query seeds.
    """
    _compatible_public_physics(true_scenario, fitted_scenario)
    seeds = _seed_list(seeds)
    _integer(action_seed, 'action_seed')
    if not callable(controller_factory) or (on_decision is not None and not callable(on_decision)):
        raise ValueError('Controller factory and optional decision hook must be callable')
    model = deepcopy(fitted_scenario)
    expected = _scenario_data(model)
    model_hash = model_scenario_sha256(model)
    controller = controller_factory(deepcopy(model))
    if any(not callable(getattr(controller, name, None)) for name in ('prices', 'dispatch')):
        raise ValueError('Controller must expose prices and dispatch callbacks')
    _assert_controller_model(controller, expected)
    state, records = {}, []

    def check():
        if _scenario_data(model) != expected:
            raise ValueError('Fitted observation model changed during evaluation')
        _assert_controller_model(controller, expected)

    def start(seed, policy_seed):
        check()
        state['episode_key'] = int(seed)
        callback = getattr(controller, 'reset_episode', None)
        if callback is None:
            callback = getattr(controller, 'on_episode_start', None)
        if callback is not None:
            if not callable(callback):
                raise ValueError('Episode reset must be callable')
            callback(seed, policy_seed)
        check()

    def prices(obs, rng):
        check()
        result = controller.prices(history_pricing_observation(obs, model), rng)
        check()
        return result

    def dispatch(obs, rng):
        check()
        before = _query_count(controller)
        current = history_dispatch_observation(obs, model)
        result = controller.dispatch(current, rng)
        check()
        queries = _query_count(controller)-before
        if queries < 0:
            raise ValueError('Planner query counter decreased during a decision')
        record = dict(episode_key=state['episode_key'], period=int(obs['period'][0]),
                      model_scenario_sha256=model_hash, future_model_queries=queries)
        records.append(record)
        if on_decision is not None:
            on_decision(deepcopy(record))
        check()
        return result

    result = evaluate_policy(true_scenario, seeds, prices, dispatch,
        action_seed=int(action_seed), label=label, on_episode_start=start)
    check()
    result['history_boundary'] = dict(schema=BOUNDARY_SCHEMA,
        model_scenario_sha256=model_hash, forecast_source='fitted_historical_model',
        current_dispatch_data='actual observed quote requests, choices and physical state',
        true_model_passed_to_controller=False, controller_constructed_by_fitted_model_factory=True,
        model_queries=sum(row['future_model_queries'] for row in records), decisions=records)
    return result


def collect_historical_logs(true_scenario, environment_seeds, *, day_ids,
                            action_seed=81231, recharge_threshold=.2):
    """Log every quote opportunity under independent randomized legal prices.

    Only the private simulator receives truth. The logistical logging rule uses
    a public placeholder Scenario, not true arrival rates, request-size bounds or
    customer preferences. At dispatch it serves observed accepted requests, then
    recharges low batteries using public physics. Random offered price levels are
    independent uniform draws for every zone; inactive zones can still be quoted
    but their unavailable mobile option is recorded as unavailable.

    ``day_ids`` are chronological identifiers, not simulator seeds. Generation
    seeds belong in the experiment protocol outside the observation dataset.
    Zero count cells and outside choices are retained, avoiding sales censoring.
    No collection-time forecast, true parameter, probability or future state is
    included in the returned HistoricalLog.
    """
    from .history_model import HistoricalLog, public_model_config, public_scenario
    truth = true_scenario.validate()
    seeds = _seed_list(environment_seeds)
    _integer(action_seed, 'action_seed')
    ids = np.asarray(day_ids)
    if (ids.shape != (len(seeds),) or ids.dtype.kind not in 'iu' or (ids < 0).any()
            or (len(ids) > 1 and (np.diff(ids.astype(np.int64)) <= 0).any())):
        raise ValueError('day_ids must be increasing nonnegative integer chronological identifiers')
    if (isinstance(recharge_threshold, (bool, np.bool_)) or not np.isfinite(recharge_threshold)
            or not 0 <= recharge_threshold <= 1):
        raise ValueError('recharge_threshold must lie in [0,1]')
    public = public_model_config(truth)
    logging_model = public_scenario(public)
    counts = np.zeros((len(seeds), logging_model.horizon, logging_model.n_zones), dtype=np.int64)
    fields = {name: [] for name in ('request_day', 'request_period', 'request_zone',
        'request_energy_kwh', 'alternative_payment', 'mobile_wait_hours',
        'fixed_wait_hours', 'fixed_travel_km', 'available', 'chosen')}
    for day, (seed, day_id) in enumerate(zip(seeds, ids)):
        simulator = ChargingSimulator(truth)
        observation, _ = simulator.reset(seed=seed)
        rng = np.random.default_rng(np.random.SeedSequence([int(action_seed), int(day_id), 0x48495354]))
        for period in range(logging_model.horizon):
            # No raw observation enters the random pricing rule.
            levels = rng.integers(0, len(logging_model.price_levels), size=logging_model.n_zones)
            post, reward, done, truncated, _ = simulator.quote_price_levels(levels)
            if reward != 0. or done or truncated:
                raise RuntimeError('Unexpected logging pricing transition')
            current = history_dispatch_observation(post, logging_model)
            details = np.asarray(current['request_details'])
            origins, energy = details[:, 0].astype(np.int64), details[:, 1]
            n = len(origins)
            counts[day, period] = current['requests_by_zone']
            payments = np.zeros((n, logging_model.n_fcs+2))
            payments[:, 0] = energy*np.asarray(current['mobile_prices'])[origins]
            payments[:, 1:-1] = energy[:, None]*np.asarray(current['fcs_retail_prices'])
            available = np.ones((n, logging_model.n_fcs+2), dtype=bool)
            available[:, 0] = np.asarray(current['mobile_available'], dtype=bool)[origins]
            available[:, 1:-1] = details[:, 3:].astype(bool)
            fields['request_day'].append(np.full(n, day, dtype=np.int64))
            fields['request_period'].append(np.full(n, period, dtype=np.int64))
            fields['request_zone'].append(origins.copy())
            fields['request_energy_kwh'].append(energy.copy())
            fields['alternative_payment'].append(payments)
            fields['mobile_wait_hours'].append(np.asarray(current['mobile_wait_hours'])[origins].copy())
            fields['fixed_wait_hours'].append(np.broadcast_to(current['fcs_wait_hours'], (n, logging_model.n_fcs)).copy())
            fields['fixed_travel_km'].append(logging_model.distances[origins][:, logging_model.fcs_zones].copy())
            fields['available'].append(available)
            fields['chosen'].append(details[:, 2].astype(np.int64))
            actions = service_first_dispatch(current, logging_model, float(recharge_threshold))
            observation, _, done, truncated, info = simulator.dispatch(actions)
            if truncated or done != (period+1 == logging_model.horizon) or info['invalid_actions']:
                raise RuntimeError('Historical logging produced an invalid physical transition')
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    return HistoricalLog(public_config=public, counts=counts, day_ids=ids.astype(np.int64), **arrays)


def collect_historical_log(true_scenario, seeds, *, logging_seed=91731, start_day=0):
    """Convenience collector with consecutive chronological day identifiers."""
    seeds = _seed_list(seeds)
    _integer(start_day, 'start_day')
    return collect_historical_logs(true_scenario, seeds,
        day_ids=np.arange(int(start_day), int(start_day)+len(seeds), dtype=np.int64),
        action_seed=logging_seed)
