"""Three interpretable benchmark families for the fitted-value comparison.

Only validation selects settings. Existing benchmark implementations and their
historical results remain available in :mod:`charging_env.benchmarks.policies`.
This module uses NumPy and the simulator, without importing PyTorch.
"""
from collections import Counter

import numpy as np

from . import policies
from .evaluation import evaluate_policy


FAMILIES = ('service_first', 'constant_inventory', 'time_inventory')
ENERGY_FRACTIONS = (.5, 1.)


def _price_index(config, price):
    matches = np.flatnonzero(np.isclose(config.price_levels, price, rtol=0., atol=1e-12))
    if len(matches) != 1:
        raise ValueError(f'The compact comparison requires the admissible price {price}')
    return int(matches[0])


def candidates(config):
    """Return the fixed 82-setting search grid, in deterministic tie order.

    Service-first uses every price and three recharge thresholds. Constant
    inventory uses five prices and nine shadow values. Time inventory fixes
    price 0.60 and uses 16 early/late shadow schedules. Both coordinated rules
    share the fitted-value controller's half/full feasible exchange grid.
    """
    config.validate()
    if len(config.price_levels) != 7:
        raise ValueError('The predeclared compact study requires seven price levels')
    result = []
    for price in range(len(config.price_levels)):
        for threshold in (.2, .4, .6):
            result.append({'id': f'service_p{price}_b{threshold:g}', 'family': 'service_first',
                           'price_index': price, 'recharge_threshold': threshold})
    for price in (.4, .5, .6, .8, 1.):
        index = _price_index(config, price)
        for shadow in (.05, .10, .15, .20, .25, .30, .35, .40, .45):
            result.append({'id': f'constant_p{index}_s{shadow:g}', 'family': 'constant_inventory',
                           'price_index': index, 'shadow_price': shadow})
    for early in (.1, .2, .3, .4):
        for late in (0., .05, .10, .15):
            result.append({'id': f'time_e{early:g}_l{late:g}', 'family': 'time_inventory',
                           'price_index': _price_index(config, .6),
                           'early_shadow_price': early, 'late_shadow_price': late,
                           'switch_period': config.horizon // 2})
    if Counter(item['family'] for item in result) != {
            'service_first': 21, 'constant_inventory': 45, 'time_inventory': 16}:
        raise ValueError('Unexpected compact candidate grid')
    return result


def inventory_shadow(obs, config, spec):
    """Use an early/late value, with zero shadow in the terminal period.

    Periods are zero based: for the twelve-period study, 0--5 use the early
    shadow and 6--10 use the late shadow. Period 11 has zero continuation value.
    The simulator's actual terminal-inventory reward is still included.
    """
    period = int(np.asarray(obs['period'])[0])
    if period + 1 >= config.horizon:
        return 0.
    if spec['family'] == 'constant_inventory':
        return float(spec['shadow_price'])
    if spec['family'] == 'time_inventory':
        key = 'early_shadow_price' if period < spec['switch_period'] else 'late_shadow_price'
        return float(spec[key])
    raise ValueError('Inventory shadow requires a coordinated inventory family')


def evaluate_rule(config, seeds, spec, label=None, *, action_seed=159991):
    """Execute a selected setting through the common public-observation API."""
    family = spec.get('family')
    if family not in FAMILIES:
        raise ValueError(f'Unknown compact benchmark family: {family}')
    # Selection rows also contain the validation score; it is not a setting.
    settings = {key: value for key, value in spec.items() if key != 'mean_discounted_profit'}
    if settings not in candidates(config):
        raise ValueError('Benchmark settings must belong to the frozen compact search grid')
    pricing = lambda obs, rng: policies.fixed_prices(obs, config, settings['price_index'])
    if family == 'service_first':
        dispatch = lambda obs, rng: policies.service_first_dispatch(obs, config, settings['recharge_threshold'])
    else:
        dispatch = lambda obs, rng: policies.coordinated_dispatch(
            obs, config, inventory_shadow(obs, config, settings), energy_fractions=ENERGY_FRACTIONS)
    result = evaluate_policy(config, seeds, pricing, dispatch, action_seed=action_seed,
                             label=label or settings['id'])
    result['policy_specification'] = settings
    if result['invalid_actions']:
        raise RuntimeError('Compact rule emitted invalid actions: ' + result['label'])
    return result
