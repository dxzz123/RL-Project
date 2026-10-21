"""Coupled advantages for two decision stages per physical period."""
from numbers import Real

import numpy as np


def inventory_potential_rewards(rewards, pre_obs, dones, gamma, coefficient):
    """Return shaped training rewards for concatenated complete episodes.

    Phi(s) is ``coefficient * sum(s['energy_kwh'])`` before each physical
    period, and zero after an episode terminates. Thus discounted shaped
    return differs from raw return only by minus the initial potential.
    Terminal inventory never receives an extra reward, and no potential is
    carried across an episode boundary. Inputs, including raw rewards, are
    not modified. Pricing's zero-reward transition is unchanged.
    """
    if (isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Real)
            or not np.isfinite(coefficient) or coefficient < 0):
        raise ValueError('coefficient must be finite and nonnegative')
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real)
            or not np.isfinite(gamma) or not 0 < gamma <= 1):
        raise ValueError('gamma must be finite and in (0, 1]')
    reward = np.asarray(rewards, dtype=np.float64)
    terminal = np.asarray(dones)
    if (reward.ndim != 1 or terminal.shape != reward.shape
            or len(pre_obs) != len(reward) or not np.isfinite(reward).all()
            or not np.isin(terminal, [0, 1]).all()):
        raise ValueError('Rewards, observations, and binary dones must have matching finite-vector lengths')
    if len(reward) and not terminal[-1]:
        raise ValueError('Inventory shaping requires complete episodes')
    if coefficient == 0 or not len(reward):
        return reward.copy()
    potential = np.empty(len(reward), dtype=np.float64)
    for index, observation in enumerate(pre_obs):
        energy = np.asarray(observation['energy_kwh'], dtype=np.float64)
        if energy.ndim != 1 or not len(energy) or not np.isfinite(energy).all():
            raise ValueError('Each pre-period observation must contain a finite energy vector')
        potential[index] = coefficient * energy.sum()
    if not np.isfinite(potential).all():
        raise ValueError('Inventory potentials must remain finite')
    following = np.zeros_like(potential)
    if len(potential) > 1:
        following[:-1] = np.where(terminal[:-1], 0., potential[1:])
    shaped = reward + gamma * following - potential
    if not np.isfinite(shaped).all():
        raise ValueError('Shaped rewards must remain finite')
    return shaped


def stage_gae(rewards, price_values, dispatch_values, next_price_values, dones,
              gamma, gae_lambda):
    """GAE with intra-period discount/trace 1 and inter-period gamma/lambda.

    Flat arrays may concatenate complete episodes; dones cut every recursion.
    At lambda=1 both value targets are ordinary discounted team returns.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (rewards, price_values, dispatch_values, next_price_values)]
    reward, vp, vd, next_vp = arrays
    terminal = np.asarray(dones)
    if (reward.ndim != 1 or any(x.shape != reward.shape for x in arrays)
            or terminal.shape != reward.shape or not np.isin(terminal, [0, 1]).all()
            or any(not np.isfinite(x).all() for x in arrays)):
        raise ValueError('All rollout arrays must be finite vectors with matching lengths; dones must be binary')
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError('gamma must be in (0,1], gae_lambda in [0,1]')
    pricing_adv = np.zeros_like(reward)
    dispatch_adv = np.zeros_like(reward)
    following_pricing_adv = 0.
    for t in range(len(reward) - 1, -1, -1):
        continuation = 1. - float(terminal[t])
        delta_p = vd[t] - vp[t]
        delta_d = reward[t] + gamma * continuation * next_vp[t] - vd[t]
        dispatch_adv[t] = delta_d + gamma * gae_lambda * continuation * following_pricing_adv
        pricing_adv[t] = delta_p + dispatch_adv[t]
        following_pricing_adv = pricing_adv[t]
    return {
        'pricing_advantages': pricing_adv,
        'dispatch_advantages': dispatch_adv,
        'pricing_returns': pricing_adv + vp,
        'dispatch_returns': dispatch_adv + vd,
    }
