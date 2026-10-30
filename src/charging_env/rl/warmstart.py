"""Inventory-rule behavior cloning before on-policy SA-PPO/MAPPO training.

This is an imitation-assisted initialization, not a pure-RL update. Teacher
data never enters a PPO rollout, and neither critics nor PPO optimizers are
updated. Subsequent PPO collection must use fresh student-policy rollouts.
"""
from numbers import Real
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..benchmarks.policies import coordinated_dispatch, fixed_prices
from ..simulator import ChargingSimulator, Mode


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _validate(trainer, episodes, epochs, seed_base, price_index, shadow_price,
              minibatch_size, learning_rate):
    for name, value in (('episodes', episodes), ('epochs', epochs),
                        ('minibatch_size', minibatch_size)):
        _positive_integer(name, value)
    if (isinstance(seed_base, (bool, np.bool_)) or not isinstance(seed_base, (int, np.integer))
            or seed_base < 0 or seed_base % 2):
        raise ValueError('seed_base must be a nonnegative even integer')
    if (isinstance(price_index, (bool, np.bool_)) or not isinstance(price_index, (int, np.integer))
            or not 0 <= price_index < trainer.scenario.n_price_levels):
        raise ValueError('price_index must be a valid integer price-level index')
    for name, value in (('shadow_price', shadow_price), ('learning_rate', learning_rate)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if learning_rate == 0:
        raise ValueError('learning_rate must be positive')
    if not trainer.dispatch_actor.include_agent_id:
        raise ValueError('Inventory imitation requires dispatch_agent_id=True to represent vehicle roles')
    if any((trainer.update_count, trainer.episode_count, trainer.environment_periods)):
        raise ValueError('Warm start requires a fresh trainer with zero PPO counters')
    if trainer.actor_optimizer.state or trainer.critic_optimizer.state:
        raise ValueError('Warm start requires fresh, unused PPO optimizers')


def _collect_demonstrations(trainer, episodes, seed_base, price_index, shadow_price):
    """Retain stage-appropriate public inputs and frozen teacher labels only."""
    c = trainer.scenario
    scale = max(c.mobile_charge_kw, c.mobile_discharge_kw) * c.period_hours
    data = {'pre_obs': [], 'local_obs': [], 'price_labels': [],
            'dispatch_labels': [], 'energy_targets': [], 'episode_profit': [],
            'episode_discounted_profit': [], 'environment_seeds': []}
    for index in range(episodes):
        seed = int(seed_base + 2 * index)
        data['environment_seeds'].append(seed)
        sim = ChargingSimulator(c)
        pre, _ = sim.reset(seed=seed)
        done, profit, discounted_profit = False, 0., 0.
        period = 0
        while not done:
            # Fixed prices and the joint dispatcher see only their public stage.
            prices = fixed_prices(pre, c, int(price_index))
            post, pricing_reward, pricing_done, truncated, _ = sim.quote_price_levels(prices)
            if pricing_reward != 0 or pricing_done or truncated:
                raise RuntimeError('Unexpected demonstration pricing transition')
            local = [sim.local_observation(m) for m in range(c.n_mcv)]
            commands = coordinated_dispatch(post, c, float(shadow_price), energy_fractions=(.5, 1.))
            labels = np.asarray([int(a.mode) * c.n_zones + a.target_zone for a in commands],
                                dtype=np.int64)
            for observation, label in zip(local, labels):
                if not np.asarray(observation['action_mask'], dtype=bool).reshape(-1)[label]:
                    raise RuntimeError('Inventory teacher produced an infeasible label')
            targets = np.asarray([
                np.clip(command.energy_kwh / scale, 0., 1.)
                if command.mode in (Mode.RECHARGE, Mode.DISCHARGE) else 0.
                for command in commands], dtype=np.float32)
            # Keep labels consistent with the legacy requested-energy mapping.
            reconstructed = trainer._dispatch_commands(labels, targets)
            if any(not np.isclose(actual.energy_kwh, expected.energy_kwh, rtol=1e-6, atol=1e-6)
                   for actual, expected in zip(reconstructed, commands)):
                raise ValueError('Teacher energy targets do not match this trainer action mapping')
            data['pre_obs'].append(pre)
            data['local_obs'].extend(local)
            data['price_labels'].append(prices.copy())
            data['dispatch_labels'].append(labels)
            data['energy_targets'].append(targets)
            pre, reward, done, truncated, info = sim.dispatch(commands)
            if truncated or info['invalid_actions']:
                raise RuntimeError('Invalid demonstration dispatch transition')
            profit += reward
            discounted_profit += c.discount_per_period ** period * reward
            period += 1
        data['episode_profit'].append(profit)
        data['episode_discounted_profit'].append(discounted_profit)
    for key in ('price_labels', 'dispatch_labels', 'energy_targets'):
        data[key] = np.asarray(data[key])
    return data


def _batch(trainer, data, indices):
    """Supervised categorical losses and finite mean regression at 0/1 labels."""
    c, device = trainer.scenario, trainer.device
    indices = np.asarray(indices, dtype=np.int64)
    pre = [data['pre_obs'][int(index)] for index in indices]
    local = [data['local_obs'][int(index) * c.n_mcv + m]
             for index in indices for m in range(c.n_mcv)]
    price_labels = torch.as_tensor(data['price_labels'][indices], dtype=torch.long, device=device)
    dispatch_labels = torch.as_tensor(data['dispatch_labels'][indices].reshape(-1),
                                      dtype=torch.long, device=device)
    pricing, active = trainer.pricing_actor._distribution(pre)
    # Unavailable nodes have a constant dummy action and contribute zero NLL.
    pricing_loss = -pricing.log_prob(price_labels).sum(dim=-1).mean() / c.n_zones
    logits, concentrations = trainer.dispatch_actor(local)
    dispatch_loss = F.cross_entropy(logits, dispatch_labels)
    selected = concentrations[torch.arange(len(dispatch_labels), device=device), dispatch_labels]
    energy_mean = selected[:, 0] / selected.sum(dim=-1)
    energy_target = torch.as_tensor(data['energy_targets'][indices].reshape(-1),
                                   dtype=energy_mean.dtype, device=device)
    energy_mask = trainer.dispatch_actor.energy_actions[dispatch_labels]
    errors = energy_mean - energy_target
    energy_loss = (errors[energy_mask].square().mean() if bool(energy_mask.any())
                   else energy_mean.sum() * 0.)
    loss = pricing_loss + dispatch_loss + 5. * energy_loss
    return {
        'loss': loss, 'pricing_loss': pricing_loss, 'dispatch_loss': dispatch_loss,
        'energy_loss': energy_loss,
        'pricing_correct': ((pricing.logits.argmax(dim=-1) == price_labels) & active).sum(),
        'pricing_count': active.sum(),
        'dispatch_correct': (logits.argmax(dim=-1) == dispatch_labels).sum(),
        'dispatch_count': len(dispatch_labels),
        'energy_absolute_error': errors[energy_mask].abs().sum(),
        'energy_count': energy_mask.sum(),
    }


def _fit_metrics(trainer, data, minibatch_size):
    totals = dict(pricing_correct=0, pricing_count=0, dispatch_correct=0,
                  dispatch_count=0, energy_absolute_error=0., energy_count=0)
    periods = len(data['pre_obs'])
    with torch.no_grad():
        for start in range(0, periods, minibatch_size):
            values = _batch(trainer, data, np.arange(start, min(start + minibatch_size, periods)))
            for key in totals:
                value = values[key]
                totals[key] += value.item() if torch.is_tensor(value) else value
    return {
        'in_sample_pricing_accuracy': (totals['pricing_correct'] / totals['pricing_count']
                                       if totals['pricing_count'] else None),
        'in_sample_pricing_active_labels': int(totals['pricing_count']),
        'in_sample_dispatch_accuracy': totals['dispatch_correct'] / totals['dispatch_count'],
        'in_sample_energy_fraction_mae': (totals['energy_absolute_error'] / totals['energy_count']
                                          if totals['energy_count'] else None),
        'in_sample_energy_labels': int(totals['energy_count']),
    }


def warm_start_from_inventory(trainer, *, episodes=256, epochs=20, seed_base=60000000,
                              price_index=1, shadow_price=.30, minibatch_size=256,
                              learning_rate=1e-3):
    """Clone a frozen rule, then leave fresh PPO optimizers ready for student data.

    ``minibatch_size`` counts physical periods, including all vehicles' labels.
    Energy supervision regresses the selected Beta mean toward the teacher's
    requested kWh divided by the trainer's fixed exchange scale. Boundary
    targets 0 and 1 are valid MSE labels, never Beta-density evaluations.
    Demonstration fit is in-sample and is not an independent policy evaluation.
    The function advances ``trainer.rng`` for supervised epoch shuffling, while
    leaving all PPO counters and both PPO optimizer states unchanged.
    """
    _validate(trainer, episodes, epochs, seed_base, price_index, shadow_price,
              minibatch_size, learning_rate)
    started = time.monotonic()
    data = _collect_demonstrations(trainer, int(episodes), int(seed_base), int(price_index), shadow_price)
    collection_seconds = time.monotonic() - started
    actor_parameters = list(trainer.pricing_actor.parameters()) + list(trainer.dispatch_actor.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=float(learning_rate), eps=1e-5)
    periods, steps = len(data['pre_obs']), 0
    last_loss, largest_gradient_norm = None, 0.
    for _ in range(int(epochs)):
        order = trainer.rng.permutation(periods)
        for start in range(0, periods, int(minibatch_size)):
            values = _batch(trainer, data, order[start:start + int(minibatch_size)])
            loss = values['loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite imitation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [p.grad for p in actor_parameters if p.grad is not None]
            if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError('Nonfinite imitation actor gradient')
            norm = float(torch.sqrt(sum(gradient.double().square().sum() for gradient in gradients)).item())
            largest_gradient_norm = max(largest_gradient_norm, norm)
            optimizer.step()
            last_loss = float(loss.item())
            steps += 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in actor_parameters):
        raise FloatingPointError('Nonfinite imitation actor parameter')
    metrics = _fit_metrics(trainer, data, int(minibatch_size))
    # Do not leave supervised gradients attached to parameters before PPO begins.
    optimizer.zero_grad(set_to_none=True)
    if any((trainer.update_count, trainer.episode_count, trainer.environment_periods)):
        raise RuntimeError('Imitation unexpectedly changed PPO counters')
    if trainer.actor_optimizer.state or trainer.critic_optimizer.state:
        raise RuntimeError('Imitation unexpectedly changed a PPO optimizer')
    scale = max(trainer.scenario.mobile_charge_kw, trainer.scenario.mobile_discharge_kw) * trainer.scenario.period_hours
    metrics['in_sample_requested_energy_mae_kwh'] = (
        metrics['in_sample_energy_fraction_mae'] * scale
        if metrics['in_sample_energy_fraction_mae'] is not None else None)
    return {
        'method': 'Inventory-rule behavior cloning before on-policy SA-PPO/MAPPO',
        'scope': 'Imitation-assisted initialization, not pure RL. Reported fit is on demonstrations; no PPO rollout or independent test is used.',
        'demonstration_episodes': int(episodes), 'demonstration_periods': periods,
        'demonstration_dispatch_samples': periods * trainer.scenario.n_mcv,
        'environment_seeds': data['environment_seeds'],
        'epochs': int(epochs), 'minibatch_size_periods': int(minibatch_size),
        'learning_rate': float(learning_rate), 'optimizer_steps': steps,
        'loss_definition': 'pricing_joint_NLL / n_zones + dispatch_categorical_NLL + 5 * selected_Beta_mean_MSE',
        'teacher_configuration': {
            'family': 'fixed_inventory', 'price_index': int(price_index),
            'price_setting': trainer.scenario.price_setting(price_index),
            'shadow_price': float(shadow_price), 'energy_fractions': [.5, 1.],
            'charging_utility_intercept': float(trainer.scenario.charging_utility_intercept),
        },
        'mean_demo_profit': float(np.mean(data['episode_profit'])),
        'mean_demo_discounted_profit': float(np.mean(data['episode_discounted_profit'])),
        'teacher_invalid_actions': 0, 'all_gradients_finite': True,
        'largest_actor_gradient_norm': largest_gradient_norm, 'last_minibatch_loss': last_loss,
        **metrics,
        'demonstration_collection_seconds': collection_seconds,
        'elapsed_seconds': time.monotonic() - started,
    }
