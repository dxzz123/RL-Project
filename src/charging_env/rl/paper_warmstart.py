"""Optional inventory-rule initialization before on-policy Joint PPO training.

The teacher supplies feasible actions and complete discounted returns. It uses
only the current observation and a fixed inventory-value schedule. Its action
grid matches Joint PPO, including maximum feasible energy exchange. Training
and evaluation rewards remain the simulator's original business profit.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from ..benchmarks.evaluation import episode_summary
from ..benchmarks.policies import joint_dispatch_outcomes
from ..simulator import ChargingSimulator, Mode


TEACHER = 'maximum_exchange_time_inventory_v1'
COORDINATE_TEACHER = 'coordinate_maximum_exchange_time_inventory_v1'


def validate_teacher_scenario(scenario, solver='exhaustive'):
    """Keep the original exact teacher explicit; allow bounded larger-fleet search."""
    if solver not in ('exhaustive', 'coordinate'):
        raise ValueError('Inventory dispatch solver must be exhaustive or coordinate')
    if solver == 'exhaustive' and scenario.n_mcv != 2:
        raise ValueError('Inventory warm start requires exactly two mobile vehicles')
    teacher_price_index(scenario)


def teacher_price_index(scenario):
    """The teacher's quote: the middle premium, or the historical 0.60 level."""
    if scenario.mcv_pricing_mode == 'premium':
        return scenario.n_price_levels//2
    matches = np.flatnonzero(np.isclose(scenario.price_levels, .60, rtol=0., atol=1e-8))
    if not matches.size:
        raise ValueError('Inventory warm start requires a 0.60 price level')
    return int(matches[0])


def inventory_teacher_prices(obs, scenario, *, solver='exhaustive'):
    from .paper_ppo import _fleet_pricing_mask
    validate_teacher_scenario(scenario, solver)
    if int(obs['stage']) != 0:
        raise ValueError('Inventory teacher pricing requires the before-choice stage')
    index = teacher_price_index(scenario)
    return np.where(_fleet_pricing_mask(obs, scenario).any(-1), index, 0).astype(np.int64)


def inventory_teacher_actions(obs, scenario, *, diagnostics=None, solver='exhaustive', sweeps=2):
    """Return feasible labels from exact or explicitly bounded inventory search."""
    from .paper_ppo import paper_dispatch_mask
    validate_teacher_scenario(scenario, solver)
    if solver == 'coordinate':
        return _coordinate_teacher_actions(obs, scenario, sweeps, diagnostics)
    modes, targets, _, outcomes = joint_dispatch_outcomes(obs, scenario, energy_fractions=(1.,))
    operation_by_mode = np.array([1, 0, 1, 2, 3], dtype=np.int64)
    actions = operation_by_mode[modes] * scenario.n_zones + targets
    mask = paper_dispatch_mask(obs, scenario).reshape(scenario.n_mcv, -1)
    feasible = mask[np.arange(scenario.n_mcv)[None, :], actions].all(axis=1)
    if diagnostics is not None:
        diagnostics['joint_candidate_outcomes'] = diagnostics.get('joint_candidate_outcomes', 0) + len(actions)
        diagnostics['admissible_joint_candidates'] = diagnostics.get('admissible_joint_candidates', 0) + int(feasible.sum())
    period = int(obs['period'][0])
    shadow = 0. if period + 1 >= scenario.horizon else (.30 if period < scenario.horizon // 2 else .15)
    scores = outcomes['reward'] + shadow * (
        outcomes['next_energy_kwh'].sum(axis=1) - np.sum(obs['energy_kwh']))
    scores[~feasible] = -np.inf
    if not feasible.any():
        raise RuntimeError('Inventory teacher has no feasible Joint PPO action')
    return actions[int(np.argmax(scores))].copy()


def _coordinate_teacher_actions(obs, scenario, sweeps, diagnostics):
    """Adapt bounded full-fleet search to the actor's four-operation action grid.

    Native WAIT and REPOSITION both map to IDLE. Repositioning to the current
    node is excluded so IDLE at that node has exactly the WAIT transition.
    Exchange quantities stay at the individually feasible maximum; shared
    station allocation is evaluated jointly by the bounded search.
    """
    from .paper_ppo import paper_dispatch_mask, paper_dispatch_commands, PaperOperation
    from ..benchmarks.scalable import coordinate_inventory_dispatch
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError('Coordinate teacher sweeps must be a positive integer')
    masks = paper_dispatch_mask(obs, scenario)
    native = np.zeros((scenario.n_mcv, len(Mode), scenario.n_zones), dtype=bool)
    native[:, Mode.SERVE] = masks[:, PaperOperation.SERVICE]
    native[:, Mode.REPOSITION] = masks[:, PaperOperation.IDLE]
    native[:, Mode.RECHARGE] = masks[:, PaperOperation.CHARGE]
    native[:, Mode.DISCHARGE] = masks[:, PaperOperation.DISCHARGE]
    current = np.asarray(obs['locations'], dtype=int)
    native[np.arange(scenario.n_mcv), Mode.REPOSITION, current] = False
    native[np.arange(scenario.n_mcv), Mode.WAIT, current] = masks[
        np.arange(scenario.n_mcv), PaperOperation.IDLE, current]
    period = int(obs['period'][0])
    shadow = 0. if period + 1 >= scenario.horizon else (.30 if period < scenario.horizon // 2 else .15)
    commands = coordinate_inventory_dispatch(obs, scenario, shadow, sweeps=sweeps,
        energy_fractions=(1.,), action_mask=native, diagnostics=diagnostics)
    if len(commands) != scenario.n_mcv:
        raise RuntimeError('Coordinate teacher must return one action per vehicle')
    operation_by_mode = np.array([1, 0, 1, 2, 3], dtype=np.int64)
    actions = np.asarray([operation_by_mode[int(a.mode)] * scenario.n_zones + a.target_zone
                          for a in commands], dtype=np.int64)
    # Verify that no quantity chosen by the search is lost when forming labels.
    reconstructed = paper_dispatch_commands(obs, actions, scenario)
    if any(a.mode != b.mode or a.target_zone != b.target_zone
           or not np.isclose(a.energy_kwh, b.energy_kwh, rtol=0., atol=1e-8)
           for a, b in zip(commands, reconstructed)):
        raise RuntimeError('Coordinate teacher action differs from the Joint PPO action convention')
    return actions


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fit_inventory_warmstart(trainer, output_dir=None):
    """Fit all four networks; called once by ``PaperPPOTrainer.warm_start``.

    Demonstrations and shuffle randomness use separate streams from subsequent
    PPO rollouts. Independent temporary optimizers leave PPO's Adam moments
    fresh. Demonstration episodes are recorded separately from on-policy work.
    """
    from .paper_ppo import paper_pricing_observation, paper_dispatch_commands
    c, scenario = trainer.config, trainer.scenario
    solver = c.warmstart_dispatch_solver
    validate_teacher_scenario(scenario, solver)
    if c.warmstart_episodes < 1:
        raise ValueError('A warm start needs at least one demonstration episode')
    directory = None if output_dir is None else Path(output_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('warmstart_summary.json', 'demonstration_summary.json', 'pretraining_metrics.json'):
            if (directory / name).exists():
                raise FileExistsError('Warm-start output already exists: ' + str(directory / name))
    started = time.perf_counter()
    source_hashes = trainer._source_hashes()
    pre, post, price_actions, dispatch_actions, targets, episodes = [], [], [], [], [], []
    seeds = [2 * (c.warmstart_seed_base + c.seed + ep) for ep in range(c.warmstart_episodes)]
    teacher_statistics = {}
    for seed in seeds:
        simulator = ChargingSimulator(scenario)
        observation, _ = simulator.reset(seed=seed)
        rewards, infos = [], []
        while simulator.stage != 2:
            pre.append(paper_pricing_observation(observation, scenario))
            prices = inventory_teacher_prices(observation, scenario, solver=solver)
            price_actions.append(prices)
            after_prices, rp, done, truncated, _ = simulator.quote_price_levels(prices)
            if rp != 0. or done or truncated:
                raise RuntimeError('Unexpected inventory demonstration pricing transition')
            post.append(after_prices)
            actions = inventory_teacher_actions(after_prices, scenario, diagnostics=teacher_statistics,
                solver=solver, sweeps=c.warmstart_coordinate_sweeps)
            dispatch_actions.append(actions)
            observation, reward, done, truncated, info = simulator.dispatch(
                paper_dispatch_commands(after_prices, actions, scenario))
            if truncated or info['invalid_actions']:
                raise RuntimeError('Inventory demonstration emitted an invalid action')
            rewards.append(reward)
            infos.append(info)
        discounted, following = [], 0.
        for reward in reversed(rewards):
            following = reward + scenario.discount_per_period * following
            discounted.append(following / c.reward_scale)
        targets.extend(reversed(discounted))
        episodes.append(episode_summary(seed, infos, rewards, scenario.discount_per_period,
                                        n_mcv=scenario.n_mcv))
    prices_np, dispatch_np, targets_np = (
        np.asarray(price_actions, dtype=np.int64), np.asarray(dispatch_actions, dtype=np.int64),
        np.asarray(targets, dtype=np.float64))
    sample_hash = hashlib.sha256()
    for value in (np.asarray(seeds, dtype=np.int64), prices_np, dispatch_np, targets_np):
        sample_hash.update(value.tobytes())
    prices = torch.as_tensor(prices_np, dtype=torch.long, device=trainer.device)
    actions = torch.as_tensor(dispatch_np, dtype=torch.long, device=trainer.device)
    returns = torch.as_tensor(targets_np, dtype=torch.float32, device=trainer.device)
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=c.warmstart_learning_rate, eps=1e-5)
                  for name, model in trainer._modules().items()}
    rng = np.random.default_rng(909000 + c.seed)
    metrics, optimizer_steps = [], 0
    for model in trainer._modules().values():
        model.train()
    for epoch in range(1, c.warmstart_epochs + 1):
        order = rng.permutation(len(pre))
        losses = []
        for start in range(0, len(pre), c.warmstart_minibatch_size):
            indices = order[start:start + c.warmstart_minibatch_size]
            ix = torch.as_tensor(indices, dtype=torch.long, device=trainer.device)
            before = [pre[int(i)] for i in indices]
            after = [post[int(i)] for i in indices]
            pricing = trainer.pricing_actor.evaluate_actions(before, prices[ix])
            dispatch = trainer.dispatch_actor.evaluate_actions(after, actions[ix])
            pricing_ce = -pricing['log_prob'].mean() / scenario.n_zones
            dispatch_ce = -dispatch['agent_log_probs'].mean()
            pricing_mse = (trainer.pricing_critic(before) - returns[ix]).square().mean()
            dispatch_mse = (trainer.dispatch_critic(after) - returns[ix]).square().mean()
            loss = pricing_ce + dispatch_ce + pricing_mse + dispatch_mse
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite inventory warm-start loss')
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for model in trainer._modules().values():
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            for optimizer in optimizers.values():
                optimizer.step()
            optimizer_steps += 1
            losses.append(dict(pricing_cross_entropy=float(pricing_ce.detach()),
                dispatch_cross_entropy=float(dispatch_ce.detach()),
                pricing_value_mse=float(pricing_mse.detach()), dispatch_value_mse=float(dispatch_mse.detach())))
        metrics.append(dict(epoch=epoch, **{key: float(np.mean([row[key] for row in losses]))
                                           for key in losses[0]}))
        if epoch == 1 or epoch % 25 == 0 or epoch == c.warmstart_epochs:
            print(f"Inventory warm start epoch {epoch}: dispatch cross entropy "
                  f"{metrics[-1]['dispatch_cross_entropy']:.4f}", flush=True)
    if trainer._source_hashes() != source_hashes:
        raise RuntimeError('Warm-start source changed during fitting')
    report = dict(teacher=TEACHER if solver == 'exhaustive' else COORDINATE_TEACHER,
        dispatch_solver=solver, coordinate_sweeps=c.warmstart_coordinate_sweeps if solver == 'coordinate' else None,
        solver_exact=solver == 'exhaustive', completed=True, demonstration_episodes=len(episodes),
        demonstration_periods=len(pre), teacher_pricing_queries=len(pre), teacher_dispatch_queries=len(post),
        seed_rule='2*(warmstart_seed_base+seed+episode_index)', seeds=seeds,
        shuffle_seed=909000 + c.seed, epochs=c.warmstart_epochs,
        minibatch_size=c.warmstart_minibatch_size, learning_rate=c.warmstart_learning_rate,
        optimizer_steps=optimizer_steps, network_optimizer_steps=4 * optimizer_steps,
        teacher_price=scenario.price_setting(teacher_price_index(scenario)),
        teacher_shadow_prices=dict(early=.30, late=.15, terminal=0.),
        teacher_exchange='maximum feasible, filtered to the Joint PPO action mask',
        teacher_joint_candidate_outcomes=teacher_statistics.get('joint_candidate_outcomes'),
        teacher_admissible_joint_candidates=teacher_statistics.get('admissible_joint_candidates'),
        teacher_solver_statistics=teacher_statistics.copy(),
        teacher_evaluated_joint_candidates=teacher_statistics.get(
            'joint_candidate_outcomes' if solver == 'exhaustive' else 'candidate_outcomes'),
        reward_scale=c.reward_scale, critic_targets='complete discounted raw returns divided by reward_scale',
        extra_training_reward=False, source_sha256=source_hashes,
        demonstration_sha256=sample_hash.hexdigest(),
        mean_demonstration_discounted_profit=float(np.mean([e['discounted_profit'] for e in episodes])),
        final_metrics=metrics[-1], seconds=time.perf_counter() - started)
    if directory is not None:
        _write_json(directory / 'demonstration_summary.json', dict(episodes=episodes))
        _write_json(directory / 'pretraining_metrics.json', metrics)
        _write_json(directory / 'warmstart_summary.json', report)
    return report
