"""Paired-budget candidate Double-Q and learned-continuation ADP training.

One transition spans one physical period, from post-choice dispatch to the
next post-choice dispatch under a frozen pricer. No rollout teacher or private
test environment enters training.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from numbers import Integral, Real
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from ..benchmarks.evaluation import _seed_list, evaluate_policy
from ..simulator import ChargingSimulator
from .candidate_value import CandidateValuePolicy, ValueNetwork, build_candidates
from .priced_rollout import _price_levels
from .rollout_adp import RolloutADPConfig

TRAINING_SCHEMA = 'charging_env.candidate_value_training.v1'


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _real(value, name, minimum=0., maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'Invalid {name}')


@dataclass(frozen=True)
class CandidateTrainingConfig:
    seed: int = 7
    episodes: int = 512
    hidden_size: int = 128
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 256
    updates_per_step: int = 2
    target_update: int = 100
    epsilon_start: float = .8
    epsilon_end: float = .05
    epsilon_decay_fraction: float = .8
    eval_interval: int = 128
    reward_scale: float = 100.
    training_seed_base: int = 4_600_000_000
    action_seed: int = 918991
    torch_threads: int = 1
    gradient_clip: float = 10.

    def validate(self):
        for key in ('seed', 'training_seed_base', 'action_seed', 'warmup_steps'):
            _integer(getattr(self, key), key, 0)
        for key in ('episodes', 'hidden_size', 'batch_size', 'replay_capacity',
                    'updates_per_step', 'target_update', 'eval_interval', 'torch_threads'):
            _integer(getattr(self, key), key)
        if self.replay_capacity < self.batch_size:
            raise ValueError('replay_capacity must hold one batch')
        for key in ('learning_rate', 'reward_scale', 'gradient_clip'):
            _real(getattr(self, key), key, minimum=1e-12)
        for key in ('epsilon_start', 'epsilon_end'):
            _real(getattr(self, key), key, maximum=1.)
        _real(self.epsilon_decay_fraction, 'epsilon_decay_fraction', 1e-12, 1.)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError('epsilon_end must not exceed epsilon_start')
        if self.training_seed_base % 2:
            raise ValueError('training_seed_base must be even')
        return self

    def training_seeds(self):
        self.validate()
        return tuple(int(self.training_seed_base + 2*i) for i in range(self.episodes))

    def epsilon(self, physical_step, total_steps):
        fraction = min(1., physical_step / max(1., self.epsilon_decay_fraction*total_steps))
        return float(self.epsilon_start + fraction*(self.epsilon_end-self.epsilon_start))


@dataclass(frozen=True)
class CandidateTransition:
    """Replay stores features, never simulator objects or private future draws."""
    features: np.ndarray
    reward: float
    done: bool
    next_q_features: np.ndarray
    next_value_features: np.ndarray
    next_rewards: np.ndarray
    next_terminal: bool


def _tensor(values):
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)


def candidate_scores(network, batch, method, gamma, reward_scale):
    """Return original-profit scores, using exact rewards in the final period."""
    if method not in ('q', 'hybrid'):
        raise ValueError('method must be q or hybrid')
    if batch.terminal:
        return np.asarray(batch.rewards, dtype=float).copy()
    features = batch.q_features if method == 'q' else batch.value_features
    with torch.no_grad():
        prediction = network(_tensor(features)).detach().cpu().numpy()
    prediction = (np.asarray(batch.rewards) + gamma*reward_scale*prediction
                  if method == 'hybrid' else reward_scale*prediction)
    if prediction.shape != np.asarray(batch.rewards).shape or not np.isfinite(prediction).all():
        raise FloatingPointError('Invalid candidate-network scores')
    return prediction


def double_targets(transitions, online, target, method, gamma, reward_scale):
    """Double-Q targets in scaled units, including exact terminal handling.

    Q target: r_t + gamma Q_target(s_next, argmax Q_online).
    Hybrid target: V(afterstate_t) is the *next* dispatch's selected total
    value, so current r_t is absent. Terminal afterstates target zero.
    """
    if method not in ('q', 'hybrid'):
        raise ValueError('method must be q or hybrid')
    values = np.asarray([t.reward/reward_scale if method == 'q' else 0.
                         for t in transitions], dtype=np.float32)
    active = [(i, t) for i, t in enumerate(transitions) if not t.done]
    if not active:
        return _tensor(values)
    next_features = [t.next_q_features if method == 'q' else t.next_value_features
                     for _, t in active]
    lengths = [len(f) for f in next_features]
    if any(n < 1 for n in lengths):
        raise ValueError('Nonterminal transitions need next candidates')
    with torch.no_grad():
        combined = _tensor(np.concatenate(next_features, axis=0))
        online_values = online(combined).detach().cpu().numpy()
        target_values = target(combined).detach().cpu().numpy()
    offset = 0
    for (index, transition), count in zip(active, lengths):
        exact = np.asarray(transition.next_rewards, dtype=np.float32)/reward_scale
        q_online = online_values[offset:offset+count]
        q_target = target_values[offset:offset+count]
        if transition.next_terminal:
            selection, evaluation = exact, exact
        elif method == 'hybrid':
            selection, evaluation = exact + gamma*q_online, exact + gamma*q_target
        else:
            selection, evaluation = q_online, q_target
        selected = int(np.argmax(selection))
        bootstrap = float(evaluation[selected])
        values[index] = values[index] + gamma*bootstrap if method == 'q' else bootstrap
        offset += count
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite Bellman targets')
    return _tensor(values)


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.01 * 2**attempt, .25))


def fit_candidate_value(scenario, pricing_policy, method, config, output_dir,
                        validation_seeds, rollout_config=None):
    """Fit and validation-select a network without accessing final test days.

    Directories must be new or empty. Snapshots and atomic progress JSON are
    retained after interruption; optimizer/replay resume is not implemented.
    """
    scenario, config = scenario.validate(), config.validate()
    if method not in ('q', 'hybrid'):
        raise ValueError('method must be q or hybrid')
    if getattr(pricing_policy, 'scenario', None) != scenario:
        raise ValueError('The frozen pricer must match the training scenario')
    rollout_config = (rollout_config or RolloutADPConfig()).validate(scenario)
    validation_seeds = _seed_list(validation_seeds)
    training_seeds = config.training_seeds()
    if set(training_seeds) & set(validation_seeds):
        raise ValueError('Training and validation days must be disjoint')
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Candidate training directory contains an earlier run')
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'checkpoints').mkdir()
    actor = deepcopy(pricing_policy)
    actor_identity = json.loads(json.dumps(actor.provenance(), sort_keys=True, allow_nan=False))
    _write(directory/'protocol.json', dict(schema=TRAINING_SCHEMA, method=method,
        config=asdict(config), rollout_config=asdict(rollout_config), training_seeds=list(training_seeds),
        validation_seeds=validation_seeds, pricing=actor_identity,
        objective='Original discounted operating profit; one gamma per physical period.',
        training_data='Executed dispatch transitions; frozen pricer; no rollout teacher.',
        selection='Highest mean validation profit; ties retain earlier checkpoint.',
        replay='Chosen actions only; fixed training-day budget shared between methods.',
        resume_supported=False, test_data_accessed=False))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.torch_threads)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            result = _fit(scenario, actor, method, config, directory,
                          validation_seeds, training_seeds, rollout_config, started)
    finally:
        torch.set_num_threads(old_threads)
    if json.loads(json.dumps(actor.provenance(), sort_keys=True, allow_nan=False)) != actor_identity:
        raise RuntimeError('Frozen pricing policy changed during training')
    return result


def _fit(scenario, actor, method, config, directory, validation_seeds,
         training_seeds, rollout_config, started):
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x43414E44]))
    probe = ChargingSimulator(scenario)
    pre, _ = probe.reset(seed=int(training_seeds[0]))
    levels, _ = _price_levels(pre, scenario, actor)
    post, _, _, _, _ = probe.quote_price_levels(levels)
    first = build_candidates(post, scenario, rollout_config, reward_scale=config.reward_scale)
    input_size = (first.q_features if method == 'q' else first.value_features).shape[1]
    network = ValueNetwork(input_size, config.hidden_size)
    target = deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    replay, loss_fn = deque(maxlen=config.replay_capacity), nn.SmoothL1Loss()
    gamma = scenario.discount_per_period
    steps = updates = 0
    # The dimension probe prepares one quote state but executes no dispatch.
    # Include its exact current/coordinate calculations in the cost ledger.
    candidate_outcomes = len(first.plans)
    coordinate_outcomes = first.coordinate_candidate_outcomes
    history, episode_rows, recent_losses = [], [], []
    best_score, best_snapshot = -np.inf, None
    validation_seconds = 0.

    def validate(completed):
        nonlocal best_score, best_snapshot, validation_seconds
        begin = time.perf_counter()
        policy = CandidateValuePolicy(scenario, actor, network, method=method,
            rollout_config=rollout_config, reward_scale=config.reward_scale)
        evaluation = evaluate_policy(scenario, validation_seeds, policy.prices,
            policy.dispatch, action_seed=config.action_seed, label=method,
            on_episode_start=policy.reset_episode)
        if evaluation['invalid_actions']:
            raise RuntimeError('Candidate validation produced invalid physical actions')
        snapshot = directory/'checkpoints'/f'episode_{completed:06d}.npz'
        policy.save_checkpoint(snapshot)
        score = float(evaluation['mean_discounted_profit'])
        if score > best_score:
            best_score, best_snapshot = score, snapshot
        elapsed = time.perf_counter()-begin
        validation_seconds += elapsed
        history.append(dict(episodes=completed, physical_steps=steps, optimizer_updates=updates,
            mean_validation_profit=score, validation_episode_profits=[
                dict(seed=e['seed'], discounted_profit=e['discounted_profit']) for e in evaluation['episodes']],
            validation_seconds=elapsed, mean_policy_decision_ms=evaluation['mean_policy_decision_ms'],
            checkpoint=str(snapshot.relative_to(directory)), selected=str(best_snapshot.relative_to(directory)),
            mean_recent_loss=float(np.mean(recent_losses)) if recent_losses else None,
            validation_candidate_outcomes=int(policy.statistics.get('current_plan_outcomes', 0)),
            validation_coordinate_outcomes=int(policy.statistics.get('coordinate_candidate_outcomes', 0))))
        recent_losses.clear()
        _write(directory/'training_history.json', history)
        print(f'{method}: {completed}/{config.episodes} episodes; '
              f'validation={score:.3f}, best={best_score:.3f}, steps={steps}, updates={updates}', flush=True)

    validate(0)
    for episode, seed in enumerate(training_seeds):
        sim = ChargingSimulator(scenario)
        pre, _ = sim.reset(seed=int(seed))
        levels, _ = _price_levels(pre, scenario, actor)
        post, price_reward, done, truncated, _ = sim.quote_price_levels(levels)
        if price_reward != 0. or done or truncated:
            raise RuntimeError('Unexpected pricing-stage transition')
        batch = build_candidates(post, scenario, rollout_config, reward_scale=config.reward_scale)
        candidate_outcomes += len(batch.plans)
        coordinate_outcomes += batch.coordinate_candidate_outcomes
        rewards, losses = [], []
        while True:
            epsilon = config.epsilon(steps, config.episodes*scenario.horizon)
            if batch.terminal:
                chosen = int(np.argmax(batch.rewards))
            elif rng.random() < epsilon:
                chosen = int(rng.integers(len(batch.plans)))
            else:
                chosen = int(np.argmax(candidate_scores(network, batch, method, gamma, config.reward_scale)))
            pre, reward, done, truncated, info = sim.dispatch(batch.plans[chosen])
            if truncated or info['invalid_actions']:
                raise RuntimeError('Candidate training produced an invalid physical transition')
            if not np.isclose(reward, batch.rewards[chosen], rtol=0., atol=1e-7):
                raise RuntimeError('Candidate current profit differs from simulator execution')
            following = None
            if not done:
                levels, _ = _price_levels(pre, scenario, actor)
                post, price_reward, price_done, price_truncated, _ = sim.quote_price_levels(levels)
                if price_reward != 0. or price_done or price_truncated:
                    raise RuntimeError('Unexpected next pricing-stage transition')
                following = build_candidates(post, scenario, rollout_config, reward_scale=config.reward_scale)
                candidate_outcomes += len(following.plans)
                coordinate_outcomes += following.coordinate_candidate_outcomes
            features = batch.q_features if method == 'q' else batch.value_features
            replay.append(CandidateTransition(np.asarray(features[chosen], dtype=np.float32).copy(),
                float(reward), bool(done),
                np.empty((0, batch.q_features.shape[1]), dtype=np.float32) if done else following.q_features.copy(),
                np.empty((0, batch.value_features.shape[1]), dtype=np.float32) if done else following.value_features.copy(),
                np.empty(0, dtype=np.float32) if done else following.rewards.copy(), False if done else following.terminal))
            steps += 1
            rewards.append(float(reward))
            if len(replay) >= config.batch_size and steps >= config.warmup_steps:
                for _ in range(config.updates_per_step):
                    indices = rng.choice(len(replay), size=config.batch_size, replace=False)
                    sampled = [replay[int(i)] for i in indices]
                    targets = double_targets(sampled, network, target, method, gamma, config.reward_scale)
                    predictions = network(_tensor(np.stack([t.features for t in sampled])))
                    loss = loss_fn(predictions, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Candidate training loss is nonfinite')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
                    optimizer.step()
                    updates += 1
                    recent_losses.append(float(loss.detach()))
                    losses.append(float(loss.detach()))
                    if updates % config.target_update == 0:
                        target.load_state_dict(network.state_dict())
            if done:
                break
            batch = following
        episode_rows.append(dict(episode=episode+1, environment_seed=int(seed),
            discounted_profit=float(sum(gamma**t*r for t, r in enumerate(rewards))),
            epsilon=epsilon, updates=updates, mean_loss=float(np.mean(losses)) if losses else None))
        _write(directory/'progress.json', dict(completed_episodes=episode+1, physical_steps=steps,
            optimizer_updates=updates, elapsed_seconds=time.perf_counter()-started,
            last_episode=episode_rows[-1], complete=False, resume_supported=False))
        if (episode+1) % config.eval_interval == 0 or episode+1 == config.episodes:
            validate(episode+1)
            _write(directory/'episode_returns.json', episode_rows)
    final_policy = CandidateValuePolicy(scenario, actor, network, method=method,
        rollout_config=rollout_config, reward_scale=config.reward_scale)
    final_policy.save_checkpoint(directory/'final_checkpoint.npz')
    shutil.copyfile(best_snapshot, directory/'best_checkpoint.npz')
    selected = CandidateValuePolicy.load_checkpoint(directory/'best_checkpoint.npz', actor)
    elapsed = time.perf_counter()-started
    summary = dict(schema=TRAINING_SCHEMA, method=method, config=asdict(config),
        training_seeds=list(training_seeds), validation_seeds=validation_seeds,
        selected_checkpoint='best_checkpoint.npz', selected_from=str(best_snapshot.relative_to(directory)),
        final_checkpoint='final_checkpoint.npz', best_validation_profit=float(best_score),
        training_seconds=elapsed-validation_seconds, validation_seconds=validation_seconds,
        elapsed_seconds=elapsed, environment_episodes=config.episodes, environment_periods=steps,
        optimizer_updates=updates, current_candidate_outcomes=candidate_outcomes,
        coordinate_candidate_outcomes=coordinate_outcomes, future_model_queries=0,
        validation_episodes=len(validation_seeds)*len(history), invalid_actions=0,
        validation_environment_periods=len(validation_seeds)*len(history)*scenario.horizon,
        validation_candidate_outcomes=sum(row['validation_candidate_outcomes'] for row in history),
        validation_coordinate_outcomes=sum(row['validation_coordinate_outcomes'] for row in history),
        probe_quote_preparations=1, probe_candidate_outcomes=len(first.plans),
        probe_coordinate_outcomes=first.coordinate_candidate_outcomes,
        all_losses_finite=True, test_data_accessed=False,
        reward='Original business profit; reward_scale changes units only.',
        hybrid_target='Next-dispatch selected total return; current reward excluded.' if method == 'hybrid' else None,
        physical_discount=gamma, replay_size=len(replay), initial_validation_included=True,
        uncertainty='Finite fixed training budget; no convergence or policy-improvement guarantee.')
    _write(directory/'summary.json', summary)
    _write(directory/'progress.json', dict(completed_episodes=config.episodes, physical_steps=steps,
        optimizer_updates=updates, elapsed_seconds=elapsed, selected_checkpoint='best_checkpoint.npz',
        complete=True, resume_supported=False))
    return selected, summary
