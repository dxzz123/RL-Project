"""On-policy SA-PPO–MAPPO with two stage-specific centralized critics."""
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ..config import Scenario
from ..simulator import ChargingSimulator, DispatchAction, Mode
from .networks import PricingActor, DispatchActor, ValueNetwork
from .returns import stage_gae, inventory_potential_rewards


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    updates: int = 10
    episodes_per_update: int = 4
    epochs: int = 4
    minibatch_size: int = 48  # Physical periods; each includes all M dispatch samples.
    hidden_size: int = 64
    learning_rate: float = 3e-4
    critic_learning_rate: float | None = None
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03  # Zero disables early stopping.
    eval_episodes: int = 4
    eval_interval: int = 0  # Zero evaluates only the initial and final policies.
    eval_seed_base: int = 100001  # Odd seeds never overlap even training seeds.
    eval_action_seed: int = 12345
    eval_policy: str = 'deterministic'
    discount_policy_loss: bool = True
    device: str = 'cpu'
    torch_threads: int = 1
    dispatch_final_activation: str = 'tanh'
    dispatch_agent_id: bool = False
    encoder_layer_norm: bool = False
    inventory_potential: float = 0.0  # Training-only potential; reported profit stays raw.
    dispatch_entropy_coef: float | None = None
    dispatch_entropy_final_coef: float | None = None
    dispatch_entropy_anneal_updates: int = 0
    separate_network_grad_clip: bool = False
    training_seed_base: int = 0

    def validate(self):
        for name in ('updates', 'episodes_per_update', 'epochs', 'minibatch_size',
                     'hidden_size', 'eval_episodes', 'torch_threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be an integer in [0, 2**31)')
        for name in ('learning_rate', 'clip_ratio', 'value_clip', 'gae_lambda', 'value_coef',
                     'entropy_coef', 'max_grad_norm', 'target_kl'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or not 0 < self.clip_ratio < 1:
            raise ValueError('learning_rate/max_grad_norm must be positive and clip_ratio in (0,1)')
        if self.gae_lambda > 1:
            raise ValueError('gae_lambda must be in [0,1]')
        if self.critic_learning_rate is not None and (not np.isfinite(self.critic_learning_rate)
                                                       or self.critic_learning_rate <= 0):
            raise ValueError('critic_learning_rate must be finite and positive')
        for name in ('eval_interval', 'eval_seed_base', 'eval_action_seed', 'training_seed_base',
                     'dispatch_entropy_anneal_updates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.eval_seed_base % 2 != 1:
            raise ValueError('eval_seed_base must be odd to avoid training seeds')
        if self.eval_policy not in ('deterministic', 'stochastic'):
            raise ValueError('eval_policy must be deterministic or stochastic')
        if (not isinstance(self.dispatch_final_activation, str)
                or self.dispatch_final_activation not in ('tanh', 'silu')):
            raise ValueError('dispatch_final_activation must be tanh or silu')
        if not isinstance(self.discount_policy_loss, bool):
            raise ValueError('discount_policy_loss must be boolean')
        for name in ('dispatch_agent_id', 'encoder_layer_norm', 'separate_network_grad_clip'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name in ('inventory_potential', 'dispatch_entropy_coef', 'dispatch_entropy_final_coef'):
            value = getattr(self, name)
            if value is None and name != 'inventory_potential':
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.dispatch_entropy_final_coef is not None and self.dispatch_entropy_anneal_updates < 1:
            raise ValueError('dispatch_entropy_anneal_updates must be positive for an entropy schedule')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError("device must be 'cpu' or 'cuda'; CPU is the tested portable default")
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available on this computer')
        return self


@dataclass
class Rollout:
    pre_obs: list
    post_obs: list
    local_obs: list  # Physical period first, MCV second.
    pricing_actions: np.ndarray
    dispatch_actions: np.ndarray
    energy_fractions: np.ndarray
    old_price_log_probs: np.ndarray
    old_dispatch_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    price_values: np.ndarray
    dispatch_values: np.ndarray
    next_price_values: np.ndarray
    episodes: list

    def __len__(self):
        return len(self.rewards)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _normalize(values, weights=None):
    if values.numel() < 2:
        return values
    weights = torch.ones_like(values) if weights is None else weights
    total = weights.sum()
    if total == 0:  # Very late active states can have underflowed gamma**t.
        return torch.zeros_like(values)
    mean = (weights*values).sum()/total
    std = ((weights*(values-mean).square()).sum()/total).sqrt()
    return (values - mean) / (std + 1e-8) if std > 1e-8 else values


class Trainer:
    """All actors remain frozen during collection; each update uses fresh episodes."""
    CHECKPOINT_VERSION = 1

    def __init__(self, scenario=None, config=None):
        self.scenario = (scenario or Scenario()).validate()
        self.config = (config or TrainingConfig()).validate()
        c = self.config
        self.device = torch.device(c.device)
        torch.set_num_threads(c.torch_threads)
        torch.manual_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.pricing_actor = PricingActor(self.scenario, c.hidden_size,
                                          layer_norm=c.encoder_layer_norm).to(self.device)
        self.dispatch_actor = DispatchActor(
            self.scenario, c.hidden_size,
            final_activation=c.dispatch_final_activation,
            include_agent_id=c.dispatch_agent_id, layer_norm=c.encoder_layer_norm).to(self.device)
        self.pricing_critic = ValueNetwork(self.scenario, c.hidden_size,
                                           layer_norm=c.encoder_layer_norm).to(self.device)
        self.dispatch_critic = ValueNetwork(self.scenario, c.hidden_size,
                                            layer_norm=c.encoder_layer_norm).to(self.device)
        self.actor_parameters = list(self.pricing_actor.parameters()) + list(self.dispatch_actor.parameters())
        self.critic_parameters = list(self.pricing_critic.parameters()) + list(self.dispatch_critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=c.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters,
                                                lr=c.critic_learning_rate or c.learning_rate, eps=1e-5)
        self.update_count = 0
        self.episode_count = 0
        self.environment_periods = 0

    def _modules(self):
        return {name: getattr(self, name) for name in
                ('pricing_actor', 'dispatch_actor', 'pricing_critic', 'dispatch_critic')}

    def _dispatch_commands(self, flat_actions, fractions):
        c = self.scenario
        scale = max(c.mobile_charge_kw, c.mobile_discharge_kw) * c.period_hours
        result = []
        for action, fraction in zip(flat_actions, fractions):
            mode, zone = divmod(int(action), c.n_zones)
            energy = float(fraction) * scale if mode in (Mode.RECHARGE, Mode.DISCHARGE) else 0.
            result.append(DispatchAction(Mode(mode), zone, energy))
        return result

    @staticmethod
    def _episode_result(seed, infos, rewards, gamma):
        total = sum(i['total_requests'] for i in infos)
        outside = sum(i['outside_requests'] for i in infos)
        served = sum(int(i['served_request_mask'].sum()) for i in infos)
        return {
            'seed': int(seed), 'profit': float(sum(rewards)),
            'discounted_profit': float(sum(gamma**t * r for t, r in enumerate(rewards))),
            'total_requests': total, 'outside_requests': outside, 'served_requests': served,
            'outside_option_share': outside / total if total else 0.,
            'served_request_share': served / total if total else 0.,
            'invalid_actions': sum(len(i['invalid_actions']) for i in infos),
            'reward_terms': {key: float(sum(i['reward_terms'][key] for i in infos))
                             for key in infos[0]['reward_terms']},
            'discounted_reward_terms': {key: float(sum(gamma**t*i['reward_terms'][key]
                                                       for t, i in enumerate(infos)))
                                       for key in infos[0]['reward_terms']},
            'action_counts': {mode.name: sum(int(np.count_nonzero(np.asarray(i['executed_modes']) == mode))
                                             for i in infos) for mode in Mode},
            'mobile_delivered_kwh': float(sum(np.sum(i['delivered_mobile_kwh']) for i in infos)),
            'unmet_kwh': float(sum(i['unmet_mobile_kwh']+i['unmet_fixed_kwh'] for i in infos)),
        }

    def collect_rollout(self):
        c = self.scenario
        data = {key: [] for key in Rollout.__dataclass_fields__}
        for _ in range(self.config.episodes_per_update):
            # Even seeds for training; fixed odd seeds for evaluation never overlap.
            seed = 2 * (self.config.training_seed_base + self.config.seed + self.episode_count)
            sim = ChargingSimulator(c)
            obs, _ = sim.reset(seed=seed)
            episode_rewards, infos = [], []
            while sim.stage != 2:
                pre = obs
                with torch.no_grad():
                    price = self.pricing_actor.sample([pre])
                    vp = float(self.pricing_critic([pre])[0].item())
                levels = price['actions'][0].cpu().numpy().copy()
                post, zero_reward, done, truncated, pricing_info = sim.quote_price_levels(levels)
                if zero_reward != 0 or done or truncated or pricing_info['discount'] != 1:
                    raise RuntimeError('Unexpected pricing transition contract')
                local = [sim.local_observation(m) for m in range(c.n_mcv)]
                with torch.no_grad():
                    dispatch = self.dispatch_actor.sample(local)
                    vd = float(self.dispatch_critic([post])[0].item())
                flat_actions = dispatch['actions'].cpu().numpy().copy()
                fractions = dispatch['energy_fraction'].cpu().numpy().copy()
                obs, reward, done, truncated, info = sim.dispatch(self._dispatch_commands(flat_actions, fractions))
                if truncated or info['invalid_actions']:
                    raise RuntimeError('Masked policy emitted invalid actions or unexpected truncation')
                with torch.no_grad():
                    next_vp = 0. if done else float(self.pricing_critic([obs])[0].item())
                data['pre_obs'].append(pre)
                data['post_obs'].append(post)
                data['local_obs'].extend(local)
                for key, value in (
                    ('pricing_actions', levels), ('dispatch_actions', flat_actions),
                    ('energy_fractions', fractions), ('old_price_log_probs', float(price['log_prob'][0].item())),
                    ('old_dispatch_log_probs', dispatch['log_prob'].cpu().numpy().copy()),
                    ('rewards', reward), ('dones', done), ('price_values', vp),
                    ('dispatch_values', vd), ('next_price_values', next_vp),
                ):
                    data[key].append(value)
                episode_rewards.append(reward)
                infos.append(info)
            data['episodes'].append(self._episode_result(seed, infos, episode_rewards, c.discount_per_period))
            self.episode_count += 1
        for key in ('pricing_actions', 'dispatch_actions', 'energy_fractions', 'old_price_log_probs',
                    'old_dispatch_log_probs', 'rewards', 'dones', 'price_values', 'dispatch_values', 'next_price_values'):
            data[key] = np.asarray(data[key])
        self.environment_periods += len(data['rewards'])
        return Rollout(**data)

    def update(self, rollout):
        c = self.config
        n, m = len(rollout), self.scenario.n_mcv
        if not n or len(rollout.local_obs) != n*m or not rollout.dones[-1]:
            raise ValueError('Update requires nonempty full episodes and one local observation per MCV/period')
        learning_rewards = inventory_potential_rewards(
            rollout.rewards, rollout.pre_obs, rollout.dones,
            self.scenario.discount_per_period, c.inventory_potential)
        targets = stage_gae(learning_rewards, rollout.price_values, rollout.dispatch_values,
                            rollout.next_price_values, rollout.dones, self.scenario.discount_per_period, c.gae_lambda)
        dispatch_entropy_coef = c.entropy_coef if c.dispatch_entropy_coef is None else c.dispatch_entropy_coef
        if c.dispatch_entropy_final_coef is not None:
            progress = min(1., self.update_count / max(1, c.dispatch_entropy_anneal_updates-1))
            dispatch_entropy_coef += progress*(c.dispatch_entropy_final_coef-dispatch_entropy_coef)
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        active = tensor([bool(np.asarray(o['pricing_mask']).any()) for o in rollout.pre_obs], torch.bool)
        # The finite-horizon objective is E[sum_t gamma**t r_t]. GAE already
        # discounts future rewards; this separate outer weight discounts occupancy.
        weights = tensor([self.scenario.discount_per_period**int(np.asarray(o['period']).item())
                          if c.discount_policy_loss else 1. for o in rollout.pre_obs])
        price_adv = tensor(targets['pricing_advantages'])
        price_adv = price_adv.clone()
        if active.any():
            price_adv[active] = _normalize(price_adv[active], weights[active])
        price_adv[~active] = 0.
        dispatch_adv = _normalize(tensor(targets['dispatch_advantages']), weights)
        vp_targets, vd_targets = tensor(targets['pricing_returns']), tensor(targets['dispatch_returns'])
        old_vp, old_vd = tensor(rollout.price_values), tensor(rollout.dispatch_values)
        old_p_logp, old_d_logp = tensor(rollout.old_price_log_probs), tensor(rollout.old_dispatch_log_probs)
        price_actions = tensor(rollout.pricing_actions, torch.long)
        dispatch_actions = tensor(rollout.dispatch_actions, torch.long)
        fractions = tensor(rollout.energy_fractions)
        before = {name: torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                  for name, model in self._modules().items()}
        records, stopped, optimizer_steps = [], False, 0
        for _ in range(c.epochs):
            for start in range(0, n, c.minibatch_size):
                if start == 0:
                    order = self.rng.permutation(n)
                indices = order[start:start+c.minibatch_size]
                ix = tensor(indices, torch.long)
                price_obs = [rollout.pre_obs[int(i)] for i in indices]
                post_obs = [rollout.post_obs[int(i)] for i in indices]
                local_obs = [rollout.local_obs[int(i)*m+j] for i in indices for j in range(m)]
                p = self.pricing_actor.evaluate_actions(price_obs, price_actions[ix])
                d = self.dispatch_actor.evaluate_actions(local_obs, dispatch_actions[ix].reshape(-1),
                                                          fractions[ix].reshape(-1))
                p_diff = p['log_prob'] - old_p_logp[ix]
                d_diff = d['log_prob'] - old_d_logp[ix].reshape(-1)
                p_ratio, d_ratio = p_diff.exp(), d_diff.exp()
                valid_price = active[ix]
                p_kl = ((p_ratio - 1) - p_diff)[valid_price].mean() if valid_price.any() else p_diff.sum()*0.
                d_kl = ((d_ratio - 1) - d_diff).mean()
                if c.target_kl and max(float(p_kl.detach()), float(d_kl.detach())) > c.target_kl:
                    stopped = True
                    break
                p_surrogate = torch.minimum(p_ratio*price_adv[ix],
                    p_ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*price_adv[ix])
                p_loss = -(p_surrogate*weights[ix])[valid_price].mean() if valid_price.any() else p['log_prob'].sum()*0.
                shared_adv = dispatch_adv[ix, None].expand(-1, m).reshape(-1)
                dispatch_weights = weights[ix, None].expand(-1, m).reshape(-1)
                d_loss = -(torch.minimum(d_ratio*shared_adv,
                    d_ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*shared_adv)*dispatch_weights).mean()
                p_entropy = (p['entropy']*weights[ix])[valid_price].mean() if valid_price.any() else p['entropy'].sum()*0.
                d_entropy = (d['entropy']*dispatch_weights).mean()
                vp, vd = self.pricing_critic(price_obs), self.dispatch_critic(post_obs)
                def value_loss(value, old, target):
                    errors = (value-target).square()
                    if c.value_clip:
                        clipped = old + (value-old).clamp(-c.value_clip, c.value_clip)
                        errors = torch.maximum(errors, (clipped-target).square())
                    return .5*errors.mean()
                vp_loss = value_loss(vp, old_vp[ix], vp_targets[ix])
                vd_loss = value_loss(vd, old_vd[ix], vd_targets[ix])
                if c.dispatch_entropy_coef is None and c.dispatch_entropy_final_coef is None:
                    actor_loss = p_loss + d_loss - c.entropy_coef*(p_entropy+d_entropy)
                else:
                    actor_loss = p_loss + d_loss - c.entropy_coef*p_entropy - dispatch_entropy_coef*d_entropy
                critic_loss = c.value_coef*(vp_loss+vd_loss)
                loss = actor_loss + critic_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss; no optimizer step applied')
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if c.separate_network_grad_clip:
                    norms = {name: nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                          error_if_nonfinite=True)
                             for name, model in self._modules().items()}
                    actor_norm = torch.hypot(norms['pricing_actor'], norms['dispatch_actor'])
                    critic_norm = torch.hypot(norms['pricing_critic'], norms['dispatch_critic'])
                else:
                    actor_norm = nn.utils.clip_grad_norm_(self.actor_parameters, c.max_grad_norm, error_if_nonfinite=True)
                    critic_norm = nn.utils.clip_grad_norm_(self.critic_parameters, c.max_grad_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                optimizer_steps += 1
                records.append({
                    'pricing_policy_loss': float(p_loss.detach()), 'dispatch_policy_loss': float(d_loss.detach()),
                    'pricing_value_loss': float(vp_loss.detach()), 'dispatch_value_loss': float(vd_loss.detach()),
                    'pricing_entropy': float(p_entropy.detach()), 'dispatch_entropy': float(d_entropy.detach()),
                    'pricing_approx_kl': float(p_kl.detach()), 'dispatch_approx_kl': float(d_kl.detach()),
                    'actor_grad_norm': float(actor_norm), 'critic_grad_norm': float(critic_norm),
                    'pricing_clip_fraction': float(((p_ratio.detach()-1).abs()[valid_price] > c.clip_ratio).float().mean()) if valid_price.any() else 0.,
                    'dispatch_clip_fraction': float(((d_ratio.detach()-1).abs() > c.clip_ratio).float().mean()),
                })
            if stopped:
                break
        if not records:
            raise RuntimeError('No optimizer updates; rollout is stale or target KL is too small')
        self.update_count += 1
        result = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
        result.update(update=self.update_count, episodes=self.episode_count,
                      environment_periods=self.environment_periods, optimizer_steps=optimizer_steps,
                      early_stopped=int(stopped),
                      mean_training_profit=float(np.mean([e['profit'] for e in rollout.episodes])),
                      mean_training_discounted_profit=float(np.mean([e['discounted_profit'] for e in rollout.episodes])),
                      dispatch_entropy_coefficient=float(dispatch_entropy_coef),
                      mean_learning_reward=float(np.mean(learning_rewards)),
                      mean_outside_share=float(np.mean([e['outside_option_share'] for e in rollout.episodes])))
        for name, model in self._modules().items():
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            if not torch.isfinite(after).all():
                raise FloatingPointError(f'Nonfinite {name} parameters')
            result[f'{name}_parameter_delta'] = float(torch.linalg.vector_norm(after-before[name]).item())
        with torch.no_grad():
            for name, model, observations, target in (
                ('pricing', self.pricing_critic, rollout.pre_obs, vp_targets),
                ('dispatch', self.dispatch_critic, rollout.post_obs, vd_targets),
            ):
                predicted = model(observations)
                variance = target.var(unbiased=False)
                result[f'{name}_explained_variance'] = float(
                    1-(target-predicted).var(unbiased=False)/variance) if variance > 1e-8 else 0.
                result[f'{name}_value_rmse'] = float((target-predicted).square().mean().sqrt())
        return result

    @torch.no_grad()
    def evaluate(self, seeds, deterministic=True, action_seed=None):
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError('Evaluation requires at least one seed')
        action_seed = self.config.eval_action_seed if action_seed is None else int(action_seed)
        episodes = []
        # torch.manual_seed also seeds CUDA; preserve every initialized device.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            for seed in seeds:
                torch.manual_seed((action_seed + seed) % (2**63-1))
                sim = ChargingSimulator(self.scenario)
                obs, _ = sim.reset(seed=seed)
                rewards, infos = [], []
                while sim.stage != 2:
                    prices = self.pricing_actor.sample([obs], deterministic=deterministic)
                    sim.quote_price_levels(prices['actions'][0].cpu().numpy())
                    local = [sim.local_observation(m) for m in range(self.scenario.n_mcv)]
                    action = self.dispatch_actor.sample(local, deterministic=deterministic)
                    obs, reward, _, _, info = sim.dispatch(self._dispatch_commands(
                        action['actions'].cpu().numpy(), action['energy_fraction'].cpu().numpy()))
                    rewards.append(reward)
                    infos.append(info)
                episodes.append(self._episode_result(seed, infos, rewards, self.scenario.discount_per_period))
        profits = np.asarray([e['discounted_profit'] for e in episodes])
        sem = float(profits.std(ddof=1)/np.sqrt(len(profits))) if len(profits) > 1 else None
        return {'deterministic': bool(deterministic), 'action_seed': action_seed,
                'seeds': seeds, 'episodes': episodes,
                'mean_profit': float(np.mean([e['profit'] for e in episodes])),
                'mean_discounted_profit': float(profits.mean()),
                'std_discounted_profit': float(profits.std()),
                'standard_error_discounted_profit': sem,
                'confidence_interval_95': [float(profits.mean()-1.96*sem), float(profits.mean()+1.96*sem)]
                                          if sem is not None else None,
                'mean_outside_option_share': float(np.mean([e['outside_option_share'] for e in episodes])),
                'mean_served_request_share': float(np.mean([e['served_request_share'] for e in episodes])),
                'mean_mobile_delivered_kwh': float(np.mean([e['mobile_delivered_kwh'] for e in episodes])),
                'mean_unmet_kwh': float(np.mean([e['unmet_kwh'] for e in episodes])),
                'mean_reward_terms': {key: float(np.mean([e['reward_terms'][key] for e in episodes]))
                                      for key in episodes[0]['reward_terms']},
                'mean_discounted_reward_terms': {
                    key: float(np.mean([e['discounted_reward_terms'][key] for e in episodes]))
                    for key in episodes[0]['discounted_reward_terms']},
                'action_counts': {mode.name: sum(e['action_counts'][mode.name] for e in episodes)
                                  for mode in Mode},
                'invalid_actions': sum(e['invalid_actions'] for e in episodes)}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': self.CHECKPOINT_VERSION,
            'scenario': asdict(self.scenario), 'training_config': asdict(self.config),
            'models': {name: model.state_dict() for name, model in self._modules().items()},
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count, 'episode_count': self.episode_count,
            'environment_periods': self.environment_periods,
            'numpy_rng': self.rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
        }
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != cls.CHECKPOINT_VERSION:
            raise ValueError('Unsupported checkpoint format')
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        scenario = Scenario(**{k: tuples(v) for k, v in payload['scenario'].items()}).validate()
        settings = dict(payload['training_config'], device=device)
        # Preserve the old objective approximation and old evaluation seeds on resume.
        settings.setdefault('discount_policy_loss', False)
        settings.setdefault('eval_seed_base', 2*settings['seed']+1)
        settings.setdefault('dispatch_final_activation', 'tanh')
        trainer = cls(scenario, TrainingConfig(**settings))
        for name, model in trainer._modules().items():
            model.load_state_dict(payload['models'][name])
        trainer.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        trainer.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        trainer.update_count = int(payload['update_count'])
        trainer.episode_count = int(payload['episode_count'])
        trainer.environment_periods = int(payload['environment_periods'])
        trainer.rng.bit_generator.state = payload['numpy_rng']
        torch.set_rng_state(payload['torch_rng'].cpu())
        if trainer.device.type == 'cuda' and payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        return trainer

    def train(self, output_dir, updates=None):
        count = self.config.updates if updates is None else updates
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('updates must be a positive integer')
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        planned = ('training_metrics.csv', 'validation_metrics.csv', 'evaluation_before.json',
                   'evaluation_after.json', 'validation_before.json', 'validation_after.json',
                   'checkpoint.pt', 'best_checkpoint.pt',
                   'best_validation.json', 'summary.json', 'scenario.json', 'training_config.json')
        if any((output/name).exists() for name in planned):
            raise FileExistsError('Output contains a previous run; choose a new output directory')
        started = time.monotonic()
        self.scenario.to_json(output/'scenario.json')
        _write_json(output/'training_config.json', dict(asdict(self.config), updates=count))
        eval_seeds = [self.config.eval_seed_base+2*i for i in range(self.config.eval_episodes)]
        history, validation_rows = [], []
        best_score, best_update = -float('inf'), self.update_count
        invalid_evaluation_actions = 0

        def validate():
            nonlocal best_score, best_update, invalid_evaluation_actions
            results = {}
            for policy in ('deterministic', 'stochastic'):
                result = self.evaluate(eval_seeds, deterministic=policy == 'deterministic')
                results[policy] = result
                invalid_evaluation_actions += result['invalid_actions']
                row = {'update': self.update_count, 'environment_periods': self.environment_periods,
                       'policy': policy, 'mean_discounted_profit': result['mean_discounted_profit'],
                       'standard_error_discounted_profit': result['standard_error_discounted_profit'],
                       'mean_outside_option_share': result['mean_outside_option_share'],
                       'mean_served_request_share': result['mean_served_request_share'],
                       'mean_mobile_delivered_kwh': result['mean_mobile_delivered_kwh'],
                       'mean_unmet_kwh': result['mean_unmet_kwh'],
                       'invalid_actions': result['invalid_actions']}
                row.update(result['mean_discounted_reward_terms'])
                validation_rows.append(row)
            selected = results[self.config.eval_policy]
            if selected['mean_discounted_profit'] > best_score:
                best_score, best_update = selected['mean_discounted_profit'], self.update_count
                self.save_checkpoint(output/'best_checkpoint.pt')
                _write_json(output/'best_validation.json', {
                    'update': best_update, 'policy': self.config.eval_policy,
                    'mean_discounted_profit': best_score, 'evaluation': selected})
            with (output/'validation_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
                writer.writeheader()
                writer.writerows(validation_rows)
            print(f"validation update={self.update_count} "
                  f"det={results['deterministic']['mean_discounted_profit']:.3f} "
                  f"stoch={results['stochastic']['mean_discounted_profit']:.3f} "
                  f"best({self.config.eval_policy})={best_score:.3f}@{best_update}", flush=True)
            return results

        initial = validate()
        before = initial[self.config.eval_policy]
        _write_json(output/'evaluation_before.json', before)
        _write_json(output/'validation_before.json', initial)
        with (output/'training_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for step in range(count):
                metrics = self.update(self.collect_rollout())
                history.append(metrics)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                stream.flush()
                self.save_checkpoint(output/'checkpoint.pt')
                print(f"update={self.update_count} periods={self.environment_periods} "
                      f"train_discounted_profit={metrics['mean_training_discounted_profit']:.3f} "
                      f"KL(P/D)={metrics['pricing_approx_kl']:.5f}/{metrics['dispatch_approx_kl']:.5f}", flush=True)
                if step == count-1 or (self.config.eval_interval and self.update_count % self.config.eval_interval == 0):
                    final = validate()
        after = final[self.config.eval_policy]
        _write_json(output/'evaluation_after.json', after)
        _write_json(output/'validation_after.json', final)
        # Loading a trainer initializes and restores global Torch RNG; isolate this check.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            restored = self.load_checkpoint(output/'checkpoint.pt', device=self.config.device)
            reloaded = restored.evaluate(eval_seeds, deterministic=self.config.eval_policy == 'deterministic')
        reload_matches = after == reloaded
        if not reload_matches:
            raise AssertionError('Checkpoint reload changed evaluation')
        summary = {
            'algorithm': 'SA-PPO-MAPPO', 'updates_this_run': count,
            'total_updates': self.update_count, 'training_episodes': self.episode_count,
            'training_physical_periods': self.environment_periods,
            'training_stage_transitions': 2*self.environment_periods,
            'eval_seeds': eval_seeds, 'eval_policy': self.config.eval_policy,
            'eval_before_mean_discounted_profit': before['mean_discounted_profit'],
            'eval_after_mean_discounted_profit': after['mean_discounted_profit'],
            'best_update': best_update, 'best_validation_mean_discounted_profit': best_score,
            'invalid_evaluation_actions': invalid_evaluation_actions,
            'all_metrics_finite': bool(all(np.isfinite(v) for row in history for v in row.values())),
            'checkpoint_reload_matches': reload_matches,
            'parameter_change_by_module': {name: float(sum(row[f'{name}_parameter_delta'] for row in history))
                                           for name in self._modules()},
            'elapsed_seconds': time.monotonic()-started,
            'versions': {'torch': str(torch.__version__), 'numpy': str(np.__version__)},
            'scope': 'Checkpoint selection uses validation only. Assess convergence across seeds and on independent test episodes.',
        }
        _write_json(output/'summary.json', summary)
        return summary
