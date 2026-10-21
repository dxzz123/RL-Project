"""Joint two-stage PPO matching the interleaved equations in MDP_RL (2).

This is an isolated controller. The older local MAPPO and fitted-value methods
are unchanged. Network sizes, nonlinearities, physical quantity conventions,
and optimizer safeguards are implementation choices recorded in each run.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
from enum import IntEnum
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..config import Scenario
from ..simulator import ChargingSimulator, DispatchAction, Mode
from ..benchmarks.evaluation import evaluate_policy, episode_summary


ALGORITHM = 'paper_joint_ppo_v1'


class PaperOperation(IntEnum):
    SERVICE = 0
    IDLE = 1
    CHARGE = 2
    DISCHARGE = 3


def interleaved_gae(pricing_rewards, dispatch_rewards, price_values, dispatch_values,
                    next_price_values, dones, gamma, gae_lambda):
    """Equations 4--7: lambda at both substages, gamma only between periods."""
    rp, rd, vp, vd, next_vp = [np.asarray(x, dtype=np.float64) for x in
                              (pricing_rewards, dispatch_rewards, price_values,
                               dispatch_values, next_price_values)]
    terminal = np.asarray(dones)
    if (rd.ndim != 1 or not len(rd) or any(a.shape != rd.shape or not np.isfinite(a).all()
                            for a in (rp, rd, vp, vd, next_vp))
            or terminal.shape != rd.shape or not np.isin(terminal, [0, 1]).all()):
        raise ValueError('Interleaved GAE requires matching finite vectors and binary dones')
    if (isinstance(gamma, (bool, np.bool_)) or isinstance(gae_lambda, (bool, np.bool_))
            or not np.isfinite(gamma) or not np.isfinite(gae_lambda)
            or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1):
        raise ValueError('gamma must be in (0,1] and gae_lambda in [0,1]')
    ap, ad = np.zeros_like(rd), np.zeros_like(rd)
    following = 0.
    for t in range(len(rd)-1, -1, -1):
        continuation = 1. - float(terminal[t])
        delta_p = rp[t] + vd[t] - vp[t]
        delta_d = rd[t] + gamma * continuation * next_vp[t] - vd[t]
        ad[t] = delta_d + gamma * gae_lambda * continuation * following
        ap[t] = delta_p + gae_lambda * ad[t]
        following = ap[t]
    return dict(pricing_advantages=ap, dispatch_advantages=ad,
                pricing_returns=ap+vp, dispatch_returns=ad+vd)


def critic_targets(gae, old_price_values, old_dispatch_values, dispatch_critic_target='dispatch'):
    """Use the paper's A2 target, or an optional pricing-advantage A1 ablation."""
    if dispatch_critic_target not in ('dispatch', 'pricing_advantage'):
        raise ValueError('dispatch_critic_target must be dispatch or pricing_advantage')
    ap = np.asarray(gae['pricing_advantages'], dtype=float)
    ad = np.asarray(gae['dispatch_advantages'], dtype=float)
    vp, vd = np.asarray(old_price_values, dtype=float), np.asarray(old_dispatch_values, dtype=float)
    if (ap.ndim != 1 or not len(ap)
            or any(a.shape != ap.shape or not np.isfinite(a).all() for a in (ap, ad, vp, vd))):
        raise ValueError('Critic targets require matching finite advantage and value arrays')
    return dict(pricing_returns=ap+vp,
                dispatch_returns=(ad if dispatch_critic_target == 'dispatch' else ap)+vd)


def joint_ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio, entropies=None):
    """Clip one product ratio per fleet action, never separate vehicle ratios."""
    if (not all(torch.is_tensor(x) for x in (new_log_probs, old_log_probs, advantages))
            or new_log_probs.ndim != 2 or not len(new_log_probs)
            or old_log_probs.shape != new_log_probs.shape
            or advantages.shape != new_log_probs.shape[:1]
            or new_log_probs.shape[1] < 1 or not 0 < clip_ratio < 1):
        raise ValueError('Joint PPO expects log probabilities [batch,agents] and advantages [batch]')
    if not all(bool(torch.isfinite(x).all()) for x in (new_log_probs, old_log_probs, advantages)):
        raise ValueError('Joint PPO log probabilities and advantages must be finite')
    if entropies is not None and entropies.shape != new_log_probs.shape:
        raise ValueError('Agent entropies must match agent log probabilities')
    if entropies is not None and not bool(torch.isfinite(entropies).all()):
        raise ValueError('Agent entropies must be finite')
    log_ratio = (new_log_probs-old_log_probs).sum(dim=-1)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio*advantages,
                              ratio.clamp(1-clip_ratio, 1+clip_ratio)*advantages)
    entropy = (entropies.sum(dim=-1).mean() if entropies is not None
               else new_log_probs.sum()*0.)
    return dict(loss=-surrogate.mean(), entropy=entropy,
                approx_kl=((ratio-1)-log_ratio).mean(),
                clip_fraction=((ratio-1).abs() > clip_ratio).float().mean(),
                log_ratio=log_ratio, ratio=ratio, surrogate=surrogate)


def _fleet_pricing_mask(obs, scenario):
    locations, energy = np.asarray(obs['locations'], dtype=int), np.asarray(obs['energy_kwh'], dtype=float)
    distance = scenario.distances[locations]
    hours = scenario.period_hours-distance/scenario.speed_kmph
    capacity = np.maximum(0., np.minimum(
        (energy[:, None]-distance*scenario.travel_kwh_per_km-scenario.reserve_kwh)
        * scenario.mobile_service_efficiency, hours*scenario.mobile_service_kw))
    available = (capacity+1e-9 >= scenario.request_min_kwh).any(axis=0)
    return np.repeat(available[:, None], len(scenario.price_levels), axis=1)


def paper_pricing_observation(obs, scenario):
    """Project S_t to prescribed rates and fleet/prices, hiding realized arrivals.

    Availability masks are recomputed from fleet physics, so neither an input
    mask nor FCS waiting time can accidentally reveal latent current requests.
    The simulator may pre-sample those requests internally; they are unobserved
    by both the pricing actor and pricing critic in this controller.
    """
    if int(obs['stage']) != 0:
        raise ValueError('Paper pricing requires the before-choice stage')
    keys = ('period', 'locations', 'energy_kwh', 'arrival_rates', 'fcs_retail_prices',
            'fcs_discharge_prices', 'grid_buy_price')
    result = {name: np.asarray(obs[name]).copy() for name in keys}
    result['stage'] = 0
    result['pricing_mask'] = _fleet_pricing_mask(obs, scenario)
    return result


def pricing_features(obs, scenario):
    """Fixed-size normalized features with no current realized request data."""
    o = paper_pricing_observation(obs, scenario)
    c = scenario
    count = max(1., float(np.max(c.resolved_arrival_rates)))
    price = max(1., c.price_max, float(np.max(c.resolved_fcs_charge_price_profile)),
                float(np.max(c.resolved_fcs_discharge_price_profile)), max(c.grid_buy_per_kwh))
    parts = [np.asarray(o['period']).reshape(-1)/c.horizon,
             np.eye(c.n_zones)[np.asarray(o['locations'], dtype=int)].reshape(-1),
             np.asarray(o['energy_kwh'])/c.battery_capacity_kwh,
             np.asarray(o['arrival_rates'])/count,
             np.asarray(o['fcs_retail_prices'])/price,
             np.asarray(o['fcs_discharge_prices'])/price,
             np.asarray(o['grid_buy_price'])/price]
    result = np.concatenate(parts).astype(np.float32)
    expected = 1+c.n_mcv*(c.n_zones+1)+c.n_zones+2*c.n_fcs+1
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError('Malformed or nonfinite paper pricing observation')
    return result


def paper_dispatch_mask(obs, scenario):
    """Four operations over mobile-request nodes union FCS nodes.

    IDLE at the current location means WAIT. IDLE at another admissible node
    means travel without service, using the existing REPOSITION transition.
    The current location is always an idle fallback, even if no demand or FCS
    is present there. These physical interpretations are not specified by the
    abstract paper and are recorded as implementation conventions.
    """
    if int(obs['stage']) != 1:
        raise ValueError('Paper dispatch requires the post-choice stage')
    c = scenario
    original = np.asarray(obs['action_mask'], dtype=bool)
    if original.shape != (c.n_mcv, len(Mode), c.n_zones):
        raise ValueError('Dispatch observation has an incompatible action mask')
    details = np.asarray(obs['request_details'])
    nodes = np.zeros(c.n_zones, dtype=bool)
    nodes[list(c.fcs_zones)] = True
    if len(details):
        nodes[details[details[:, 2] == 0, 0].astype(int)] = True
    masks = np.zeros((c.n_mcv, 4, c.n_zones), dtype=bool)
    masks[:, PaperOperation.SERVICE] = original[:, Mode.SERVE] & nodes
    masks[:, PaperOperation.IDLE] = original[:, Mode.REPOSITION] & nodes
    masks[:, PaperOperation.CHARGE] = original[:, Mode.RECHARGE] & nodes
    masks[:, PaperOperation.DISCHARGE] = original[:, Mode.DISCHARGE] & nodes
    for m, location in enumerate(np.asarray(obs['locations'], dtype=int)):
        masks[m, PaperOperation.IDLE, location] = True
    return masks


def paper_dispatch_commands(obs, flat_actions, scenario):
    """Translate discrete operations; exchange requests use the feasible maximum."""
    actions = np.asarray(flat_actions)
    mask = paper_dispatch_mask(obs, scenario).reshape(scenario.n_mcv, -1)
    if (actions.shape != (scenario.n_mcv,) or actions.dtype.kind not in 'iu'
            or (actions < 0).any() or (actions >= 4*scenario.n_zones).any()
            or not mask[np.arange(scenario.n_mcv), actions].all()):
        raise ValueError('Paper policy must emit one feasible discrete action per vehicle')
    commands = []
    for m, action in enumerate(actions):
        operation, node = divmod(int(action), scenario.n_zones)
        location = int(obs['locations'][m])
        if operation == PaperOperation.SERVICE:
            mode = Mode.SERVE
        elif operation == PaperOperation.IDLE:
            mode = Mode.WAIT if node == location else Mode.REPOSITION
        else:
            mode = Mode.RECHARGE if operation == PaperOperation.CHARGE else Mode.DISCHARGE
        amount = 0.
        if mode in (Mode.RECHARGE, Mode.DISCHARGE):
            distance = scenario.distances[location, node]
            hours = max(0., scenario.period_hours-distance/scenario.speed_kmph)
            after_travel = float(obs['energy_kwh'][m])-distance*scenario.travel_kwh_per_km
            amount = (min(hours*scenario.mobile_charge_kw,
                          max(0., scenario.battery_capacity_kwh-after_travel)/scenario.charge_efficiency)
                      if mode == Mode.RECHARGE else
                      min(hours*scenario.mobile_discharge_kw,
                          max(0., after_travel-scenario.reserve_kwh)*scenario.discharge_efficiency))
        commands.append(DispatchAction(mode, node, float(amount)))
    return commands


def _linear(inputs, outputs, gain=np.sqrt(2.)):
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    nn.init.zeros_(layer.bias)
    return layer


def _body(inputs, hidden):
    return nn.Sequential(_linear(inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         _linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())


class _PaperPricingEncoder(nn.Module):
    def __init__(self, scenario, hidden):
        super().__init__()
        self.scenario = scenario
        c = scenario
        self.body = _body(1+c.n_mcv*(c.n_zones+1)+c.n_zones+2*c.n_fcs+1, hidden)

    def forward(self, observations):
        reference = next(self.parameters())
        features = torch.as_tensor(np.stack([pricing_features(o, self.scenario) for o in observations]),
                                   dtype=reference.dtype, device=reference.device)
        return self.body(features)


class _PaperPostEncoder(nn.Module):
    def __init__(self, scenario, hidden):
        super().__init__()
        self.scenario, self.hidden = scenario, hidden
        c = scenario
        macro = 1+c.n_mcv*(c.n_zones+1)+c.n_zones+2*c.n_fcs+1
        self.request_size = c.n_zones+1+(c.n_fcs+3)+c.n_fcs+1
        self.request_encoder = _body(self.request_size, hidden)
        dense = macro+c.n_zones+4*c.n_zones*c.n_mcv+c.n_zones+c.n_fcs+1
        self.body = _body(dense+(c.n_zones+1)*hidden, hidden)

    def forward(self, observations):
        c = self.scenario
        reference = next(self.parameters())
        def tensor(value):
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        count = max(1., float(np.max(c.resolved_arrival_rates)))
        dense, details = [], []
        for obs in observations:
            # The macro part is reused with its stage flag changed locally;
            # current request information enters only the post-choice encoder.
            macro = dict(obs, stage=0)
            dense.append(np.concatenate((pricing_features(macro, c),
                np.asarray(obs['mobile_prices'])/max(1., c.price_max),
                paper_dispatch_mask(obs, c).reshape(-1),
                np.asarray(obs['mobile_requested_kwh'])/(count*c.request_max_kwh),
                np.asarray(obs['fixed_requested_kwh'])/(count*c.request_max_kwh),
                np.asarray(obs['outside_requests'])/(count*c.n_zones))))
            details.append(np.asarray(obs['request_details']))
        width = max(1, max(len(rows) for rows in details))
        requests = np.zeros((len(details), width, self.request_size), dtype=np.float32)
        valid = np.zeros((len(details), width), dtype=np.float32)
        origins = np.zeros((len(details), width, c.n_zones), dtype=np.float32)
        for index, rows in enumerate(details):
            n = len(rows)
            if not n:
                continue
            origin = np.eye(c.n_zones)[rows[:, 0].astype(int)]
            choice = np.eye(c.n_fcs+3)[rows[:, 2].astype(int)+1]
            requests[index, :n] = np.concatenate((origin, rows[:, 1:2]/c.request_max_kwh,
                choice, rows[:, 3:], (np.arange(n)/max(1, n-1))[:, None]), axis=1)
            valid[index, :n] = 1.
            origins[index, :n] = origin
        valid_tensor = tensor(valid)
        encoded = self.request_encoder(tensor(requests))*valid_tensor[..., None]
        mean = encoded.sum(dim=1)/valid_tensor.sum(dim=1, keepdim=True).clamp_min(1)
        by_zone = torch.bmm(tensor(origins).transpose(1, 2), encoded)/count
        return self.body(torch.cat((tensor(np.stack(dense)), mean, by_zone.flatten(1)), dim=-1))


class _PaperPricingActor(nn.Module):
    def __init__(self, scenario, hidden_size=64):
        super().__init__()
        self.scenario = scenario
        self.encoder = _PaperPricingEncoder(scenario, hidden_size)
        self.head = _linear(hidden_size, scenario.n_zones*len(scenario.price_levels), .01)

    def _distribution(self, observations):
        c = self.scenario
        logits = self.head(self.encoder(observations)).reshape(-1, c.n_zones, len(c.price_levels))
        masks = torch.as_tensor(np.stack([_fleet_pricing_mask(o, c) for o in observations]),
                                dtype=torch.bool, device=logits.device)
        active = masks.any(dim=-1)
        safe = masks.clone()
        safe[..., 0] |= ~active
        logits = torch.where(active[..., None], logits, torch.zeros_like(logits))
        return Categorical(logits=logits.masked_fill(~safe, -torch.inf)), active

    def sample(self, observations, deterministic=False):
        distribution, active = self._distribution(observations)
        actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return dict(actions=actions, log_prob=distribution.log_prob(actions).sum(-1),
                    entropy=distribution.entropy().sum(-1), active=active.any(-1))

    def evaluate_actions(self, observations, actions):
        distribution, active = self._distribution(observations)
        actions = torch.as_tensor(actions, dtype=torch.long, device=distribution.logits.device)
        if actions.shape != active.shape:
            raise ValueError('Pricing actions must have shape [batch,zones]')
        return dict(log_prob=distribution.log_prob(actions).sum(-1),
                    entropy=distribution.entropy().sum(-1), active=active.any(-1))


class _PaperDispatchActor(nn.Module):
    def __init__(self, scenario, hidden_size=64):
        super().__init__()
        self.scenario = scenario
        self.encoder = _PaperPostEncoder(scenario, hidden_size)
        self.head = _linear(hidden_size, scenario.n_mcv*4*scenario.n_zones, .01)

    def _distribution(self, observations):
        c = self.scenario
        logits = self.head(self.encoder(observations)).reshape(-1, c.n_mcv, 4*c.n_zones)
        masks = torch.as_tensor(np.stack([paper_dispatch_mask(o, c).reshape(c.n_mcv, -1)
                                          for o in observations]), dtype=torch.bool, device=logits.device)
        return Categorical(logits=logits.masked_fill(~masks, -torch.inf))

    def _statistics(self, distribution, actions):
        log_probs, entropies = distribution.log_prob(actions), distribution.entropy()
        return dict(agent_log_probs=log_probs, agent_entropies=entropies,
                    log_prob=log_probs.sum(-1), entropy=entropies.sum(-1))

    def sample(self, observations, deterministic=False):
        distribution = self._distribution(observations)
        actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return dict(actions=actions, **self._statistics(distribution, actions))

    def evaluate_actions(self, observations, actions):
        distribution = self._distribution(observations)
        actions = torch.as_tensor(actions, dtype=torch.long, device=distribution.logits.device)
        if actions.shape != distribution.batch_shape:
            raise ValueError('Dispatch actions must have shape [batch,vehicles]')
        return self._statistics(distribution, actions)


class _PaperValueNetwork(nn.Module):
    def __init__(self, scenario, hidden_size=64, *, pricing=False):
        super().__init__()
        self.encoder = (_PaperPricingEncoder if pricing else _PaperPostEncoder)(scenario, hidden_size)
        self.head = _linear(hidden_size, 1, 1.)

    def forward(self, observations):
        return self.head(self.encoder(observations)).squeeze(-1)


@dataclass(frozen=True)
class PaperPPOConfig:
    seed: int = 7
    updates: int = 500
    episodes_per_update: int = 16
    epochs: int = 6
    minibatch_size: int = 192
    hidden_size: int = 64
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    clip_ratio: float = .2
    gae_lambda: float = .95
    pricing_entropy_coef: float = 0.
    dispatch_entropy_coef: float = .01
    max_grad_norm: float = .5
    target_kl: float = 0.
    eval_episodes: int = 64
    eval_interval: int = 25
    eval_seed_base: int = 179000001
    eval_action_seed: int = 189991
    eval_policy: str = 'stochastic'
    normalize_advantages: bool = False
    dispatch_critic_target: str = 'dispatch'
    training_seed_base: int = 210000000
    torch_threads: int = 1
    device: str = 'cpu'

    def validate(self):
        for name in ('updates', 'episodes_per_update', 'epochs', 'minibatch_size', 'hidden_size',
                     'eval_episodes', 'torch_threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'eval_interval', 'eval_seed_base', 'eval_action_seed', 'training_seed_base'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.seed >= 2**31 or self.eval_seed_base % 2 != 1:
            raise ValueError('seed must be below 2**31 and eval_seed_base must be odd')
        for name in ('actor_learning_rate', 'critic_learning_rate', 'clip_ratio', 'gae_lambda',
                     'pricing_entropy_coef', 'dispatch_entropy_coef', 'max_grad_norm', 'target_kl'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if (min(self.actor_learning_rate, self.critic_learning_rate, self.max_grad_norm) <= 0
                or not 0 < self.clip_ratio < 1 or self.gae_lambda > 1):
            raise ValueError('Invalid learning rate, gradient bound, clipping ratio, or GAE lambda')
        if not isinstance(self.normalize_advantages, bool):
            raise ValueError('normalize_advantages must be boolean')
        if self.dispatch_critic_target not in ('dispatch', 'pricing_advantage'):
            raise ValueError('dispatch_critic_target must be dispatch or pricing_advantage')
        if self.eval_policy not in ('stochastic', 'deterministic'):
            raise ValueError('eval_policy must be stochastic or deterministic')
        if self.device not in ('cpu', 'cuda') or self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Requested training device is unavailable')
        return self


@dataclass
class PaperRollout:
    pre_obs: list
    post_obs: list
    pricing_actions: np.ndarray
    dispatch_actions: np.ndarray
    old_price_log_probs: np.ndarray
    old_dispatch_log_probs: np.ndarray
    pricing_rewards: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    price_values: np.ndarray
    dispatch_values: np.ndarray
    next_price_values: np.ndarray
    episodes: list

    def __len__(self):
        return len(self.rewards)

    @property
    def dispatch_rewards(self):
        return self.rewards


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _scenario_data(data):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    return Scenario(**{key: tuples(value) for key, value in data.items()}).validate()


class PaperPPOTrainer:
    """Four independent networks and optimizers; one joint dispatch PPO sample."""
    CHECKPOINT_VERSION = 1

    def __init__(self, scenario=None, config=None):
        self.scenario = (scenario or Scenario()).validate()
        self.training_scenario = self.scenario
        self.config = (config or PaperPPOConfig()).validate()
        self.device = torch.device(self.config.device)
        torch.set_num_threads(self.config.torch_threads)
        self._cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.rng = np.random.default_rng(self.config.seed)
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.manual_seed(self.config.seed)
            self.pricing_actor = _PaperPricingActor(self.scenario, self.config.hidden_size).to(self.device)
            self.dispatch_actor = _PaperDispatchActor(self.scenario, self.config.hidden_size).to(self.device)
            self.pricing_critic = _PaperValueNetwork(self.scenario, self.config.hidden_size, pricing=True).to(self.device)
            self.dispatch_critic = _PaperValueNetwork(self.scenario, self.config.hidden_size).to(self.device)
            self._train_torch_rng = torch.get_rng_state().clone()
            self._train_cuda_rng = torch.cuda.get_rng_state_all() if self._cuda_devices else []
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(),
                lr=self.config.actor_learning_rate if name.endswith('actor') else self.config.critic_learning_rate,
                eps=1e-5) for name, model in self._modules().items()}
        self.update_count = self.episode_count = self.environment_periods = 0
        self._evaluation_deterministic = self.config.eval_policy == 'deterministic'
        self._adapter_torch_rng = self._train_torch_rng.clone()
        self._adapter_cuda_rng = [x.clone() for x in self._train_cuda_rng]

    def _modules(self):
        return {name: getattr(self, name) for name in
                ('pricing_actor', 'pricing_critic', 'dispatch_actor', 'dispatch_critic')}

    @contextmanager
    def _random_context(self, *, training):
        prefix = '_train' if training else '_adapter'
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.set_rng_state(getattr(self, prefix+'_torch_rng'))
            if self._cuda_devices:
                torch.cuda.set_rng_state_all(getattr(self, prefix+'_cuda_rng'))
            try:
                yield
            finally:
                setattr(self, prefix+'_torch_rng', torch.get_rng_state().clone())
                if self._cuda_devices:
                    setattr(self, prefix+'_cuda_rng', [x.clone() for x in torch.cuda.get_rng_state_all()])

    def set_evaluation_mode(self, deterministic=False):
        if not isinstance(deterministic, bool):
            raise ValueError('deterministic must be boolean')
        self._evaluation_deterministic = deterministic
        for model in self._modules().values():
            model.eval()
        return self

    def on_episode_start(self, env_seed, action_seed):
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0
               for x in (env_seed, action_seed)):
            raise ValueError('Episode and action seeds must be nonnegative integers')
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.manual_seed((int(env_seed)+int(action_seed)) % (2**63-1))
            self._adapter_torch_rng = torch.get_rng_state().clone()
            self._adapter_cuda_rng = torch.cuda.get_rng_state_all() if self._cuda_devices else []

    @torch.no_grad()
    def prices(self, obs, rng=None):
        with self._random_context(training=False):
            return self.pricing_actor.sample([obs], self._evaluation_deterministic)['actions'][0].cpu().numpy().copy()

    @torch.no_grad()
    def dispatch(self, obs, rng=None):
        with self._random_context(training=False):
            actions = self.dispatch_actor.sample([obs], self._evaluation_deterministic)['actions'][0].cpu().numpy()
        return paper_dispatch_commands(obs, actions, self.scenario)

    def with_scenario(self, scenario, allow_distribution_shift=False):
        if not isinstance(allow_distribution_shift, bool):
            raise ValueError('allow_distribution_shift must be boolean')
        scenario.validate()
        old, new = asdict(self.training_scenario), asdict(scenario)
        changed = {key for key in old if old[key] != new[key]}
        allowed = {'arrival_rates', 'arrival_profile', 'mean_requests_by_zone', 'initial_energy_kwh'}
        if changed and (not allow_distribution_shift or changed-allowed):
            raise ValueError(f'Unsupported change from training scenario: {sorted(changed)}')
        clone = deepcopy(self)
        clone.scenario = scenario
        return clone

    def collect_rollout(self):
        if self.scenario != self.training_scenario:
            raise ValueError('Distribution-shift adapters are for frozen evaluation, not further training')
        data = {key: [] for key in PaperRollout.__dataclass_fields__}
        for model in self._modules().values():
            model.train()
        with self._random_context(training=True), torch.no_grad():
            for _ in range(self.config.episodes_per_update):
                seed = 2*(self.config.training_seed_base+self.config.seed+self.episode_count)
                sim = ChargingSimulator(self.scenario)
                obs, _ = sim.reset(seed=seed)
                infos, rewards = [], []
                while sim.stage != 2:
                    pre = paper_pricing_observation(obs, self.training_scenario)
                    price = self.pricing_actor.sample([pre])
                    vp = float(self.pricing_critic([pre])[0])
                    levels = price['actions'][0].cpu().numpy().copy()
                    post, rp, done, truncated, _ = sim.quote_price_levels(levels)
                    if rp != 0. or done or truncated:
                        raise RuntimeError('Unexpected simulator pricing transition')
                    dispatch = self.dispatch_actor.sample([post])
                    vd = float(self.dispatch_critic([post])[0])
                    actions = dispatch['actions'][0].cpu().numpy().copy()
                    obs, reward, done, truncated, info = sim.dispatch(
                        paper_dispatch_commands(post, actions, self.scenario))
                    if truncated or info['invalid_actions']:
                        raise RuntimeError('Paper action mask emitted an invalid physical command')
                    next_vp = 0. if done else float(self.pricing_critic([obs])[0])
                    for key, value in (
                        ('pre_obs', pre), ('post_obs', post), ('pricing_actions', levels),
                        ('dispatch_actions', actions), ('old_price_log_probs', float(price['log_prob'][0])),
                        ('old_dispatch_log_probs', dispatch['agent_log_probs'][0].cpu().numpy().copy()),
                        ('pricing_rewards', rp), ('rewards', reward), ('dones', done),
                        ('price_values', vp), ('dispatch_values', vd), ('next_price_values', next_vp)):
                        data[key].append(value)
                    rewards.append(reward)
                    infos.append(info)
                data['episodes'].append(episode_summary(seed, infos, rewards,
                    self.scenario.discount_per_period, n_mcv=self.scenario.n_mcv))
                self.episode_count += 1
        for key in data.keys()-{'pre_obs', 'post_obs', 'episodes'}:
            data[key] = np.asarray(data[key])
        self.environment_periods += len(data['rewards'])
        return PaperRollout(**data)

    def update(self, rollout):
        c, n = self.config, len(rollout)
        if not n or not rollout.dones[-1]:
            raise ValueError('Paper PPO update requires complete nonempty episodes')
        gae = interleaved_gae(rollout.pricing_rewards, rollout.rewards, rollout.price_values,
            rollout.dispatch_values, rollout.next_price_values, rollout.dones,
            self.scenario.discount_per_period, c.gae_lambda)
        targets = critic_targets(gae, rollout.price_values, rollout.dispatch_values, c.dispatch_critic_target)
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        ap, ad = tensor(gae['pricing_advantages']), tensor(gae['dispatch_advantages'])
        if c.normalize_advantages:
            def normalize(x):
                return (x-x.mean())/(x.std(unbiased=False)+1e-8) if x.numel() > 1 and x.std(unbiased=False) > 1e-8 else x
            ap, ad = normalize(ap), normalize(ad)
        p_returns, d_returns = tensor(targets['pricing_returns']), tensor(targets['dispatch_returns'])
        p_actions, d_actions = tensor(rollout.pricing_actions, torch.long), tensor(rollout.dispatch_actions, torch.long)
        old_p, old_d = tensor(rollout.old_price_log_probs), tensor(rollout.old_dispatch_log_probs)
        records, stopped, steps = [], False, 0
        before = {name: torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
                  for name, model in self._modules().items()}
        for _ in range(c.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, c.minibatch_size):
                indices = order[start:start+c.minibatch_size]
                ix = tensor(indices, torch.long)
                pre = [rollout.pre_obs[int(i)] for i in indices]
                post = [rollout.post_obs[int(i)] for i in indices]
                p = self.pricing_actor.evaluate_actions(pre, p_actions[ix])
                d = self.dispatch_actor.evaluate_actions(post, d_actions[ix])
                p_stats = joint_ppo_loss(p['log_prob'][:, None], old_p[ix, None], ap[ix],
                                         c.clip_ratio, p['entropy'][:, None])
                d_stats = joint_ppo_loss(d['agent_log_probs'], old_d[ix], ad[ix],
                                         c.clip_ratio, d['agent_entropies'])
                if c.target_kl and max(float(p_stats['approx_kl'].detach()),
                                      float(d_stats['approx_kl'].detach())) > c.target_kl:
                    stopped = True
                    break
                vp, vd = self.pricing_critic(pre), self.dispatch_critic(post)
                vp_loss, vd_loss = (vp-p_returns[ix]).square().mean(), (vd-d_returns[ix]).square().mean()
                p_loss = p_stats['loss']-c.pricing_entropy_coef*p_stats['entropy']
                d_loss = d_stats['loss']-c.dispatch_entropy_coef*d_stats['entropy']
                loss = p_loss+d_loss+vp_loss+vd_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite paper PPO loss; no optimizer step applied')
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = {name: nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                       error_if_nonfinite=True)
                         for name, model in self._modules().items()}
                for optimizer in self.optimizers.values():
                    optimizer.step()
                steps += 1
                records.append(dict(pricing_policy_loss=float(p_stats['loss'].detach()),
                    dispatch_policy_loss=float(d_stats['loss'].detach()), pricing_value_loss=float(vp_loss.detach()),
                    dispatch_value_loss=float(vd_loss.detach()), pricing_entropy=float(p_stats['entropy'].detach()),
                    dispatch_entropy=float(d_stats['entropy'].detach()),
                    pricing_approx_kl=float(p_stats['approx_kl'].detach()),
                    dispatch_approx_kl=float(d_stats['approx_kl'].detach()),
                    pricing_clip_fraction=float(p_stats['clip_fraction']),
                    dispatch_clip_fraction=float(d_stats['clip_fraction']),
                    **{name+'_grad_norm': float(norm) for name, norm in norms.items()}))
            if stopped:
                break
        if not records:
            raise RuntimeError('No paper PPO optimizer updates; rollout is stale or KL threshold too small')
        self.update_count += 1
        result = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
        result.update(update=self.update_count, episodes=self.episode_count,
            environment_periods=self.environment_periods, optimizer_steps=steps, early_stopped=int(stopped),
            mean_training_profit=float(np.mean([e['profit'] for e in rollout.episodes])),
            mean_training_discounted_profit=float(np.mean([e['discounted_profit'] for e in rollout.episodes])),
            mean_outside_share=float(np.mean([e['outside_option_share'] for e in rollout.episodes])))
        with torch.no_grad():
            for name, model in self._modules().items():
                after = torch.cat([p.detach().flatten() for p in model.parameters()])
                if not torch.isfinite(after).all():
                    raise FloatingPointError('Nonfinite '+name+' parameters')
                result[name+'_parameter_delta'] = float(torch.linalg.vector_norm(after-before[name]))
            for name, model, observations, target in (
                ('pricing', self.pricing_critic, rollout.pre_obs, p_returns),
                ('dispatch', self.dispatch_critic, rollout.post_obs, d_returns)):
                prediction = model(observations)
                variance = target.var(unbiased=False)
                result[name+'_explained_variance'] = float(1-(target-prediction).var(unbiased=False)/variance) if variance > 1e-8 else 0.
                result[name+'_value_rmse'] = float((target-prediction).square().mean().sqrt())
        return result

    def evaluate(self, seeds, deterministic=False, action_seed=None):
        deterministic = bool(deterministic)
        previous_mode = self._evaluation_deterministic
        adapter_rng = self._adapter_torch_rng.clone()
        adapter_cuda = [x.clone() for x in self._adapter_cuda_rng]
        training_flags = {name: model.training for name, model in self._modules().items()}
        try:
            self.set_evaluation_mode(deterministic)
            result = evaluate_policy(self.scenario, seeds, self.prices, self.dispatch,
                action_seed=self.config.eval_action_seed if action_seed is None else action_seed,
                label=ALGORITHM, on_episode_start=self.on_episode_start)
            result['deterministic'] = deterministic
            return result
        finally:
            self._evaluation_deterministic = previous_mode
            self._adapter_torch_rng, self._adapter_cuda_rng = adapter_rng, adapter_cuda
            for name, model in self._modules().items():
                model.train(training_flags[name])

    def save_checkpoint(self, path):
        if self.scenario != self.training_scenario:
            raise ValueError('Save the original training model, not a stress-evaluation adapter')
        payload = dict(format_version=self.CHECKPOINT_VERSION, algorithm=ALGORITHM,
            scenario=asdict(self.training_scenario), training_config=asdict(self.config),
            models={name: model.state_dict() for name, model in self._modules().items()},
            optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            update_count=self.update_count, episode_count=self.episode_count,
            environment_periods=self.environment_periods, numpy_rng=self.rng.bit_generator.state,
            torch_rng=self._train_torch_rng, cuda_rng=self._train_cuda_rng)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        required = {'format_version', 'algorithm', 'scenario', 'training_config', 'models', 'optimizers',
                    'update_count', 'episode_count', 'environment_periods', 'numpy_rng', 'torch_rng', 'cuda_rng'}
        if (not isinstance(payload, dict) or set(payload) != required
                or payload['format_version'] != cls.CHECKPOINT_VERSION or payload['algorithm'] != ALGORITHM):
            raise ValueError('Unsupported paper PPO checkpoint format or algorithm')
        def check_finite(value):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Paper PPO checkpoint contains a nonfinite tensor')
            if isinstance(value, dict):
                for item in value.values(): check_finite(item)
            elif isinstance(value, (list, tuple)):
                for item in value: check_finite(item)
        check_finite(payload)
        scenario = _scenario_data(payload['scenario'])
        config = PaperPPOConfig(**dict(payload['training_config'], device=device)).validate()
        trainer = cls(scenario, config)
        if set(payload['models']) != set(trainer._modules()) or set(payload['optimizers']) != set(trainer.optimizers):
            raise ValueError('Paper PPO checkpoint must contain four independent models and optimizers')
        for name, model in trainer._modules().items():
            model.load_state_dict(payload['models'][name], strict=True)
            trainer.optimizers[name].load_state_dict(payload['optimizers'][name])
        for name in ('update_count', 'episode_count', 'environment_periods'):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Invalid checkpoint counter '+name)
            setattr(trainer, name, value)
        trainer.rng.bit_generator.state = payload['numpy_rng']
        trainer._train_torch_rng = payload['torch_rng'].cpu().clone()
        if device == 'cuda':
            if len(payload['cuda_rng']) != len(trainer._cuda_devices):
                raise ValueError('CUDA RNG device count differs from checkpoint')
            trainer._train_cuda_rng = [x.cpu().clone() for x in payload['cuda_rng']]
        return trainer

    def _source_hashes(self):
        package = Path(__file__).resolve().parents[1]
        files = ('rl/paper_ppo.py', 'rl/paper_train.py', 'simulator.py', 'config.py', 'choice.py',
                 'benchmarks/evaluation.py')
        return {'src/charging_env/'+name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files}

    def train(self, output_dir):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError('Paper PPO output contains a previous run; choose an empty directory')
        c = self.config
        hashes = self._source_hashes()
        protocol = dict(algorithm=ALGORITHM, scenario=asdict(self.scenario), training_config=asdict(c),
            source_sha256=hashes, pricing_information='time,fleet,arrival intensities,FCS prices; realized arrivals hidden',
            dispatch_likelihood='sum vehicle log probabilities, exponentiate once, clip joint ratio once',
            interleaved_gae='lambda at both substages; gamma only at physical-period transitions',
            network_architecture='four independent encoders; two hidden SiLU layers with LayerNorm; centralized multihead dispatch',
            quantity_convention='maximum individually feasible charge/export request, then original shared-station allocation',
            idle_convention='stay=current WAIT; admissible other node=travel-only REPOSITION; current node idle fallback',
            raw_rewards=True, reward_shaping=False, imitation=False, exhaustive_joint_dispatch=False,
            policy_surrogate='raw advantages by default, no outer period-discount weights',
            dispatch_critic_target=c.dispatch_critic_target,
            training_seed_rule='2*(training_seed_base+seed+episode_count)')
        _write_json(directory/'scenario.json', asdict(self.scenario))
        _write_json(directory/'training_config.json', asdict(c))
        _write_json(directory/'protocol.json', protocol)
        seeds = [c.eval_seed_base+2*i for i in range(c.eval_episodes)]
        initial = self.evaluate(seeds, c.eval_policy == 'deterministic')
        _write_json(directory/'initial_evaluation.json', initial)
        evaluations = []
        validation_history = [dict(update=self.update_count, phase='initial', **initial)]
        _write_json(directory/'validation_history.json', validation_history)
        best = initial['mean_discounted_profit']
        self.save_checkpoint(directory/'best_checkpoint.pt')
        started = time.perf_counter()
        rows = []
        with (directory/'training_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = None
            for run_update in range(c.updates):
                row = self.update(self.collect_rollout())
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                if c.eval_interval and self.update_count % c.eval_interval == 0:
                    evaluation = self.evaluate(seeds, c.eval_policy == 'deterministic')
                    evaluations.append(dict(update=self.update_count, **evaluation))
                    validation_history.append(dict(update=self.update_count, phase='periodic', **evaluation))
                    if evaluation['mean_discounted_profit'] > best:
                        best = evaluation['mean_discounted_profit']
                        self.save_checkpoint(directory/'best_checkpoint.pt')
                    _write_json(directory/'evaluations.json', evaluations)
                    _write_json(directory/'validation_history.json', validation_history)
                    print(f"Paper PPO update {self.update_count}: validation profit "
                          f"{evaluation['mean_discounted_profit']:.4f}, joint KL {row['dispatch_approx_kl']:.6f}", flush=True)
                elif run_update == 0 or (run_update+1) % 10 == 0:
                    print(f"Paper PPO update {self.update_count}: training profit "
                          f"{row['mean_training_discounted_profit']:.4f}", flush=True)
        final = self.evaluate(seeds, c.eval_policy == 'deterministic')
        _write_json(directory/'final_evaluation.json', final)
        if validation_history[-1]['update'] != self.update_count:
            validation_history.append(dict(update=self.update_count, phase='final', **final))
        _write_json(directory/'validation_history.json', validation_history)
        if final['mean_discounted_profit'] > best:
            best = final['mean_discounted_profit']
            self.save_checkpoint(directory/'best_checkpoint.pt')
        if self._source_hashes() != hashes:
            raise RuntimeError('Paper PPO source changed during training; final checkpoint was not saved')
        checkpoint = directory/'checkpoint.pt'
        self.save_checkpoint(checkpoint)
        loaded = self.load_checkpoint(checkpoint, device=c.device)
        replay = loaded.evaluate(seeds, c.eval_policy == 'deterministic')
        replay_matches = all(x['discounted_profit'] == y['discounted_profit'] and x['profit'] == y['profit']
                             for x, y in zip(final['episodes'], replay['episodes']))
        if not replay_matches:
            raise RuntimeError('Paper PPO checkpoint replay differs from the saved evaluation')
        summary = dict(algorithm=ALGORITHM, seed=c.seed, update_count=self.update_count,
            episode_count=self.episode_count, environment_periods=self.environment_periods,
            updates=self.update_count, episodes=self.episode_count,
            initial_mean_discounted_profit=initial['mean_discounted_profit'],
            final_mean_discounted_profit=final['mean_discounted_profit'],
            best_validation_profit=best, all_metrics_finite=all(np.isfinite(value)
                for row in rows for value in row.values()), checkpoint_reload_matches=replay_matches,
            invalid_evaluation_actions=initial['invalid_actions']+final['invalid_actions']+replay['invalid_actions']
                +sum(e['invalid_actions'] for e in evaluations),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest(),
            monitoring_evaluation_episodes=(2+len(evaluations))*c.eval_episodes,
            checkpoint_replay_episodes=c.eval_episodes,
            total_evaluation_policy_episodes=(3+len(evaluations))*c.eval_episodes,
            training_seconds=time.perf_counter()-started, final_evaluation=final)
        _write_json(directory/'summary.json', summary)
        return summary
