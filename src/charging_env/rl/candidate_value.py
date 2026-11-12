"""Learned Q and continuation scorers for the original rollout candidates.

Q predicts full return from exact current profit and afterstate. Hybrid retains
exact current profit and learns only expected return from the next period.
Neither variant samples future customers at deployment.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn

from ..benchmarks.policies import dispatch_outcome
from ..config import Scenario
from .priced_rollout import _price_levels
from .rollout_adp import RolloutADPConfig, RolloutADPPolicy

FEATURE_SCHEMA = 'charging_env.candidate_afterstate_features.v1'
CHECKPOINT_SCHEMA = 'charging_env.candidate_value.v1'
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def _normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _scale(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError('reward_scale must be finite and positive')
    return float(value)


def _pricing_identity(pricer):
    provenance = _normalized(pricer.provenance())
    # A file digest may become known after saving without changing any prices.
    provenance.pop('checkpoint_sha256', None)
    return provenance.get('policy_sha256') or hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _network_identity(network):
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        array = value.detach().cpu().numpy().astype('<f4', copy=False)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def afterstate_feature_size(scenario):
    return scenario.horizon + 2 + scenario.n_mcv * (2 * scenario.n_zones + 2) + 4 * scenario.n_zones + 2


def encode_afterstate(scenario, next_period, locations, energies):
    """Return [batch,features] using ordered fleet and prescribed forecasts.

    The time index includes T at closing. Exact location one-hots, usable
    energy, energy squared and location-energy interactions retain vehicle
    order. Spatial fleet summaries, forecasts and known tariff summaries aid
    interpolation. No realized future prices/requests are read. Features are
    in [0,1], independently of a fitting data set.
    """
    if (isinstance(next_period, (bool, np.bool_)) or not isinstance(next_period, Integral)
            or not 0 <= next_period <= scenario.horizon):
        raise ValueError('next_period must be an integer in [0,T]')
    loc, energy = np.asarray(locations), np.asarray(energies, dtype=float)
    if loc.ndim == 1:
        loc = loc[None, :]
    if energy.ndim == 1:
        energy = energy[None, :]
    if (loc.ndim != 2 or loc.shape[1] != scenario.n_mcv or loc.dtype.kind not in 'iu'
            or energy.shape != loc.shape or (loc < 0).any() or (loc >= scenario.n_zones).any()
            or not np.isfinite(energy).all()):
        raise ValueError('Afterstate needs ordered valid locations and finite energies')
    if ((energy < scenario.reserve_kwh - 1e-8).any()
            or (energy > scenario.battery_capacity_kwh + 1e-8).any()):
        raise ValueError('Afterstate energy violates physical bounds')
    energy = np.clip(energy, scenario.reserve_kwh, scenario.battery_capacity_kwh)
    width = scenario.battery_capacity_kwh - scenario.reserve_kwh
    usable = (energy - scenario.reserve_kwh) / width if width else np.zeros_like(energy)
    batch = len(loc)
    onehot = np.eye(scenario.n_zones, dtype=float)[loc]
    ordered = np.concatenate((onehot, usable[..., None], usable[..., None] ** 2,
                              onehot * usable[..., None]), axis=2).reshape(batch, -1)
    fleet = np.concatenate((onehot.mean(axis=1), (onehot * usable[..., None]).mean(axis=1)), axis=1)
    time_features = np.zeros((batch, scenario.horizon + 2))
    time_features[:, int(next_period)] = 1.
    time_features[:, -1] = (scenario.horizon - next_period) / scenario.horizon
    rates = np.asarray(scenario.resolved_arrival_rates, dtype=float)
    demand_scale = max(float(rates.max()), 1.)
    if next_period < scenario.horizon:
        forecasts = np.concatenate((rates[next_period] / demand_scale,
            rates[next_period:].sum(axis=0) / (scenario.horizon * demand_scale)))
        buy = np.asarray(scenario.grid_buy_per_kwh, dtype=float)
        price_scale = max(float(np.max(buy)), 1e-6)
        tariff = np.array([buy[next_period] / price_scale,
                          buy[next_period:].sum() / (scenario.horizon * price_scale)])
    else:
        forecasts, tariff = np.zeros(2 * scenario.n_zones), np.zeros(2)
    result = np.concatenate((time_features, ordered, fleet,
        np.broadcast_to(forecasts, (batch, len(forecasts))),
        np.broadcast_to(tariff, (batch, 2))), axis=1).astype(np.float32)
    if result.shape != (batch, afterstate_feature_size(scenario)) or not np.isfinite(result).all():
        raise FloatingPointError('Invalid afterstate features')
    return result


@dataclass(frozen=True)
class CandidateBatch:
    names: list
    plans: list
    rewards: np.ndarray
    next_locations: np.ndarray
    next_energy: np.ndarray
    q_features: np.ndarray
    value_features: np.ndarray
    terminal: bool
    coordinate_candidate_outcomes: int = 0
    period: int = 0


def build_candidates(obs, scenario, rollout_config=None, reward_scale=100.):
    """Reuse the original deduplicated generator and exact joint transition."""
    scale = _scale(reward_scale)
    period_array = np.asarray(obs['period'])
    if (int(obs['stage']) != 1 or period_array.shape != (1,) or period_array.dtype.kind not in 'iu'
            or not 0 <= period_array[0] < scenario.horizon):
        raise ValueError('Candidates require a valid dispatch observation')
    period = int(period_array[0])
    generator = RolloutADPPolicy(scenario, rollout_config or RolloutADPConfig())
    names, plans, statistics = generator._candidate_plans(obs)
    outcomes = [dispatch_outcome(obs, scenario, plan) for plan in plans]
    rewards = np.asarray([o['reward'] for o in outcomes], dtype=np.float64)
    locations = np.asarray([o['next_locations'] for o in outcomes], dtype=np.int64)
    energies = np.asarray([o['next_energy_kwh'] for o in outcomes], dtype=np.float64)
    values = encode_afterstate(scenario, period + 1, locations, energies)
    q_features = np.concatenate(((rewards / scale).astype(np.float32)[:, None], values), axis=1)
    return CandidateBatch(names, plans, rewards, locations, energies, q_features, values,
        period + 1 == scenario.horizon, int(statistics['candidate_outcomes']), period)


class ValueNetwork(nn.Module):
    """Shared scalar candidate scorer; output units are reward_scale dollars."""
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (input_dim, hidden_size)):
            raise ValueError('Network dimensions must be positive integers')
        self.input_dim, self.hidden_size = int(input_dim), int(hidden_size)
        self.layers = nn.Sequential(nn.Linear(self.input_dim, self.hidden_size), nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class CandidateValuePolicy:
    """Frozen same-price dispatch with learned Q or continuation scores."""
    def __init__(self, scenario, pricing_policy, network, method='q', rollout_config=None, reward_scale=100.):
        self.scenario = scenario.validate()
        if method not in ('q', 'hybrid'):
            raise ValueError('method must be q or hybrid')
        if (getattr(pricing_policy, 'scenario', None) != scenario
                or not callable(getattr(pricing_policy, 'prices', None))
                or not callable(getattr(pricing_policy, 'provenance', None))):
            raise ValueError('Pricing policy must match the dispatch scenario')
        self.config = (rollout_config or RolloutADPConfig()).validate(scenario)
        self.reward_scale, self.method = _scale(reward_scale), method
        expected = afterstate_feature_size(scenario) + int(method == 'q')
        if not isinstance(network, ValueNetwork) or network.input_dim != expected:
            raise ValueError('Network input width does not match method/scenario')
        if any(not torch.isfinite(p).all() for p in network.parameters()):
            raise ValueError('Network weights must be finite')
        self.pricing_policy = deepcopy(pricing_policy)
        self.network = deepcopy(network).to(device='cpu', dtype=torch.float32).eval()
        self.network.requires_grad_(False)
        self.statistics = dict(decisions=0, candidate_plans=0, current_plan_outcomes=0,
            coordinate_candidate_outcomes=0, future_model_queries=0, exogenous_draws=0)
        self.decisions, self.last_batch = [], None
        self.reset_episode(0)

    def reset_episode(self, episode_key=0, action_seed=None):
        if (isinstance(episode_key, (bool, np.bool_)) or not isinstance(episode_key, Integral)
                or episode_key < 0):
            raise ValueError('episode_key must be a nonnegative integer')
        self.episode_key, self.episode_decision = int(episode_key), 0

    def prices(self, obs, rng=None):
        return _price_levels(obs, self.scenario, self.pricing_policy)[0]

    def score_batch(self, batch):
        if batch.terminal:
            return batch.rewards.copy()
        features = batch.q_features if self.method == 'q' else batch.value_features
        with torch.inference_mode():
            values = self.network(torch.from_numpy(features)).cpu().numpy().astype(np.float64)
        scores = self.reward_scale * values if self.method == 'q' else (
            batch.rewards + self.scenario.discount_per_period * self.reward_scale * values)
        if scores.shape != batch.rewards.shape or not np.isfinite(scores).all():
            raise FloatingPointError('Candidate scores must be finite scalars')
        return scores

    def dispatch(self, obs, rng=None):
        started = time.perf_counter()
        batch = build_candidates(obs, self.scenario, self.config, self.reward_scale)
        scores = self.score_batch(batch)
        winner = int(np.argmax(scores))
        self.last_batch = batch
        self.statistics['decisions'] += 1
        self.statistics['candidate_plans'] += len(batch.names)
        self.statistics['current_plan_outcomes'] += len(batch.names)
        self.statistics['coordinate_candidate_outcomes'] += batch.coordinate_candidate_outcomes
        self.decisions.append(dict(episode_key=self.episode_key, episode_decision=self.episode_decision,
            period=batch.period, candidates=batch.names, selected=batch.names[winner],
            current_profit=batch.rewards.tolist(), mean_total_scores=scores.tolist(),
            future_model_queries=0, coordinate_candidate_outcomes=batch.coordinate_candidate_outcomes,
            seconds=time.perf_counter() - started))
        self.episode_decision += 1
        return batch.plans[winner]

    def provenance(self):
        return dict(schema=CHECKPOINT_SCHEMA, algorithm='candidate_q_learning' if self.method == 'q'
            else 'exact_reward_learned_continuation_adp', method=self.method,
            scenario=_normalized(asdict(self.scenario)), config=_normalized(asdict(self.config)),
            reward_scale=self.reward_scale, feature_schema=FEATURE_SCHEMA,
            input_dim=self.network.input_dim, hidden_size=self.network.hidden_size,
            network_sha256=_network_identity(self.network),
            pricing_identity=_pricing_identity(self.pricing_policy), future_model_queries_at_deployment=0,
            current_transition='exact known full-fleet physics',
            candidate_generator='RolloutADPPolicy._candidate_plans; original order and deduplication',
            q_representation='normalized exact current profit plus ordered afterstate; full return predicted',
            hybrid_representation='next-period expected return from afterstate; exact current profit retained',
            terminal='exact current profit; no network continuation', statistics=dict(self.statistics))

    def save_checkpoint(self, path):
        metadata = self.provenance()
        metadata.pop('statistics')
        arrays = {key: value.detach().cpu().numpy() for key, value in self.network.state_dict().items()}
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)), **arrays)

    @classmethod
    def load_checkpoint(cls, path, pricing_policy):
        path = Path(path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError('Checkpoint exceeds size limit')
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if sum(item.file_size for item in entries) > MAX_CHECKPOINT_BYTES:
                raise ValueError('Expanded checkpoint exceeds size limit')
            if len(set(item.filename for item in entries)) != len(entries):
                raise ValueError('Duplicate checkpoint entries')
        with np.load(path, allow_pickle=False) as data:
            raw = data['metadata']
            if raw.shape != () or raw.dtype.kind != 'U' or raw.nbytes > 2 * 1024 * 1024:
                raise ValueError('Invalid checkpoint metadata')
            metadata = json.loads(str(raw.item()))
            if (metadata.get('schema') != CHECKPOINT_SCHEMA or metadata.get('feature_schema') != FEATURE_SCHEMA
                    or metadata.get('method') not in ('q', 'hybrid')):
                raise ValueError('Unsupported candidate-value checkpoint')
            if metadata['pricing_identity'] != _pricing_identity(pricing_policy):
                raise ValueError('Checkpoint requires the original frozen pricing policy')
            def tuples(value):
                return tuple(map(tuples, value)) if isinstance(value, list) else value
            scenario = Scenario(**{key: tuples(value) for key, value in metadata['scenario'].items()}).validate()
            config = RolloutADPConfig(**{key: tuples(value) for key, value in metadata['config'].items()})
            expected = afterstate_feature_size(scenario) + int(metadata['method'] == 'q')
            if (metadata['input_dim'] != expected or not isinstance(metadata['hidden_size'], int)
                    or not 1 <= metadata['hidden_size'] <= 4096):
                raise ValueError('Checkpoint network dimensions differ')
            with torch.random.fork_rng(devices=[]):
                network = ValueNetwork(expected, metadata['hidden_size'])
            reference = network.state_dict()
            if set(data.files) != set(reference) | {'metadata'}:
                raise ValueError('Unexpected checkpoint arrays')
            state = {}
            for key, value in reference.items():
                array = data[key]
                if array.shape != tuple(value.shape) or array.dtype.kind != 'f' or not np.isfinite(array).all():
                    raise ValueError('Invalid network weights: ' + key)
                state[key] = torch.from_numpy(array.astype(np.float32, copy=True))
            network.load_state_dict(state)
            if metadata.get('network_sha256') != _network_identity(network):
                raise ValueError('Checkpoint network identity differs')
        return cls(scenario, pricing_policy, network, metadata['method'], config, metadata['reward_scale'])
