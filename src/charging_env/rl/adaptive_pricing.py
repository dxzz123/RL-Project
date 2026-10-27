"""Small adaptive pricing policies fitted by direct, paired policy search.

Pricing observes prescribed arrival rates, fleet state and current electricity
prices. Realized requests and customer choices are deliberately hidden. Search
evaluates full episodes of original business profit with a supplied dispatch
backend; it fits no value function and adds no surrogate economic reward.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import json
from numbers import Integral, Real
from pathlib import Path
import time
import zipfile

import numpy as np

from ..benchmarks.evaluation import evaluate_policy
from ..config import Scenario


FEATURE_SCHEMA = 'charging_env.shared_zone_pricing_features.v1'
CHECKPOINT_SCHEMA = 'charging_env.learned_pricing.v1'
FEATURE_NAMES = ('bias', 'time', 'arrival_rate', 'capacity_demand_balance',
    'fleet_battery', 'eligible_fleet_fraction', 'mobile_travel_time',
    'station_generalized_price', 'replacement_energy_price', 'station_load_balance',
    'time_battery', 'time_capacity_demand')
PARAMETER_COUNT = len(FEATURE_NAMES)
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024
MAX_CANDIDATE_EPISODES = 100_000
_COUNTERS = ('decisions', 'candidate_plans', 'current_plan_outcomes', 'future_model_queries',
    'exogenous_draws', 'coordinate_candidate_outcomes', 'boundary_roundoff_corrections',
    'calls', 'candidate_outcomes', 'candidate_budget_total')


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _real(value, name, minimum=0., maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'Invalid {name}')


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _scenario(data):
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(Scenario)}:
        raise ValueError('Checkpoint must contain exactly the Scenario fields')
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    return Scenario(**{key: tuples(value) for key, value in data.items()}).validate()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parents[1]
    return {'src/charging_env/'+name: _hash(root/name) for name in
        ('rl/adaptive_pricing.py', 'benchmarks/evaluation.py', 'simulator.py', 'choice.py', 'config.py')}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def pricing_observation(obs, scenario):
    """Pure NumPy equivalent of ``paper_pricing_observation``.

    Never copy input availability masks or station waiting times: those can
    encode latent requests. Recompute availability solely from fleet physics.
    """
    if int(obs['stage']) != 0:
        raise ValueError('Learned pricing requires the before-choice stage')
    keys = ('period', 'locations', 'energy_kwh', 'arrival_rates', 'fcs_retail_prices',
            'fcs_discharge_prices', 'grid_buy_price')
    result = {key: np.asarray(obs[key]).copy() for key in keys}
    loc, energy = result['locations'], result['energy_kwh']
    if (result['period'].shape != (1,) or result['period'].dtype.kind not in 'iu'
            or not 0 <= result['period'][0] < scenario.horizon
            or loc.shape != (scenario.n_mcv,) or loc.dtype.kind not in 'iu'
            or (loc < 0).any() or (loc >= scenario.n_zones).any()
            or energy.shape != (scenario.n_mcv,) or energy.dtype.kind not in 'iuf'
            or not np.isfinite(energy).all() or (energy < scenario.reserve_kwh-1e-8).any()
            or (energy > scenario.battery_capacity_kwh+1e-8).any()):
        raise ValueError('Malformed pricing fleet or period')
    for key, shape in (('arrival_rates', (scenario.n_zones,)),
            ('fcs_retail_prices', (scenario.n_fcs,)), ('fcs_discharge_prices', (scenario.n_fcs,)),
            ('grid_buy_price', (1,))):
        value = result[key]
        if value.shape != shape or value.dtype.kind not in 'iuf' or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Malformed pricing input: '+key)
    distance = scenario.distances[loc]
    hours = scenario.period_hours-distance/scenario.speed_kmph
    capacity = np.maximum(0., np.minimum(
        (energy[:, None]-distance*scenario.travel_kwh_per_km-scenario.reserve_kwh)
        * scenario.mobile_service_efficiency, hours*scenario.mobile_service_kw))
    available = (capacity+1e-9 >= scenario.request_min_kwh).any(axis=0)
    result['stage'] = 0
    result['pricing_mask'] = np.repeat(available[:, None], len(scenario.price_levels), axis=1)
    return result


def pricing_features(obs, scenario):
    """Twelve shared-zone features, each bounded in [-1, 1].

    Capacity is apportioned using prescribed expected demand, rather than
    counted independently at every reachable zone. Station load likewise uses
    expected request energy and static accessibility, never realized congestion.
    These are inputs to a fitted policy, not a claim of exact service capacity.
    """
    return _pricing_features(pricing_observation(obs, scenario), scenario)


def _pricing_features(o, c):
    rates = o['arrival_rates'].astype(float)
    mean_request = (c.request_min_kwh+c.request_max_kwh)/2.
    demand = rates*mean_request
    distance = c.distances[o['locations']]
    hours = c.period_hours-distance/c.speed_kmph
    capacities = np.maximum(0., np.minimum(
        (o['energy_kwh'][:, None]-distance*c.travel_kwh_per_km-c.reserve_kwh)
        * c.mobile_service_efficiency, hours*c.mobile_service_kw))
    eligible = capacities+1e-9 >= c.request_min_kwh
    weights = eligible*demand[None, :]
    weights = np.divide(weights, weights.sum(axis=1, keepdims=True),
        out=np.zeros_like(weights), where=weights.sum(axis=1, keepdims=True) > 0)
    supply = (weights*capacities).sum(axis=0)
    balance = (supply-demand)/(supply+demand+mean_request)
    width = c.battery_capacity_kwh-c.reserve_kwh
    battery = 2*float(np.mean((o['energy_kwh']-c.reserve_kwh)/max(width, 1e-12)))-1
    travel = np.divide((eligible*distance/c.speed_kmph).sum(axis=0), eligible.sum(axis=0),
        out=np.full(c.n_zones, c.period_hours), where=eligible.sum(axis=0) > 0)
    station_distance = c.distances[:, c.fcs_zones]
    access = ((station_distance <= c.max_fcs_access_km)
              & (np.asarray(c.fcs_ev_capacity_kwh)[None, :] >= c.request_min_kwh))
    load_weights = np.divide(access, access.sum(axis=1, keepdims=True),
        out=np.zeros_like(station_distance), where=access.sum(axis=1, keepdims=True) > 0)
    station_load = (load_weights*demand[:, None]).sum(axis=0)
    station_capacity = np.asarray(c.fcs_ev_capacity_kwh)
    station_balance = (station_capacity-station_load)/(station_capacity+station_load+mean_request)
    generalized = o['fcs_retail_prices'][None, :]+c.fixed_travel_cost_per_km*station_distance/mean_request
    generalized = np.where(access, generalized, np.inf)
    competitor = generalized.min(axis=1)
    levels = np.asarray(c.price_levels)
    middle, span = float((levels.min()+levels.max())/2), max(float(np.ptp(levels)), 1e-12)
    competitor = np.where(np.isfinite(competitor), competitor, levels.max())
    t = 2*float(o['period'][0])/max(1, c.horizon-1)-1
    columns = (np.ones(c.n_zones), np.full(c.n_zones, t),
        2*rates/max(1., float(np.max(c.resolved_arrival_rates)))-1, balance,
        np.full(c.n_zones, battery), 2*eligible.mean(axis=0)-1,
        2*travel/c.period_hours-1, (competitor-middle)/span,
        np.full(c.n_zones, (float(o['grid_buy_price'][0])/(c.charge_efficiency*c.mobile_service_efficiency)-middle)/span),
        (load_weights*station_balance[None, :]).sum(axis=1),
        np.full(c.n_zones, t*battery), t*balance)
    result = np.clip(np.column_stack(columns), -1., 1.)
    if result.shape != (c.n_zones, PARAMETER_COUNT) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or malformed pricing features')
    return result


class LearnedPricingPolicy:
    """A shared linear dollar-price target, rounded to a legal price level.

    All zones share twelve coefficients. Ties use the first configured level.
    Returned scenario and coefficient copies cannot mutate an existing actor.
    """
    def __init__(self, scenario, coefficients):
        self._scenario_data = _normalized(asdict(scenario.validate()))
        self._scenario = _scenario(self._scenario_data)
        values = np.asarray(coefficients)
        if (values.shape != (PARAMETER_COUNT,) or values.dtype.kind not in 'iuf'
                or not np.isfinite(values).all()
                or np.max(np.abs(values)) > np.finfo(float).max/(4*PARAMETER_COUNT)):
            raise ValueError(f'coefficients must contain {PARAMETER_COUNT} finite bounded real values')
        self._coefficients = values.astype(float, copy=True)
        self._coefficients.setflags(write=False)
        self._checkpoint_sha256 = None

    @property
    def scenario(self):
        return _scenario(self._scenario_data)

    @property
    def coefficients(self):
        result = self._coefficients.copy(); result.setflags(write=False)
        return result

    def prices(self, obs, rng=None):
        filtered = pricing_observation(obs, self._scenario)
        targets = _pricing_features(filtered, self._scenario) @ self._coefficients
        levels = np.asarray(self._scenario.price_levels)
        targets = np.clip(targets, levels.min(), levels.max())
        indices = np.abs(targets[:, None]-levels).argmin(axis=1).astype(np.int64)
        indices[~filtered['pricing_mask'].any(axis=1)] = 0
        return indices

    def provenance(self):
        identity = dict(schema=CHECKPOINT_SCHEMA, feature_schema=FEATURE_SCHEMA,
            scenario=_normalized(self._scenario_data), coefficients=self._coefficients.tolist())
        return dict(identity, checkpoint_sha256=self._checkpoint_sha256,
            policy_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True,
            allow_nan=False).encode()).hexdigest(), feature_names=list(FEATURE_NAMES),
            parameter_count=PARAMETER_COUNT, source_sha256=_sources(),
            information='Prescribed rates, fleet state and observed electricity prices; no realized requests or choices.',
            output='Nearest configured legal dollar price; first index wins ties; unavailable zones use index zero.')

    def save_checkpoint(self, path):
        path = Path(path)
        metadata = self.provenance()
        metadata['checkpoint_sha256'] = None  # A file cannot contain its own digest.
        with path.open('xb') as stream:
            np.savez_compressed(stream, coefficients=self._coefficients,
                                metadata=np.asarray(json.dumps(metadata, allow_nan=False)))

    @classmethod
    def load_checkpoint(cls, path):
        path = Path(path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError('Pricing checkpoint exceeds byte budget')
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if ({e.filename for e in entries} != {'coefficients.npy', 'metadata.npy'}
                    or len(entries) != 2 or sum(e.file_size for e in entries) > MAX_CHECKPOINT_BYTES):
                raise ValueError('Pricing checkpoint has unexpected arrays or excessive size')
            for entry in entries:
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported pricing array format')
                    expected_shape = (PARAMETER_COUNT,) if entry.filename == 'coefficients.npy' else ()
                    expected_kind = 'iuf' if entry.filename == 'coefficients.npy' else 'U'
                    if (shape != expected_shape or dtype.kind not in expected_kind or order
                            or dtype.itemsize*np.prod(shape, dtype=np.int64) != entry.file_size-stream.tell()):
                        raise ValueError('Malformed pricing array header or payload')
        with np.load(path, allow_pickle=False) as data:
            meta = data['metadata']
            if meta.shape != () or meta.dtype.kind != 'U':
                raise ValueError('Pricing checkpoint metadata must be a Unicode scalar')
            def reject(value): raise ValueError('Nonfinite checkpoint metadata: '+value)
            metadata = json.loads(str(meta), parse_constant=reject)
            if metadata.get('schema') != CHECKPOINT_SCHEMA or metadata.get('feature_schema') != FEATURE_SCHEMA:
                raise ValueError('Unsupported pricing checkpoint schema')
            policy = cls(_scenario(metadata['scenario']), data['coefficients'])
        expected = policy.provenance()
        if (metadata.get('coefficients') != expected['coefficients']
                or metadata.get('policy_sha256') != expected['policy_sha256']
                or metadata.get('feature_names') != list(FEATURE_NAMES)
                or metadata.get('parameter_count') != PARAMETER_COUNT):
            raise ValueError('Pricing checkpoint identity or feature definition differs')
        policy._checkpoint_sha256 = _hash(path)
        return policy


@dataclass(frozen=True)
class PriceSearchConfig:
    seed: int = 7
    generations: int = 12
    population: int = 24
    elite_count: int = 6
    episodes_per_generation: int = 16
    training_seed_base: int = 2_100_000_000
    planning_seed: int | None = None
    action_seed: int = 719991
    initial_std: float = .15
    minimum_std: float = .02
    smoothing: float = .6
    coefficient_bound: float = 1.
    fixed_anchor_indices: tuple | None = None

    def validate(self, scenario):
        scenario.validate()
        for name in ('seed', 'training_seed_base', 'action_seed'):
            _integer(getattr(self, name), name)
        for name in ('generations', 'population', 'elite_count', 'episodes_per_generation'):
            _integer(getattr(self, name), name, 1)
        if self.planning_seed is not None: _integer(self.planning_seed, 'planning_seed')
        if self.training_seed_base % 2:
            raise ValueError('training_seed_base must be even')
        if self.elite_count > self.population or self.population < len(self.anchors(scenario))+2:
            raise ValueError('Population must hold anchors, incumbent, mean and all elites')
        if self.generations*self.population*self.episodes_per_generation > MAX_CANDIDATE_EPISODES:
            raise ValueError('Pricing search exceeds the episode budget')
        for name in ('initial_std', 'minimum_std', 'coefficient_bound'):
            _real(getattr(self, name), name, minimum=1e-12)
        _real(self.smoothing, 'smoothing', minimum=1e-12, maximum=1.)
        if self.minimum_std > self.initial_std:
            raise ValueError('minimum_std exceeds initial_std')
        anchors = self.anchors(scenario)
        if len(set(anchors)) != len(anchors): raise ValueError('Fixed-price anchors must be unique')
        for index in anchors:
            _integer(index, 'fixed_anchor_index')
            if index >= len(scenario.price_levels): raise ValueError('Fixed-price anchor is outside price levels')
        return self

    def anchors(self, scenario):
        return tuple(range(len(scenario.price_levels))) if self.fixed_anchor_indices is None else tuple(self.fixed_anchor_indices)

    def generation_seeds(self):
        return [[int(self.training_seed_base+2*(g*self.episodes_per_generation+i))
            for i in range(self.episodes_per_generation)] for g in range(self.generations)]


def fit_price_policy(scenario, config, backend_factory, output_dir, initial_policy=None):
    """Fit pricing using paired, original-return cross-entropy policy search.

    The factory receives ``(pricer, planning_seed)`` and returns an object with
    ``prices``, ``dispatch`` and ``reset_episode(env_seed, action_seed)``. A new
    backend is constructed for each candidate. Every candidate in a generation
    sees the same fresh environment bank and planning seed. The final actor is
    the last generation's winner, with the previous incumbent always included;
    scores from different fresh banks are never directly ranked against one
    another. Returns ``(policy, summary)``; full candidate evidence is on disk.
    """
    scenario, config = scenario.validate(), config.validate(scenario)
    if not callable(backend_factory): raise ValueError('backend_factory must be callable')
    if initial_policy is not None and (not isinstance(initial_policy, LearnedPricingPolicy)
            or initial_policy.scenario != scenario):
        raise ValueError('Initial pricing policy must match this scenario')
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Pricing output directory contains an earlier run')
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    levels = np.asarray(scenario.price_levels)
    span = max(float(np.ptp(levels)), 1e-6)
    lower = np.full(PARAMETER_COUNT, -config.coefficient_bound*span)
    upper = -lower
    lower[0], upper[0] = levels.min(), levels.max()
    if initial_policy is None:
        initial = np.zeros(PARAMETER_COUNT); initial[0] = levels[len(levels)//2]
        initial_policy = LearnedPricingPolicy(scenario, initial)
    incumbent = initial_policy.coefficients.copy()
    if (incumbent < lower).any() or (incumbent > upper).any():
        raise ValueError('Initial policy exceeds configured coefficient bounds')
    mean, std = incumbent.copy(), np.full(PARAMETER_COUNT, config.initial_std*span)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, config.training_seed_base, 0x50524943]))
    plan_seed = config.seed if config.planning_seed is None else config.planning_seed
    seed_banks = config.generation_seeds()
    source_hashes = _sources()
    try:
        factory_file = inspect.getsourcefile(backend_factory)
    except TypeError:
        factory_file = inspect.getsourcefile(type(backend_factory))
    factory_identity = dict(module=getattr(backend_factory, '__module__', ''),
        name=getattr(backend_factory, '__qualname__', type(backend_factory).__qualname__),
        source_file=str(factory_file) if factory_file else None,
        source_sha256=_hash(factory_file) if factory_file and Path(factory_file).is_file() else None)
    protocol = dict(schema='charging_env.price_search.v1', scenario=_normalized(asdict(scenario)),
        config=_normalized(asdict(config)), feature_schema=FEATURE_SCHEMA,
        feature_names=list(FEATURE_NAMES), parameter_count=PARAMETER_COUNT,
        generation_seeds=seed_banks, planning_seed=plan_seed, backend_factory=factory_identity,
        initial_policy_sha256=initial_policy.provenance()['policy_sha256'], source_sha256=source_hashes,
        expected_candidate_evaluations=config.generations*config.population,
        expected_environment_episodes=config.generations*config.population*config.episodes_per_generation,
        objective='Full-episode original discounted business profit under the supplied backend.',
        selection='Last-generation winner; incumbent retained and reevaluated on each fresh common bank.',
        uncertainty='Training scores are optimization evidence, not independent performance estimates.')
    _write(directory/'protocol.json', protocol)
    history, total_counters, candidates_evaluated = [], {}, 0
    for generation, seeds in enumerate(seed_banks):
        proposals, seen = [], set()
        def add(values, kind):
            value = np.clip(np.asarray(values, dtype=float), lower, upper)
            key = tuple(value)
            if key not in seen:
                proposals.append((value.copy(), kind)); seen.add(key)
        add(incumbent, 'incumbent'); add(mean, 'search_mean')
        for index in config.anchors(scenario):
            anchor = np.zeros(PARAMETER_COUNT); anchor[0] = levels[index]
            add(anchor, f'fixed_price_{index}')
        while len(proposals) < config.population:
            noise = rng.standard_normal(PARAMETER_COUNT)
            add(mean+std*noise, 'sample')
            if len(proposals) < config.population: add(mean-std*noise, 'antithetic_sample')
        records = []
        for index, (coefficients, kind) in enumerate(proposals):
            pricer = LearnedPricingPolicy(scenario, coefficients)
            backend = backend_factory(pricer, int(plan_seed))
            if any(not callable(getattr(backend, name, None)) for name in ('prices', 'dispatch', 'reset_episode')):
                raise ValueError('Backend must expose prices, dispatch and reset_episode')
            result = evaluate_policy(scenario, seeds, backend.prices, backend.dispatch,
                action_seed=config.action_seed, label='pricing_search', on_episode_start=backend.reset_episode)
            profits = np.asarray([episode['discounted_profit'] for episode in result['episodes']])
            if result['invalid_actions'] or not np.isfinite(profits).all():
                raise ValueError('Pricing search encountered invalid actions or nonfinite profit')
            counters = {key: int(getattr(backend, 'statistics', {}).get(key, 0)) for key in _COUNTERS}
            if any(value < 0 for value in counters.values()): raise ValueError('Negative backend query counter')
            for key, value in counters.items(): total_counters[key] = total_counters.get(key, 0)+value
            records.append(dict(candidate=index, kind=kind, coefficients=coefficients.tolist(),
                mean_discounted_profit=float(profits.mean()), episode_discounted_profits=profits.tolist(),
                invalid_actions=0, backend_counters=counters,
                mean_policy_decision_ms=result['mean_policy_decision_ms']))
            candidates_evaluated += 1
        scores = np.asarray([row['mean_discounted_profit'] for row in records])
        order = np.argsort(-scores, kind='stable')
        winner = int(order[0]); incumbent = proposals[winner][0].copy()
        elites = np.stack([proposals[int(i)][0] for i in order[:config.elite_count]])
        mean = np.clip((1-config.smoothing)*mean+config.smoothing*elites.mean(axis=0), lower, upper)
        std = np.maximum(config.minimum_std*span,
            (1-config.smoothing)*std+config.smoothing*elites.std(axis=0))
        history.append(dict(generation=generation, environment_seeds=seeds, planning_seed=int(plan_seed),
            candidates=records, winner_index=winner, incumbent_score=float(scores[0]),
            winner_score=float(scores[winner]), search_mean=mean.tolist(), search_std=std.tolist()))
        _write(directory/'training_history.json', history)
        print(f'Pricing generation {generation+1}/{config.generations}: '
              f'{scores[winner]:.3f} (incumbent {scores[0]:.3f}, {len(records)} candidates)', flush=True)
    if _sources() != source_hashes:
        raise RuntimeError('Pricing training source changed during the run')
    if factory_identity['source_sha256'] and _hash(factory_file) != factory_identity['source_sha256']:
        raise RuntimeError('Pricing backend factory source changed during the run')
    policy = LearnedPricingPolicy(scenario, incumbent)
    checkpoint = directory/'checkpoint.npz'; policy.save_checkpoint(checkpoint)
    restored = LearnedPricingPolicy.load_checkpoint(checkpoint)
    if restored.provenance()['policy_sha256'] != policy.provenance()['policy_sha256']:
        raise RuntimeError('Pricing checkpoint reload changed its policy')
    summary = dict(schema='charging_env.price_search_summary.v1', config=_normalized(asdict(config)),
        seed=config.seed, planning_seed=int(plan_seed), parameter_count=PARAMETER_COUNT,
        feature_schema=FEATURE_SCHEMA, generations=len(history), generation_seeds=seed_banks,
        candidate_evaluations=candidates_evaluated,
        environment_episodes=candidates_evaluated*config.episodes_per_generation,
        environment_periods=candidates_evaluated*config.episodes_per_generation*scenario.horizon,
        backend_counters=total_counters, invalid_actions=0, all_returns_finite=True,
        final_mean_discounted_profit=history[-1]['winner_score'], final_candidate_index=history[-1]['winner_index'],
        policy_sha256=policy.provenance()['policy_sha256'], checkpoint_reload_matches=True,
        checkpoint_sha256=_hash(checkpoint), protocol_sha256=_hash(directory/'protocol.json'),
        training_history_sha256=_hash(directory/'training_history.json'), source_sha256=source_hashes,
        backend_factory=factory_identity, seconds=time.perf_counter()-started)
    _write(directory/'summary.json', summary)
    return policy, summary
