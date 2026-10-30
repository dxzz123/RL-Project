"""Estimate a planning model from observable quote opportunities.

The assumed families are time/zone Poisson arrivals, independent uniform request
sizes, and multinomial-logit choice. They are modeling assumptions, not quantities
learned from these logs. Every arrival, including an outside choice, must be
logged; completed transactions alone cannot identify demand or participation.
Only the explicit public physical/tariff allowlist may enter from a Scenario.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from ..config import Scenario


PUBLIC_FIELDS = (
    'horizon', 'period_hours', 'zone_xy_km', 'fcs_zones', 'initial_locations',
    'battery_capacity_kwh', 'initial_energy_kwh', 'reserve_kwh',
    'fcs_retail_per_kwh', 'grid_buy_per_kwh', 'grid_sell_per_kwh',
    'fcs_charge_price_profile', 'fcs_discharge_price_profile',
    'price_noise_distribution', 'price_noise_scale', 'price_min', 'price_max',
    'price_levels', 'max_fcs_access_km', 'speed_kmph', 'travel_kwh_per_km',
    'travel_cost_per_km', 'mobile_service_kw', 'mobile_charge_kw',
    'mobile_discharge_kw', 'charge_efficiency', 'mobile_service_efficiency',
    'discharge_efficiency', 'fixed_service_efficiency', 'fcs_ev_capacity_kwh',
    'fcs_fleet_exchange_capacity_kwh', 'operating_cost_per_mcv_period',
    'unmet_penalty_per_kwh', 'terminal_value_per_stored_kwh', 'discount_per_period',
    # Posted tariff structure: the utility's time-of-use bands, the station
    # service fee, and the premium menu are observable, like price_levels above.
    # Demand parameters stay private and are replaced by neutral placeholders.
    'fcs_pricing_mode', 'period_types', 'first_period_hour_of_day',
    'fcs_base_price_mean', 'fcs_base_price_halfwidth', 'fcs_service_fee_rate',
    'mcv_pricing_mode', 'mcv_premium_levels', 'mcv_premium_reference',
)
COEFFICIENT_NAMES = ('service_intercept', 'payment', 'mobile_wait', 'fixed_wait', 'fixed_travel')
LOG_ARRAYS = ('counts', 'request_day', 'request_period', 'request_zone',
    'request_energy_kwh', 'alternative_payment', 'mobile_wait_hours',
    'fixed_wait_hours', 'fixed_travel_km', 'available', 'chosen', 'day_ids')
LOG_SCHEMA = 'charging_env.observable_quote_log.v1'
MODEL_SCHEMA = 'charging_env.history_model.v1'
MAX_LOG_BYTES = 512 * 1024**2
MAX_MODEL_BYTES = 16 * 1024**2


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _normalized(value):
    return json.loads(_json(value))


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def public_model_config(scenario):
    """Copy known physical and tariff fields, never behavioral truth fields."""
    if not isinstance(scenario, Scenario):
        raise ValueError('Expected a Scenario as the public configuration source')
    public = _normalized({name: getattr(scenario, name) for name in PUBLIC_FIELDS})
    public_scenario(public)  # Validate only the allowlisted fields.
    return public


def public_scenario(public_config):
    """Neutral physical adapter; its placeholder behavior is not an estimate."""
    if not isinstance(public_config, dict) or set(public_config) != set(PUBLIC_FIELDS):
        raise ValueError('Public configuration must contain exactly PUBLIC_FIELDS')
    public = _normalized(public_config)
    h, z = public['horizon'], len(public['zone_xy_km'])
    if (isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= 1000
            or not 1 <= z <= 4096 or len(public['initial_locations']) > 1024):
        raise ValueError('Public configuration exceeds supported dimensions')
    return Scenario(**{k: _tuples(v) for k, v in public.items()},
        arrival_rates=tuple((0.,)*z for _ in range(h)),
        mean_requests_by_zone=(0.,)*z, arrival_profile=(1.,)*h,
        request_min_kwh=1e-6, request_max_kwh=1e6, request_kwh=1.,
        max_requests_per_zone=1, charging_utility_intercept=0.,
        price_sensitivity_per_dollar=1., mobile_wait_cost_per_hour=0.,
        fixed_wait_cost_per_hour=0., fixed_travel_cost_per_km=0.).validate()


def _array(value, name, shape=None, integer=False, boolean=False):
    a = np.asarray(value)
    accepted = 'b' if boolean else ('iu' if integer else 'fiu')
    if a.dtype.kind not in accepted or (shape is not None and a.shape != shape):
        raise ValueError(f'Invalid {name} shape or dtype')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite')
    if integer and a.size and (a.min() < 0 or a.max() > np.iinfo(np.int64).max):
        raise ValueError(f'{name} must contain nonnegative int64 values')
    result = np.array(a, dtype=bool if boolean else (np.int64 if integer else np.float64), copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(metadata, arrays):
    h = hashlib.sha256(_json(metadata).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        h.update(_json([name, array.dtype.str, list(array.shape)]).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _save(path, metadata, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, metadata=np.asarray(_json(metadata)), **arrays)


def _load(path, expected_arrays, byte_budget):
    """Validate ZIP/NPY byte lengths before NumPy allocates advertised shapes."""
    path = Path(path)
    if path.stat().st_size > byte_budget:
        raise ValueError('Archive exceeds its byte budget')
    data = path.read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            expected = {name+'.npy' for name in expected_arrays} | {'metadata.npy'}
            if len(members) != len(expected) or {m.filename for m in members} != expected:
                raise ValueError('Archive members differ from the schema')
            if sum(m.file_size for m in members) > byte_budget:
                raise ValueError('Expanded archive exceeds its byte budget')
            for member in members:
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported NPY version')
                    if dtype.hasobject or dtype.kind not in 'fiuUb' or len(shape) > 3:
                        raise ValueError('Unsupported array dtype or rank')
                    size = int(dtype.itemsize)
                    for dimension in shape:
                        if dimension < 0: raise ValueError('Negative array dimension')
                        size *= dimension
                    if size > byte_budget or size != member.file_size-stream.tell():
                        raise ValueError('Array header exceeds available data or byte budget')
                    if member.filename == 'metadata.npy' and (shape != () or size > 4*1024**2):
                        raise ValueError('Metadata must be a bounded scalar string')
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays['metadata']), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            return metadata, {name: arrays[name].copy() for name in expected_arrays}
    except (OSError, zipfile.BadZipFile, EOFError, TypeError) as error:
        raise ValueError('Invalid model/log archive') from error


class HistoricalLog:
    """Complete observable quote opportunities with zero-inclusive daily counts.

    Alternative columns are mobile, each FCS, then outside. Payments are total
    dollars for this request, waits are hours, and station travel is kilometers.
    ``request_day`` indexes the first counts axis; ``day_ids`` are abstract
    chronological IDs, with generation seeds kept separately by the experiment.
    """
    def __init__(self, public_config, counts, request_day, request_period, request_zone,
                 request_energy_kwh, alternative_payment, mobile_wait_hours,
                 fixed_wait_hours, fixed_travel_km, available, chosen, day_ids):
        c = public_scenario(public_config)
        self._public_config = _normalized(public_config)
        counts = np.asarray(counts)
        if (counts.ndim != 3 or counts.shape[1:] != (c.horizon, c.n_zones)
                or not 1 <= counts.shape[0] <= 10000 or counts.size > 5_000_000):
            raise ValueError('counts must be bounded [days, horizon, zones]')
        self.counts = _array(counts, 'counts', integer=True)
        n = int(self.counts.sum(dtype=np.float64))
        if n > 2_000_000 or n*(c.n_fcs+2) > 20_000_000:
            raise ValueError('Historical log exceeds its request/alternative budget')
        days = self.counts.shape[0]
        for name, value, shape, integer, boolean in (
            ('request_day', request_day, (n,), True, False),
            ('request_period', request_period, (n,), True, False),
            ('request_zone', request_zone, (n,), True, False),
            ('request_energy_kwh', request_energy_kwh, (n,), False, False),
            ('alternative_payment', alternative_payment, (n, c.n_fcs+2), False, False),
            ('mobile_wait_hours', mobile_wait_hours, (n,), False, False),
            ('fixed_wait_hours', fixed_wait_hours, (n, c.n_fcs), False, False),
            ('fixed_travel_km', fixed_travel_km, (n, c.n_fcs), False, False),
            ('available', available, (n, c.n_fcs+2), False, True),
            ('chosen', chosen, (n,), True, False),
            ('day_ids', day_ids, (days,), True, False),
        ):
            setattr(self, name, _array(value, name, shape, integer, boolean))
        if (np.unique(self.day_ids).size != days or (np.diff(self.day_ids) <= 0).any()
                or (self.request_day >= days).any() or (self.request_period >= c.horizon).any()
                or (self.request_zone >= c.n_zones).any() or (self.chosen >= c.n_fcs+2).any()):
            raise ValueError('Invalid chronological IDs or request indices')
        for name in ('alternative_payment', 'mobile_wait_hours', 'fixed_wait_hours', 'fixed_travel_km'):
            if (getattr(self, name) < 0).any(): raise ValueError(name+' must be nonnegative')
        if ((self.request_energy_kwh <= 0).any() or not self.available[:, -1].all()
                or (self.alternative_payment[:, -1] != 0).any()
                or not self.available[np.arange(n), self.chosen].all()):
            raise ValueError('Invalid energy, outside option, or unavailable chosen alternative')
        cell = (self.request_day*c.horizon+self.request_period)*c.n_zones+self.request_zone
        observed = np.bincount(cell, minlength=self.counts.size).reshape(self.counts.shape)
        if not np.array_equal(observed, self.counts):
            raise ValueError('Request records do not match the complete day counts')

    @property
    def public_config(self):
        return deepcopy(self._public_config)

    @property
    def n_requests(self):
        return self.request_energy_kwh.size

    def provenance(self):
        metadata = dict(schema=LOG_SCHEMA, public_config=self._public_config)
        return dict(schema=LOG_SCHEMA, log_sha256=_fingerprint(metadata, {n: getattr(self, n) for n in LOG_ARRAYS}),
            days=self.counts.shape[0], requests=self.n_requests, day_ids=self.day_ids.tolist())

    def save(self, path):
        metadata = dict(schema=LOG_SCHEMA, public_config=self._public_config, log_sha256=self.provenance()['log_sha256'])
        _save(path, metadata, {n: getattr(self, n) for n in LOG_ARRAYS})

    @classmethod
    def load(cls, path):
        metadata, arrays = _load(path, LOG_ARRAYS, MAX_LOG_BYTES)
        if set(metadata) != {'schema', 'public_config', 'log_sha256'} or metadata['schema'] != LOG_SCHEMA:
            raise ValueError('Unknown historical log schema')
        result = cls(metadata['public_config'], **arrays)
        if result.provenance()['log_sha256'] != metadata['log_sha256']:
            raise ValueError('Historical log fingerprint differs')
        return result


@dataclass(frozen=True)
class HistoryFitConfig:
    demand_pseudodays: float = 1.
    choice_ridge: float = 1e-6
    max_iterations: int = 80
    gradient_tolerance: float = 1e-7
    batch_size: int = 8192
    minimum_payment_coefficient: float = 1e-5
    maximum_cost_coefficient: float = 100.
    maximum_abs_intercept: float = 30.

    def validate(self):
        for name in ('max_iterations', 'batch_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name+' must be a positive integer')
        if self.max_iterations > 500 or self.batch_size > 65536:
            raise ValueError('Fit exceeds its iteration/batch budget')
        for name in ('demand_pseudodays', 'choice_ridge', 'gradient_tolerance',
                     'minimum_payment_coefficient', 'maximum_cost_coefficient', 'maximum_abs_intercept'):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError('Invalid '+name)
        if (min(self.gradient_tolerance, self.minimum_payment_coefficient,
                self.maximum_abs_intercept) <= 0 or self.maximum_cost_coefficient <= self.minimum_payment_coefficient):
            raise ValueError('Invalid fit tolerance or coefficient bounds')
        return self


def estimate_arrival_rates(counts, pseudodays=1.):
    """Shrink each time/zone mean toward its observed same-zone daily mean."""
    counts = _array(counts, 'counts', integer=True)
    if counts.ndim != 3 or min(counts.shape) < 1:
        raise ValueError('counts must be nonempty [days, horizon, zones]')
    if isinstance(pseudodays, bool) or not np.isfinite(pseudodays) or pseudodays < 0:
        raise ValueError('pseudodays must be finite and nonnegative')
    prior = counts.mean(axis=(0, 1))
    return (counts.sum(axis=0, dtype=float)+pseudodays*prior)/(counts.shape[0]+pseudodays)


def _features(log, start, stop):
    """Positive preference coefficients multiply negative cost attributes."""
    x = np.zeros((stop-start, log.available.shape[1], 5))
    x[:, :-1, 0] = 1.
    x[:, :, 1] = -log.alternative_payment[start:stop]
    x[:, 0, 2] = -log.mobile_wait_hours[start:stop]
    x[:, 1:-1, 3] = -log.fixed_wait_hours[start:stop]
    x[:, 1:-1, 4] = -log.fixed_travel_km[start:stop]
    return x


def _objective(log, w, scales, batch_size, ridge=0., derivatives=True):
    loss = 0.; gradient = np.zeros(5); hessian = np.zeros((5, 5))
    for start in range(0, log.n_requests, batch_size):
        stop = min(start+batch_size, log.n_requests)
        x = _features(log, start, stop)/scales
        utility = x @ w
        utility[~log.available[start:stop]] = -np.inf
        shift = utility.max(axis=1)
        weights = np.exp(utility-shift[:, None]); total = weights.sum(axis=1)
        selected = log.chosen[start:stop]; rows = np.arange(stop-start)
        loss += float((np.log(total)+shift-utility[rows, selected]).sum())
        if derivatives:
            probabilities = weights/total[:, None]
            mean = np.einsum('na,nak->nk', probabilities, x)
            gradient += (mean-x[rows, selected]).sum(axis=0)
            hessian += np.einsum('na,nai,naj->ij', probabilities, x, x)-mean.T@mean
    n = log.n_requests
    return (loss/n+.5*ridge*float(w@w), gradient/n+ridge*w,
            hessian/n+ridge*np.eye(5))


def _fit_choice(log, config):
    scales = np.zeros(5); alternatives = 0
    for start in range(0, log.n_requests, config.batch_size):
        stop = min(start+config.batch_size, log.n_requests)
        x = _features(log, start, stop); mask = log.available[start:stop].copy(); mask[:, -1] = False
        scales += (x*x*mask[:, :, None]).sum(axis=(0, 1)); alternatives += int(mask.sum())
    scales = np.sqrt(scales/max(1, alternatives)); scales[scales < 1e-12] = 1.
    low = np.array([-config.maximum_abs_intercept, config.minimum_payment_coefficient, 0., 0., 0.])*scales
    high = np.array([config.maximum_abs_intercept]+[config.maximum_cost_coefficient]*4)*scales
    w = np.zeros(5); w[1] = max(config.minimum_payment_coefficient*scales[1], .1)
    w = np.clip(w, low, high)
    initial, _, _ = _objective(log, w, scales, config.batch_size)
    converged, reason, trace = False, 'maximum_iterations', []
    for iteration in range(config.max_iterations):
        loss, gradient, hessian = _objective(log, w, scales, config.batch_size, config.choice_ridge)
        free = ~(((w <= low+1e-10) & (gradient > 0)) | ((w >= high-1e-10) & (gradient < 0)))
        projected = np.where(free, gradient, 0.)
        trace.append(float(loss))
        if np.max(np.abs(projected)) <= config.gradient_tolerance:
            converged, reason = True, 'projected_gradient'; break
        direction = np.zeros(5)
        direction[free] = -np.linalg.solve(hessian[np.ix_(free, free)]+1e-10*np.eye(int(free.sum())), gradient[free])
        if gradient@direction >= 0: direction = -projected
        accepted = False
        for attempt in range(30):
            proposed = np.clip(w+(0.5**attempt)*direction, low, high)
            step = proposed-w
            if gradient@step >= 0: continue
            next_loss, _, _ = _objective(log, proposed, scales, config.batch_size, config.choice_ridge, False)
            if next_loss <= loss+1e-4*float(gradient@step):
                w, accepted = proposed, True; break
        if not accepted:
            reason = 'line_search_stalled'; break
    raw_loss, _, fisher = _objective(log, w, scales, config.batch_size)
    _, gradient, _ = _objective(log, w, scales, config.batch_size, config.choice_ridge)
    projected = gradient.copy()
    projected[((w <= low+1e-9) & (gradient > 0)) | ((w >= high-1e-9) & (gradient < 0))] = 0.
    eigenvalues = np.linalg.eigvalsh(fisher)
    threshold = max(1e-12, float(eigenvalues[-1])*1e-8)
    rank = int((eigenvalues > threshold).sum())
    standard_errors = None if rank < 5 else (np.sqrt(np.diag(np.linalg.inv(fisher))/log.n_requests)/scales).tolist()
    return w/scales, dict(converged=converged, stop_reason=reason, iterations=len(trace),
        initial_mean_negative_log_likelihood=initial, mean_negative_log_likelihood=raw_loss,
        penalized_objective_trace=trace, maximum_projected_gradient=float(np.abs(projected).max()),
        gradient_standardized=gradient.tolist(), feature_scales=scales.tolist(),
        fisher_eigenvalues_standardized=eigenvalues.tolist(), fisher_rank=rank,
        fisher_condition_number=None if rank < 5 else float(eigenvalues[-1]/eigenvalues[0]),
        weakly_identified=bool(rank < 5 or (rank == 5 and eigenvalues[-1]/eigenvalues[0] > 1e8)),
        approximate_coefficient_standard_errors=standard_errors,
        at_coefficient_bounds=((np.abs(w-low) < 1e-8) | (np.abs(w-high) < 1e-8)).tolist(),
        standard_error_scope='Conditional model-based local Fisher approximation; not robust to misspecification or correlated days.')


class HistoricalModel:
    def __init__(self, public_config, arrival_rates, request_bounds, choice_coefficients, fit_report):
        base = public_scenario(public_config)
        self._public = _normalized(public_config)
        self._arrival = _array(arrival_rates, 'arrival_rates', (base.horizon, base.n_zones))
        self._bounds = _array(request_bounds, 'request_bounds', (2,))
        self._coefficients = _array(choice_coefficients, 'choice_coefficients', (5,))
        if ((self._arrival < 0).any() or self._bounds[0] <= 0 or self._bounds[1] <= self._bounds[0]
                or self._coefficients[1] <= 0 or (self._coefficients[2:] < 0).any()):
            raise ValueError('Invalid estimated model parameters')
        self._report = _normalized(fit_report)
        self._checkpoint_sha256 = None
        self.scenario.validate()

    @property
    def choice_coefficients(self):
        result = self._coefficients.copy(); result.setflags(write=False); return result

    @property
    def scenario(self):
        b, eta, mobile, fixed, travel = self._coefficients
        return Scenario(**{k: _tuples(v) for k, v in self._public.items()},
            arrival_rates=tuple(tuple(float(value) for value in row) for row in self._arrival),
            mean_requests_by_zone=tuple(float(value) for value in self._arrival.mean(axis=0)),
            arrival_profile=(1.,)*self._arrival.shape[0],
            request_min_kwh=float(self._bounds[0]), request_max_kwh=float(self._bounds[1]),
            request_kwh=float(self._bounds.mean()), max_requests_per_zone=1,
            charging_utility_intercept=float(b), price_sensitivity_per_dollar=float(eta),
            mobile_wait_cost_per_hour=float(mobile/eta), fixed_wait_cost_per_hour=float(fixed/eta),
            fixed_travel_cost_per_km=float(travel/eta)).validate()

    def choice_log_likelihood(self, log):
        if log.public_config != self._public or not log.n_requests:
            raise ValueError('Likelihood requires nonempty logs with matching public configuration')
        loss, _, _ = _objective(log, self._coefficients, np.ones(5), 8192, derivatives=False)
        return dict(requests=log.n_requests, total_log_likelihood=-loss*log.n_requests,
            mean_log_likelihood=-loss, mean_negative_log_likelihood=loss)

    def diagnostics(self, log):
        """Observable predictive checks; never uses a true behavioral Scenario.

        The demand baseline uses the fitted log's constant mean for each zone.
        Poisson deviance uses a disclosed 1e-12 rate floor so an unseen arrival
        in a fitted-zero cell has a finite report; those cells are also counted.
        Callers, not this method, decide and record the training/holdout split.
        """
        if not isinstance(log, HistoricalLog) or log.public_config != self._public:
            raise ValueError('Diagnostics require matching public configuration')
        counts = log.counts.astype(float)
        def demand_metrics(rates):
            prediction = np.broadcast_to(rates, counts.shape)
            floored = np.maximum(prediction, 1e-12)
            positive = counts > 0
            terms = floored-counts
            terms[positive] += counts[positive]*np.log(counts[positive]/floored[positive])
            return dict(predicted_mean_requests_per_day=float(rates.sum()),
                observed_mean_requests_per_day=float(counts.sum(axis=(1, 2)).mean()),
                per_day_count_rmse=float(np.sqrt(np.mean((counts-prediction)**2))),
                mean_cell_count_rmse=float(np.sqrt(np.mean((counts.mean(axis=0)-rates)**2))),
                mean_poisson_deviance=float(2*terms.mean()),
                positive_counts_at_zero_predicted_rate=int(np.count_nonzero(positive & (prediction == 0))))
        baseline = np.broadcast_to(self._arrival.mean(axis=0), self._arrival.shape)
        choice = dict(requests=0, mean_negative_log_likelihood=None, mean_multiclass_brier_score=None,
            predicted_alternative_shares=None, observed_alternative_shares=None)
        if log.n_requests:
            choice = self.choice_log_likelihood(log)
            brier, predicted = 0., np.zeros(log.available.shape[1])
            observed = np.bincount(log.chosen, minlength=predicted.size)
            for start in range(0, log.n_requests, 8192):
                stop = min(start+8192, log.n_requests)
                utility = _features(log, start, stop)@self._coefficients
                utility[~log.available[start:stop]] = -np.inf
                weights = np.exp(utility-utility.max(axis=1, keepdims=True))
                probabilities = weights/weights.sum(axis=1, keepdims=True)
                predicted += probabilities.sum(axis=0)
                brier += float((probabilities**2).sum()-2*probabilities[np.arange(stop-start), log.chosen[start:stop]].sum()+stop-start)
            choice.update(mean_multiclass_brier_score=brier/log.n_requests,
                predicted_alternative_shares=(predicted/log.n_requests).tolist(),
                observed_alternative_shares=(observed/log.n_requests).tolist())
        energy = log.request_energy_kwh
        return dict(log=log.provenance(), model_sha256=self.provenance()['model_sha256'],
            demand=demand_metrics(self._arrival), zone_constant_baseline=demand_metrics(baseline),
            poisson_rate_floor=1e-12, choice=choice,
            request_energy=dict(fitted_min_kwh=float(self._bounds[0]), fitted_max_kwh=float(self._bounds[1]),
                observed_min_kwh=float(energy.min()) if energy.size else None,
                observed_max_kwh=float(energy.max()) if energy.size else None,
                fraction_in_fitted_range=float(np.mean((energy >= self._bounds[0]) & (energy <= self._bounds[1]))) if energy.size else None),
            uses_behavioral_truth=False)

    def provenance(self):
        arrays = dict(arrival_rates=self._arrival, request_bounds=self._bounds,
            choice_coefficients=self._coefficients)
        return dict(schema=MODEL_SCHEMA, model_sha256=_fingerprint(dict(schema=MODEL_SCHEMA, public_config=self._public), arrays),
            public_config=deepcopy(self._public), coefficient_names=list(COEFFICIENT_NAMES),
            fit=deepcopy(self._report), checkpoint_sha256=self._checkpoint_sha256)

    def save_checkpoint(self, path):
        metadata = dict(schema=MODEL_SCHEMA, public_config=self._public, fit_report=self._report,
            model_sha256=self.provenance()['model_sha256'])
        _save(path, metadata, dict(arrival_rates=self._arrival, request_bounds=self._bounds,
            choice_coefficients=self._coefficients))

    @classmethod
    def load_checkpoint(cls, path):
        metadata, arrays = _load(path, ('arrival_rates', 'request_bounds', 'choice_coefficients'), MAX_MODEL_BYTES)
        if (set(metadata) != {'schema', 'public_config', 'fit_report', 'model_sha256'}
                or metadata['schema'] != MODEL_SCHEMA):
            raise ValueError('Unknown historical model schema')
        result = cls(metadata['public_config'], fit_report=metadata['fit_report'], **arrays)
        if result.provenance()['model_sha256'] != metadata['model_sha256']:
            raise ValueError('Model fingerprint differs')
        result._checkpoint_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return result


def fit_history_model(log, config=None):
    """Fit once from historical records; held-out histories are diagnostics only."""
    if not isinstance(log, HistoricalLog): raise ValueError('Expected a HistoricalLog')
    config = (HistoryFitConfig() if config is None else config).validate()
    if log.n_requests < 2:
        raise ValueError('At least two observed requests are needed to fit energy and choice')
    started = time.perf_counter()
    log_before = log.provenance()
    rates = estimate_arrival_rates(log.counts, config.demand_pseudodays)
    smallest, largest = float(log.request_energy_kwh.min()), float(log.request_energy_kwh.max())
    if largest > smallest:
        correction = (largest-smallest)/(log.n_requests-1)
        bounds = (max(np.finfo(float).eps, smallest-correction), largest+correction)
        bounds_method = 'Uniform endpoints: sample extrema expanded by observed range/(N-1), lower endpoint clipped positive.'
    else:
        width = max(smallest*.01, 1e-6)
        bounds = (max(np.finfo(float).eps, smallest-width), smallest+width)
        bounds_method = 'Degenerate sizes: observed value plus/minus 1 percent (minimum half-width 1e-6 kWh); endpoints unidentifiable.'
    coefficients, diagnostics = _fit_choice(log, config)
    report = dict(schema=MODEL_SCHEMA, config=asdict(config), historical_log=log_before,
        fit_days=log.counts.shape[0], fit_requests=log.n_requests,
        assumed_families=['Poisson independent period/zone arrivals', 'Uniform independent request energy', 'Multinomial logit with outside utility zero'],
        demand_smoothing='Each period/zone count mean shrinks toward the observed all-period mean for that same zone; zeros remain in the denominator.',
        request_bounds_method=bounds_method, choice=diagnostics,
        choice_coefficients=dict(zip(COEFFICIENT_NAMES, map(float, coefficients))),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_hidden_behavior_parameters=True, held_out_data_used=False, seconds=time.perf_counter()-started)
    if log.provenance() != log_before: raise ValueError('Historical log changed during fitting')
    model = HistoricalModel(log.public_config, rates, bounds, coefficients, report)
    return model, deepcopy(report)
