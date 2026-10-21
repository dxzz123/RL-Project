"""Finite-horizon fitted continuation values for a two-vehicle fleet.

The dispatch observation retains all current requests and customer choices. Only
the *future* value is compressed to time, both ordered vehicle locations, and
their battery energies. Training uses a generative simulator: it can set an
arbitrary feasible fleet state and sample its current-period uncertainty. This
is more information than an ordinary on-policy, model-free PPO rollout uses.
"""
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import json
from numbers import Real
from pathlib import Path
import time
import zipfile

import numpy as np

from ..config import Scenario
from ..simulator import ChargingSimulator, DispatchAction, Mode
from ..benchmarks.policies import fixed_prices, joint_dispatch_outcomes


CHECKPOINT_SCHEMA = "charging_env.fitted_value.v1"
MAX_VALUE_ENTRIES = 2_000_000
MAX_STATE_DRAW_QUERIES = 5_000_000
MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_SOURCE_FILES = (
    "rl/fitted_value.py", "rl/fitted_train.py", "benchmarks/policies.py",
    "simulator.py", "choice.py", "config.py",
)


def _json_dict(value):
    """Return a JSON-normalized mapping, including nested tuple conversion."""
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value, name, *, minimum=1):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def battery_grid(scenario, step):
    """Include the reserve and capacity, even when the last interval is short."""
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, Real) or not np.isfinite(step) or step <= 0:
        raise ValueError("grid_step must be finite and positive")
    width = scenario.battery_capacity_kwh - scenario.reserve_kwh
    # Refuse an oversized allocation before calling arange.
    if width / step > MAX_VALUE_ENTRIES:
        raise ValueError("Battery grid exceeds the fitted-value resource budget")
    grid = np.arange(scenario.reserve_kwh, scenario.battery_capacity_kwh, step, dtype=float)
    grid = grid[grid < scenario.battery_capacity_kwh]
    return np.append(grid, float(scenario.battery_capacity_kwh))


@dataclass(frozen=True)
class FittedValueConfig:
    seed: int = 7
    grid_step: float = 10.
    samples_per_period: int = 8
    price_index: int = 4
    energy_fractions: tuple = (.5, 1.)
    max_joint_actions: int = 200_000
    training_seed_base: int = 130_000_000

    def validate(self, scenario):
        scenario.validate()
        if scenario.n_mcv != 2:
            raise ValueError("Fitted continuation values currently require exactly two vehicles")
        for name in ("seed", "price_index", "training_seed_base"):
            _positive_integer(getattr(self, name), name, minimum=0)
        if self.training_seed_base % 2:
            raise ValueError("training_seed_base must be even to keep generative query seeds even")
        if self.price_index >= len(scenario.price_levels):
            raise ValueError("price_index is outside the scenario's admissible price levels")
        for name in ("samples_per_period", "max_joint_actions"):
            _positive_integer(getattr(self, name), name)
        fractions = np.asarray(self.energy_fractions)
        if (fractions.ndim != 1 or fractions.size == 0 or fractions.dtype.kind not in "iuf"
                or not np.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any()):
            raise ValueError("energy_fractions must be a nonempty finite sequence in [0, 1]")
        grid = battery_grid(scenario, self.grid_step)
        entries = (scenario.horizon + 1) * scenario.n_zones**2 * len(grid)**2
        queries = scenario.horizon * scenario.n_zones**2 * len(grid)**2 * self.samples_per_period
        if entries > MAX_VALUE_ENTRIES:
            raise ValueError(f"Value table exceeds {MAX_VALUE_ENTRIES} entries; use a coarser grid or smaller scenario")
        if queries > MAX_STATE_DRAW_QUERIES:
            raise ValueError(f"Training exceeds {MAX_STATE_DRAW_QUERIES} state/draw queries; reduce the sampling budget")
        return self


def _scenario_from_mapping(data):
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(Scenario)}:
        raise ValueError("Checkpoint scenario must contain exactly the Scenario fields")
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    return Scenario(**{key: tuples(value) for key, value in data.items()}).validate()


def _config_from_mapping(data, scenario):
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(FittedValueConfig)}:
        raise ValueError("Checkpoint training_config must contain exactly the FittedValueConfig fields")
    copied = dict(data)
    copied["energy_fractions"] = tuple(copied["energy_fractions"])
    return FittedValueConfig(**copied).validate(scenario)


def sample_dispatch_observation(scenario, period, locations, energies, seed, price_index=4):
    """Sample one synthetic dispatch observation from an arbitrary fleet state.

    This training-only generative interface intentionally sets simulator private
    state before calling its original period preparation and customer-choice
    logic. It neither steps through earlier periods nor peeks at future draws.
    Using the same seed at different fleet states reuses arrivals, request sizes,
    station shocks and choice uniforms; choice outcomes can correctly differ
    because mobile availability and waiting times depend on the fleet state.
    """
    _positive_integer(period, "period", minimum=0)
    _positive_integer(seed, "seed", minimum=0)
    if period >= scenario.horizon:
        raise ValueError("Synthetic dispatch period must precede the horizon")
    locations = np.asarray(locations)
    energies = np.asarray(energies, dtype=float)
    if (locations.shape != (scenario.n_mcv,) or locations.dtype.kind not in "iu"
            or (locations < 0).any() or (locations >= scenario.n_zones).any()):
        raise ValueError("Synthetic locations must be one valid integer node per vehicle")
    if (energies.shape != (scenario.n_mcv,) or not np.isfinite(energies).all()
            or (energies < scenario.reserve_kwh).any() or (energies > scenario.battery_capacity_kwh).any()):
        raise ValueError("Synthetic energy must lie between reserve and capacity")
    sim = ChargingSimulator(scenario)
    sim.rng = np.random.default_rng(int(seed))
    sim.period = int(period)
    sim.stage = 0
    sim.locations = locations.astype(np.int64, copy=True)
    sim.energy = energies.copy()
    sim.cumulative_profit = 0.
    sim.cumulative_discounted_profit = 0.
    sim._prepare_period()
    sim._ready = True
    levels = fixed_prices(sim.observe(), scenario, price_index)
    post, reward, done, truncated, _ = sim.quote_price_levels(levels)
    if reward != 0. or done or truncated:
        raise RuntimeError("Unexpected synthetic pricing transition")
    return post


class FittedValuePolicy:
    """Fixed pricing and exact current dispatch with interpolated future profit."""

    def __init__(self, scenario, config, energy_grid, values):
        self.scenario = scenario.validate()
        self.config = config.validate(scenario)
        self.training_scenario = self.scenario
        expected_grid = battery_grid(scenario, config.grid_step)
        raw_grid = np.asarray(energy_grid)
        if raw_grid.dtype.kind not in "iuf" or not np.isfinite(raw_grid).all():
            raise ValueError("energy_grid must contain finite real numeric entries")
        grid = raw_grid.astype(float)
        if grid.shape != expected_grid.shape or not np.array_equal(grid, expected_grid):
            raise ValueError("energy_grid must exactly match the configured reserve-to-capacity grid")
        values = np.asarray(values)
        shape = (scenario.horizon + 1, scenario.n_zones, scenario.n_zones, len(grid), len(grid))
        if values.shape != shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError(f"values must be a finite numeric table with shape {shape}")
        if np.any(values[-1] != 0.):
            raise ValueError("The continuation-value terminal layer must be exactly zero")
        self.energy_grid = grid.copy()
        self.values = values.astype(float, copy=True)
        self.training_provenance = {}

    def prices(self, obs, rng=None):
        """Return admissible fixed price indices; no policy RNG is consumed."""
        return fixed_prices(obs, self.scenario, self.config.price_index)

    def continuation(self, period, locations, energies):
        """Bilinear battery interpolation for ordered locations, optionally batched.

        Inputs end in a two-vehicle axis and may broadcast over leading axes.
        Energies outside the physical grid are rejected, except for floating
        point roundoff within 1e-8 kWh of reserve or capacity.
        """
        _positive_integer(period, "period", minimum=0)
        if period > self.scenario.horizon:
            raise ValueError("Continuation period exceeds the horizon")
        loc = np.asarray(locations)
        energy = np.asarray(energies, dtype=float)
        if (loc.ndim < 1 or loc.shape[-1] != 2 or loc.dtype.kind not in "iu"
                or (loc < 0).any() or (loc >= self.scenario.n_zones).any()):
            raise ValueError("locations must end in two valid integer node indices")
        if energy.ndim < 1 or energy.shape[-1] != 2 or not np.isfinite(energy).all():
            raise ValueError("energies must end in two finite vehicle energies")
        try:
            loc, energy = np.broadcast_arrays(loc, energy)
        except ValueError as exc:
            raise ValueError("locations and energies must have broadcast-compatible shapes") from exc
        loc = loc.astype(np.int64)
        grid = self.energy_grid
        if (energy < grid[0] - 1e-8).any() or (energy > grid[-1] + 1e-8).any():
            raise ValueError("Continuation energy is outside the reserve-to-capacity grid")
        energy = np.clip(energy, grid[0], grid[-1])
        table = self.values[int(period)]
        if len(grid) == 1:
            return table[loc[..., 0], loc[..., 1], 0, 0]
        lower = np.clip(np.searchsorted(grid, energy, side="right") - 1, 0, len(grid) - 2)
        upper = lower + 1
        weight = (energy - grid[lower]) / (grid[upper] - grid[lower])
        l0, l1 = lower[..., 0], lower[..., 1]
        u0, u1 = upper[..., 0], upper[..., 1]
        w0, w1 = weight[..., 0], weight[..., 1]
        x, y = loc[..., 0], loc[..., 1]
        return ((1. - w0) * (1. - w1) * table[x, y, l0, l1]
                + w0 * (1. - w1) * table[x, y, u0, l1]
                + (1. - w0) * w1 * table[x, y, l0, u1]
                + w0 * w1 * table[x, y, u0, u1])

    def dispatch(self, obs, rng=None):
        """Observe current demand, then choose the best feasible joint action."""
        if int(obs["stage"]) != 1:
            raise ValueError("Fitted-value dispatch requires the dispatch stage")
        period = int(np.asarray(obs["period"])[0])
        if not 0 <= period < self.scenario.horizon:
            raise ValueError("Dispatch period is outside the fitted horizon")
        modes, targets, energy, outcomes = joint_dispatch_outcomes(
            obs, self.scenario, self.config.energy_fractions, self.config.max_joint_actions)
        scores = outcomes["reward"] + self.scenario.discount_per_period * self.continuation(
            period + 1, outcomes["next_locations"], outcomes["next_energy_kwh"])
        winner = int(np.argmax(scores))
        return [DispatchAction(Mode(int(modes[winner, m])), int(targets[winner, m]),
                               float(energy[winner, m])) for m in range(2)]

    def with_scenario(self, scenario, allow_distribution_shift=False):
        """Reuse frozen values on the same scenario or an explicit demand/stock shift.

        Distribution shift permits only arrival-distribution fields and initial
        battery energy. Physics, prices, outside utility and the horizon must
        remain exactly those used for fitting. No values are adapted or refit.
        """
        if not isinstance(allow_distribution_shift, (bool, np.bool_)):
            raise ValueError("allow_distribution_shift must be boolean")
        scenario.validate()
        previous = _json_dict(asdict(self.training_scenario))
        current = _json_dict(asdict(scenario))
        changed = {key for key in previous if previous[key] != current[key]}
        allowed = {"arrival_rates", "arrival_profile", "mean_requests_by_zone", "initial_energy_kwh"}
        if changed and (not allow_distribution_shift or changed - allowed):
            raise ValueError(f"Scenario differs from training in unsupported fields: {sorted(changed)}")
        result = FittedValuePolicy(scenario, self.config, self.energy_grid, self.values)
        result.training_scenario = self.training_scenario
        result.training_provenance = _json_dict(self.training_provenance)
        return result

    def save_checkpoint(self, path):
        """Save numeric arrays and Unicode JSON only; existing files are protected."""
        path = Path(path)
        metadata = {"schema": CHECKPOINT_SCHEMA, "scenario": asdict(self.training_scenario),
                    "training_config": asdict(self.config), "training_provenance": self.training_provenance}
        encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, energy_grid=self.energy_grid, values=self.values,
                                metadata=np.asarray(encoded))

    @classmethod
    def load_checkpoint(cls, path):
        """Validate schema and arrays without unpickling executable objects."""
        path = Path(path)
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("Fitted-value checkpoint exceeds the file-size budget")
        try:
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                    raise ValueError("Fitted-value checkpoint exceeds the expanded-size budget")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"energy_grid", "values", "metadata"}:
                    raise ValueError("Unexpected fitted-value checkpoint arrays")
                raw = data["metadata"]
                if raw.shape != () or raw.dtype.kind != "U" or raw.nbytes > 1_000_000:
                    raise ValueError("Checkpoint metadata must be one bounded Unicode JSON string")
                metadata = json.loads(str(raw.item()))
                if (not isinstance(metadata, dict)
                        or set(metadata) != {"schema", "scenario", "training_config", "training_provenance"}
                        or metadata["schema"] != CHECKPOINT_SCHEMA
                        or not isinstance(metadata["training_provenance"], dict)):
                    raise ValueError("Unsupported fitted-value checkpoint schema")
                scenario = _scenario_from_mapping(metadata["scenario"])
                config = _config_from_mapping(metadata["training_config"], scenario)
                policy = cls(scenario, config, data["energy_grid"], data["values"])
                policy.training_provenance = _json_dict(metadata["training_provenance"])
                return policy
        except (OSError, zipfile.BadZipFile, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid fitted-value checkpoint: {exc}") from exc


class FittedValueTrainer:
    """Backward sample-average dynamic programming on the joint fleet grid."""

    def __init__(self, scenario=None, config=None):
        self.scenario = (scenario or Scenario()).validate()
        self.config = (config or FittedValueConfig()).validate(self.scenario)
        self.energy_grid = battery_grid(self.scenario, self.config.grid_step)
        self.policy = None

    def query_seeds(self):
        """One even seed per period/draw, reused at every fleet grid state."""
        c, h = self.config, self.scenario.horizon
        offset = c.seed * h * c.samples_per_period
        return [[int(c.training_seed_base + 2 * (offset + t * c.samples_per_period + k))
                 for k in range(c.samples_per_period)] for t in range(h)]

    def _source_hashes(self):
        package = Path(__file__).resolve().parents[1]
        return {"src/charging_env/" + name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in _SOURCE_FILES}

    def train(self, output_dir):
        """Fit once, save full provenance and counters, and return a JSON summary."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError("Output contains a previous run; use an empty output directory")
        scenario, config, grid = self.scenario, self.config, self.energy_grid
        shape = (scenario.horizon + 1, scenario.n_zones, scenario.n_zones, len(grid), len(grid))
        policy = FittedValuePolicy(scenario, config, grid, np.zeros(shape, dtype=float))
        seeds = self.query_seeds()
        states_per_period = scenario.n_zones**2 * len(grid)**2
        source_hashes = self._source_hashes()
        protocol = {
            "schema": CHECKPOINT_SCHEMA, "algorithm": "finite_horizon_fitted_continuation_value",
            "scenario": asdict(scenario), "training_config": asdict(config),
            "energy_grid_kwh": grid.tolist(), "value_shape": list(shape),
            "state_order": ["period", "vehicle_0_location", "vehicle_1_location", "vehicle_0_energy", "vehicle_1_energy"],
            "ordered_vehicle_locations": True, "terminal_continuation": 0.,
            "backup": "mean_draw(max_action(original_reward + gamma * bilinear_next_value))",
            "query_seed_rule": "training_seed_base + 2 * (seed * horizon * samples_per_period + period * samples_per_period + draw)",
            "period_draw_seeds": seeds,
            "common_random_numbers": "Each period/draw seed is reused at every fleet grid state; observed choices still depend on that state",
            "generative_access": "Arbitrary feasible fleet states are set before the simulator's original period preparation and fixed-price customer-choice logic",
            "states_per_period": states_per_period,
            "expected_state_draw_queries": scenario.horizon * states_per_period * config.samples_per_period,
            "unique_exogenous_draw_seeds": scenario.horizon * config.samples_per_period,
            "evaluation_data_used": False, "source_sha256": source_hashes,
        }
        _write_json(directory / "scenario.json", asdict(scenario))
        _write_json(directory / "training_config.json", asdict(config))
        _write_json(directory / "protocol.json", protocol)
        protocol_hash = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
        metrics_path = directory / "training_metrics.csv"
        metric_fields = ("period", "grid_states", "state_draw_queries", "candidate_outcomes", "mean_value", "min_value", "max_value", "wall_seconds")
        total_queries, total_candidates = 0, 0
        started = time.perf_counter()
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=metric_fields)
            writer.writeheader()
            for t in range(scenario.horizon - 1, -1, -1):
                layer_start = time.perf_counter()
                layer_queries, layer_candidates = 0, 0
                for l0 in range(scenario.n_zones):
                    for l1 in range(scenario.n_zones):
                        for e0, energy0 in enumerate(grid):
                            for e1, energy1 in enumerate(grid):
                                scores = []
                                for seed in seeds[t]:
                                    obs = sample_dispatch_observation(scenario, t, (l0, l1),
                                        (energy0, energy1), seed, config.price_index)
                                    _, _, _, outcomes = joint_dispatch_outcomes(
                                        obs, scenario, config.energy_fractions, config.max_joint_actions)
                                    continuation = policy.continuation(t + 1, outcomes["next_locations"],
                                                                      outcomes["next_energy_kwh"])
                                    candidate_scores = outcomes["reward"] + scenario.discount_per_period * continuation
                                    if not np.isfinite(candidate_scores).all():
                                        raise FloatingPointError("Nonfinite fitted-value backup")
                                    scores.append(float(np.max(candidate_scores)))
                                    layer_queries += 1
                                    layer_candidates += len(candidate_scores)
                                policy.values[t, l0, l1, e0, e1] = float(np.mean(scores))
                layer = policy.values[t]
                row = {"period": t, "grid_states": states_per_period,
                       "state_draw_queries": layer_queries, "candidate_outcomes": layer_candidates,
                       "mean_value": float(layer.mean()), "min_value": float(layer.min()),
                       "max_value": float(layer.max()), "wall_seconds": time.perf_counter() - layer_start}
                writer.writerow(row)
                stream.flush()
                total_queries += layer_queries
                total_candidates += layer_candidates
                print(f"Fitted layer {t:2d}: {layer_queries} state/draw queries, "
                      f"{layer_candidates} candidate outcomes, mean value {row['mean_value']:.4f} "
                      f"({row['wall_seconds']:.2f}s)", flush=True)
        if total_queries != protocol["expected_state_draw_queries"]:
            raise RuntimeError("Fitted-value query counters do not match the frozen protocol")
        if self._source_hashes() != source_hashes:
            raise RuntimeError("Training source changed during fitting; results were not checkpointed")
        policy.training_provenance = {"protocol_sha256": protocol_hash, "source_sha256": source_hashes,
                                      "state_draw_queries": total_queries, "candidate_outcomes": total_candidates}
        checkpoint = directory / "checkpoint.npz"
        policy.save_checkpoint(checkpoint)
        loaded = FittedValuePolicy.load_checkpoint(checkpoint)
        replay_equal = np.array_equal(loaded.values, policy.values) and np.array_equal(loaded.energy_grid, grid)
        if not replay_equal:
            raise RuntimeError("Fitted-value checkpoint reload changed the value table")
        summary = {
            "algorithm": protocol["algorithm"], "seed": config.seed,
            "output_dir": str(directory.resolve()), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "protocol_sha256": protocol_hash, "value_shape": list(shape), "value_entries": int(policy.values.size),
            "energy_grid_kwh": grid.tolist(), "fixed_mobile_price": float(scenario.price_levels[config.price_index]),
            "completed_layers": scenario.horizon, "states_per_period": states_per_period,
            "state_draw_queries": total_queries, "candidate_outcomes": total_candidates,
            "unique_exogenous_draw_seeds": scenario.horizon * config.samples_per_period,
            "all_values_finite": bool(np.isfinite(policy.values).all()), "exact_checkpoint_reload": replay_equal,
            "training_wall_seconds": time.perf_counter() - started,
            "evaluation_data_used": False,
            "information_limit": "Model-based generative-state training; not equal-information reward-only PPO",
        }
        _write_json(directory / "summary.json", summary)
        self.policy = policy
        return summary
