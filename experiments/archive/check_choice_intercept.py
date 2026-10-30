"""Measure a common charging utility intercept on identical reference states.

Generate a fixed state bank with the previously selected inventory rule under
the unchanged b=0 scenario. Compare b=0,2,3,4 without changing those observations,
prices, accessibility, waiting estimates or fleet states. This is a descriptive
choice-probability check, not a training run or a profit benchmark.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from charging_env.choice import charging_choice_probabilities
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator
from charging_env.benchmarks import policies as rules

INTERCEPTS = (0., 2., 3., 4.)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_value(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False, default=json_default).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=json_default)
        stream.write('\n')


def perceived_costs(observation, prices, scenario):
    """Reconstruct costs from public pre-choice fields, not simulator internals.

    Probability computation itself is delegated to the shared production
    helper, so this diagnostic cannot silently retain the old b=0 choice law.
    """
    c = scenario
    if int(observation['stage']) != 0:
        raise ValueError('The sensitivity checker requires pre-choice states')
    details = np.asarray(observation['request_details'], dtype=np.float64)
    origins = details[:, 0].astype(np.int64)
    energy = details[:, 1]
    mobile_cost = (prices[origins] * energy
                   + c.mobile_wait_cost_per_hour * observation['mobile_wait_hours'][origins])
    fixed_cost = (energy[:, None] * observation['fcs_retail_prices']
                  + c.fixed_travel_cost_per_km * c.distances[origins][:, c.fcs_zones]
                  + c.fixed_wait_cost_per_hour * observation['fcs_wait_hours'])
    available_mobile = np.asarray(observation['mobile_available'], dtype=bool)[origins]
    accessible_fixed = details[:, 3:].astype(bool)
    return mobile_cost, fixed_cost, available_mobile, accessible_fixed


def build_reference_bank(scenario, seeds, specification):
    """Generate ordinary complete b=0 episodes; do not evaluate other b values."""
    bank = []
    baseline_helper_error = 0.
    invalid_actions = 0
    for seed in seeds:
        sim = ChargingSimulator(scenario)
        observation, _ = sim.reset(seed=seed)
        while sim.stage != 2:
            levels = rules.fixed_prices(observation, scenario, specification['price_index'])
            prices = np.asarray(observation['mobile_price_menu'], dtype=np.float64)[
                np.arange(scenario.n_zones), levels]
            mobile_cost, fixed_cost, mobile_available, accessible_fcs = perceived_costs(
                observation, prices, scenario)
            actual = sim.choice_probabilities(prices)
            reconstructed = charging_choice_probabilities(
                scenario, mobile_cost, fixed_cost, mobile_available, accessible_fcs)
            if actual.size:
                baseline_helper_error = max(baseline_helper_error,
                                            float(np.max(np.abs(actual - reconstructed))))
            bank.append({'environment_seed': int(seed), 'period': int(observation['period'][0]),
                         'observation': observation, 'quoted_price_levels': levels,
                         'quoted_prices': prices, 'mobile_cost': mobile_cost,
                         'fixed_cost': fixed_cost, 'mobile_available_by_request': mobile_available,
                         'accessible_fcs_by_request': accessible_fcs})
            post, _, _, _, _ = sim.quote_price_levels(levels)
            actions = rules.coordinated_dispatch(post, scenario,
                                                 shadow_price=specification['shadow_price'],
                                                 energy_fractions=(.5, 1.))
            observation, _, _, _, info = sim.dispatch(actions)
            invalid_actions += len(info['invalid_actions'])
    if invalid_actions or baseline_helper_error > 1e-12:
        raise AssertionError('Reference bank does not match the public-state choice contract')
    return bank, {'invalid_reference_actions': invalid_actions,
                  'maximum_baseline_helper_vs_simulator_error': baseline_helper_error}


def sensitivity(scenario, bank):
    """Aggregate expected shares by request count, not by episode averages."""
    mobile_cost = np.concatenate([row['mobile_cost'] for row in bank])
    fixed_cost = np.concatenate([row['fixed_cost'] for row in bank])
    mobile_available = np.concatenate([row['mobile_available_by_request'] for row in bank])
    accessible_fcs = np.concatenate([row['accessible_fcs_by_request'] for row in bank])
    count = len(mobile_cost)
    if count == 0:
        raise ValueError('The reference bank contains no requests')
    available = np.column_stack((mobile_available, accessible_fcs))
    baseline = charging_choice_probabilities(scenario, mobile_cost, fixed_cost,
                                            mobile_available, accessible_fcs)
    baseline_mass = baseline[:, :-1].sum(axis=1)
    previous = None
    rows = []
    for intercept in INTERCEPTS:
        c = replace(scenario, charging_utility_intercept=intercept).validate()
        probabilities = charging_choice_probabilities(c, mobile_cost, fixed_cost,
                                                       mobile_available, accessible_fcs)
        if probabilities.shape != (count, c.n_fcs + 2):
            raise AssertionError('Unexpected probability shape')
        finite = bool(np.isfinite(probabilities).all())
        nonnegative = bool((probabilities >= 0.).all())
        normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.)))
        unavailable_violations = int(np.count_nonzero(probabilities[:, :-1][~available]))
        charging_mass = probabilities[:, :-1].sum(axis=1)
        positive = (baseline_mass > 0.) & (charging_mass > 0.)
        conditional_error = (float(np.max(np.abs(
            probabilities[positive, :-1] / charging_mass[positive, None]
            - baseline[positive, :-1] / baseline_mass[positive, None])))
                             if positive.any() else None)
        outside_increase = (max(0., float(np.max(probabilities[:, -1] - previous[:, -1])))
                            if previous is not None else 0.)
        row = {'charging_utility_intercept': intercept, 'number_of_requests': count,
               'request_weighted_expected_outside_share': float(probabilities[:, -1].mean()),
               'request_weighted_expected_charging_participation_share': float(charging_mass.mean()),
               'expected_outside_count': float(probabilities[:, -1].sum()),
               'expected_charging_count': float(charging_mass.sum()),
               'expected_provider_counts': probabilities[:, :-1].sum(axis=0).tolist(),
               'conditional_provider_comparable_requests': int(positive.sum()),
               'conditional_provider_excluded_requests': int((~positive).sum()),
               'maximum_conditional_provider_share_error_vs_b0': conditional_error,
               'unavailable_provider_probability_violations': unavailable_violations,
               'all_probabilities_finite': finite, 'all_probabilities_nonnegative': nonnegative,
               'maximum_probability_sum_error': normalization_error,
               'maximum_per_request_outside_increase_from_previous_b': outside_increase}
        if (not finite or not nonnegative or unavailable_violations
                or normalization_error > 1e-12 or outside_increase > 1e-12
                or (conditional_error is not None and conditional_error > 1e-12)):
            raise AssertionError(f'Choice-intercept invariance check failed: {row}')
        rows.append(row)
        previous = probabilities
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-path', type=Path,
                        default=ROOT / 'outputs/policy_benchmarks/selection.json',
                        help='Existing original b=0 benchmark selection; never retuned here')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/choice_intercept_check')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seed-base', type=int, default=2900001)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 64:
        parser.error('--episodes must be in [1,64]')
    if args.seed_base < 0 or args.seed_base % 2 != 1:
        parser.error('--seed-base must be a nonnegative odd integer')
    if not args.selection_path.is_file():
        parser.error('Provide --selection-path pointing to the saved original benchmark selection')
    if args.output_dir.exists():
        parser.error('Use a new output directory')
    selection = json.loads(args.selection_path.read_text(encoding='utf-8'))
    try:
        specification = dict(selection['family_winners']['fixed_inventory'])
    except (KeyError, TypeError):
        parser.error('Selection must contain the previously selected fixed_inventory rule')
    if specification.get('family') != 'fixed_inventory':
        parser.error('Selected family must be fixed_inventory')
    scenario = Scenario().validate()
    if scenario.charging_utility_intercept != 0.:
        parser.error('This reference-bank protocol requires the unchanged default b=0 scenario')
    index = specification.get('price_index')
    # The frozen rule quotes the medium 1.4x premium under premium pricing, or
    # the historical 0.50 per kWh under the legacy absolute menu.
    expected_price = 1.4 if scenario.mcv_pricing_mode == 'premium' else .50
    if (isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < scenario.n_price_levels
            or scenario.price_setting(index) != expected_price
            or 'shadow_price' not in specification):
        parser.error(f'The original frozen inventory rule must quote {expected_price}')
    seeds = [args.seed_base + 2*i for i in range(args.episodes)]
    sources = ['src/charging_env/config.py', 'src/charging_env/choice.py',
               'src/charging_env/simulator.py', 'src/charging_env/benchmarks/policies.py',
               'experiments/archive/check_choice_intercept.py']
    source_hashes = {name: digest_file(ROOT / name) for name in sources}
    selection_hash = digest_file(args.selection_path)
    protocol = {
        'purpose': 'Fixed-state descriptive choice sensitivity; no training, no profit test, no retuning',
        'scenario': asdict(scenario), 'scenario_sha256': digest_value(asdict(scenario)),
        'selection_path': str(args.selection_path.resolve()), 'selection_sha256': selection_hash,
        'reference_policy': specification, 'reference_energy_fractions': [.5, 1.],
        'reference_charging_utility_intercept': 0., 'compared_intercepts': list(INTERCEPTS),
        'environment_seeds': seeds, 'episodes': args.episodes,
        'action_seed': None, 'action_randomness': 'None: frozen deterministic inventory rule',
        'source_sha256': source_hashes,
        'weighting': 'Each potential request has equal weight across all reference periods and episodes',
        'fixed_inputs': 'Origins, energies, accessibility, availability, quoted prices, waiting estimates and physical states are identical across b values',
        'conditional_provider_invariance': 'Compare probabilities conditional on any charging provider, only where both b=0 and candidate charging mass are positive',
        'interpretation': 'Expected participation under frozen b=0 state visitation. This is neither a new-b rollout nor evidence about realized service, capacity adequacy, profit or an optimal/calibrated intercept.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'protocol.json', protocol)
    numpy_rng_before = np.random.get_state()
    bank, reference_checks = build_reference_bank(scenario, seeds, specification)
    rows = sensitivity(scenario, bank)
    numpy_rng_after = np.random.get_state()
    numpy_preserved = (numpy_rng_before[0] == numpy_rng_after[0]
                       and np.array_equal(numpy_rng_before[1], numpy_rng_after[1])
                       and numpy_rng_before[2:] == numpy_rng_after[2:])
    if not numpy_preserved:
        raise AssertionError('The sensitivity checker changed the global NumPy RNG')
    if digest_file(args.selection_path) != selection_hash:
        raise RuntimeError('Frozen rule selection changed during the check')
    if any(digest_file(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError('Source files changed during the check; rerun after freezing sources')
    write_json(args.output_dir / 'reference_states.json', bank)
    summary = {'protocol_sha256': digest_value(protocol), 'reference_states_sha256': digest_value(bank),
               'physical_reference_states': len(bank), 'number_of_requests': rows[0]['number_of_requests'],
               'numpy_global_rng_preserved': numpy_preserved, 'reference_checks': reference_checks,
               'sensitivity': rows, 'limitations': protocol['interpretation']}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
