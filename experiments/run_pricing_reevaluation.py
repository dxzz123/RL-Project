"""Re-evaluate every controller on this branch under the new station pricing.

The frozen studies 08-12 cannot be rerun here: their artifacts live on another
machine and several guards read recorded protocols from absolute paths. This
runner rebuilds an equivalent comparison for both supplied cities using the
project's own selection, trainers and paired evaluation harness, on the current
LADWP base-price / premium-menu scenario.

Test conditions follow study 12, so the structure is comparable even though the
numbers are not: the original pattern, demand +40%, and a shift that rotates
demand locations by half the grid and delays the hourly pattern by three hours.

Methods are per city. Table ADP supports exactly two vehicles, so it runs in
the small city only. The large city dispatches inventory rules with the bounded
coordinate solver rather than joint enumeration.

Phases, each writing JSON into --study-dir so they can run independently:
  select     best rule per family, on validation days only
  train-adp  one table-ADP value function per seed (small city only)
  train-ppo  one Joint PPO controller per seed
  evaluate   score every available method on all three conditions
  summarize  print the comparison table
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from charging_env.config import Scenario
from charging_env.benchmarks.compact import candidates, inventory_shadow, ENERGY_FRACTIONS
from charging_env.benchmarks.policies import (coordinated_dispatch, fixed_prices,
                                              service_first_dispatch)
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.rl.fitted_value import FittedValueConfig, FittedValuePolicy, FittedValueTrainer
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy

CASES = {'small': 'configs/utility_b3.json', 'city': 'configs/synthetic_25_zones.json'}
FAMILIES = {'small': ('service_first', 'constant_inventory', 'time_inventory'),
            'city': ('service_first', 'time_inventory')}
LEARNED = {'small': ('table_adp', 'rollout_adp', 'joint_ppo'),
           'city': ('rollout_adp', 'joint_ppo')}
SEEDS = (7, 19, 37)
# Study 12 used 128 test days per condition in the large city; the small city is
# cheap enough for more. Rollout ADP costs about 0.7 s/episode small, 5 s large.
VALIDATION = {'small': 128, 'city': 64}
TESTING = {'small': 256, 'city': 128}
BANK_BASE = {'small': (181_000_001, 183_000_001), 'city': (191_000_001, 193_000_001)}
ACTION_SEED = 189991
BASELINE = 'service_first'
CONDITIONS = ('original', 'demand_plus_40', 'shifted_places_times')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seeds_for(case, kind):
    base = BANK_BASE[case][0 if kind == 'validation' else 1]
    count = (VALIDATION if kind == 'validation' else TESTING)[case]
    return [base + 2 * i for i in range(count)]


def conditions(scenario):
    """The original pattern and study 12's two shifts, all on the same horizon.

    Both shifts are applied to the resolved [horizon, zone] rate matrix and set
    as an explicit arrival_rates table. Editing mean_requests_by_zone instead
    would silently do nothing for a scenario that already pins arrival_rates,
    which configs/synthetic_25_zones.json does.
    """
    rates = scenario.resolved_arrival_rates
    zones = rates.shape[1]
    shifted = np.roll(np.roll(rates, zones // 2, axis=1), 3, axis=0)
    as_table = lambda matrix: tuple(tuple(float(x) for x in row) for row in matrix)
    return {
        'original': scenario,
        'demand_plus_40': replace(scenario, arrival_rates=as_table(1.4 * rates)).validate(),
        'shifted_places_times': replace(scenario, arrival_rates=as_table(shifted)).validate(),
    }


def rule_callbacks(scenario, case, spec):
    """Pricing and dispatch for one rule setting, with the city's own solver."""
    pricing = lambda obs, rng: fixed_prices(obs, scenario, spec['price_index'])
    if spec['family'] == 'service_first':
        dispatch = lambda obs, rng: service_first_dispatch(obs, scenario, spec['recharge_threshold'])
    elif case == 'city':
        # Ten vehicles make joint enumeration infeasible; use the bounded search.
        dispatch = lambda obs, rng: coordinate_inventory_dispatch(
            obs, scenario, inventory_shadow(obs, scenario, spec),
            sweeps=2, energy_fractions=(1.,))
    else:
        dispatch = lambda obs, rng: coordinated_dispatch(
            obs, scenario, inventory_shadow(obs, scenario, spec),
            energy_fractions=ENERGY_FRACTIONS)
    return pricing, dispatch


def score(scenario, seeds, pricing, dispatch, label, on_episode_start=None):
    result = evaluate_policy(scenario, seeds, pricing, dispatch, action_seed=ACTION_SEED,
                             label=label, on_episode_start=on_episode_start)
    if result['invalid_actions']:
        raise RuntimeError(f'{label} emitted {result["invalid_actions"]} invalid actions')
    # 'episodes' is kept for the paired bootstrap and stripped before writing.
    return result


SUMMARY_KEYS = ('label', 'mean_profit', 'mean_discounted_profit',
                'standard_error_discounted_profit', 'profit_p10')


def compact(row):
    """Summary plus the paired-bootstrap inputs, which stay small."""
    out = {key: row[key] for key in SUMMARY_KEYS if key in row}
    out['episodes'] = [{'seed': int(e['seed']),
                        'discounted_profit': float(e['discounted_profit'])}
                       for e in row.get('episodes', ())]
    return out


def phase_select(case, scenario, study):
    specs = [s for s in candidates(scenario) if s['family'] in FAMILIES[case]]
    seeds = seeds_for(case, 'validation')
    print(f'[{case}] scoring {len(specs)} rule settings on {len(seeds)} validation days')
    rows, started = [], time.monotonic()
    for index, spec in enumerate(specs, start=1):
        pricing, dispatch = rule_callbacks(scenario, case, spec)
        row = score(scenario, seeds, pricing, dispatch, spec['id'])
        rows.append(dict(spec, mean_discounted_profit=row['mean_discounted_profit']))
        if index % 10 == 0 or index == len(specs):
            print(f'  {index}/{len(specs)} ({time.monotonic()-started:.0f}s)')
    winners = {family: max((r for r in rows if r['family'] == family),
                           key=lambda r: r['mean_discounted_profit'])
               for family in FAMILIES[case]}
    for family, win in winners.items():
        print(f'  best {family:<20} {win["id"]:<26} {win["mean_discounted_profit"]:+9.2f}')
    write_json(study / case / 'selection.json',
               {'validation_seeds': seeds, 'action_seed': ACTION_SEED,
                'candidates': rows, 'family_winners': winners,
                'note': 'Validation days only; test banks were not consulted.'})
    return winners


def phase_select_adp(case, scenario, study):
    """Choose each ADP controller's fixed premium on validation days only.

    Without this the controllers keep RolloutADPConfig's default middle index,
    while every rule selects its own price, so the comparison measures the
    price rather than the planner.
    """
    seeds = seeds_for(case, 'validation')
    chosen = {}
    rows = []
    for index in range(scenario.n_price_levels):
        policy = RolloutADPPolicy(scenario, RolloutADPConfig(seed=SEEDS[0], price_index=index))
        row = score(scenario, seeds, policy.prices, policy.dispatch,
                    f'rollout_adp/p{index}', on_episode_start=policy.reset_episode)
        rows.append({'price_index': index, 'premium': scenario.price_setting(index),
                     'mean_discounted_profit': row['mean_discounted_profit']})
        print(f'  rollout_adp index {index} ({scenario.price_setting(index)}x): '
              f'{row["mean_discounted_profit"]:+9.2f}')
    chosen['rollout_adp'] = max(rows, key=lambda r: r['mean_discounted_profit'])['price_index']
    if 'table_adp' in LEARNED[case]:
        table_rows = []
        for index in range(scenario.n_price_levels):
            path = study / case / 'table_adp' / f'p{index}' / f'seed_{SEEDS[0]}' / 'checkpoint.npz'
            if not path.exists():
                print(f'  table_adp index {index}: not fitted, skipped')
                continue
            policy = FittedValuePolicy.load_checkpoint(path)
            row = score(scenario, seeds, policy.prices, policy.dispatch, f'table_adp/p{index}')
            table_rows.append({'price_index': index,
                               'mean_discounted_profit': row['mean_discounted_profit']})
            print(f'  table_adp index {index} ({scenario.price_setting(index)}x): '
                  f'{row["mean_discounted_profit"]:+9.2f}')
        if table_rows:
            chosen['table_adp'] = max(table_rows,
                                      key=lambda r: r['mean_discounted_profit'])['price_index']
    print(f'  selected: {chosen}')
    write_json(study / case / 'adp_price_selection.json',
               {'validation_seeds': seeds, 'rollout_adp_rows': rows, 'chosen': chosen,
                'note': 'Validation days only, mirroring the rule selection.'})
    return chosen


def adp_price_index(case, scenario, study, method):
    """The validation-selected premium, or the controller default if unselected."""
    path = study / case / 'adp_price_selection.json'
    if path.exists():
        chosen = read_json(path)['chosen']
        if method in chosen:
            return int(chosen[method])
    return scenario.n_price_levels // 2


def phase_train_adp(case, scenario, study, seeds=SEEDS, price_index=None):
    if 'table_adp' not in LEARNED[case]:
        print(f'[{case}] table ADP needs exactly two vehicles; skipping')
        return
    index = scenario.n_price_levels // 2 if price_index is None else int(price_index)
    for seed in seeds:
        directory = study / case / 'table_adp' / f'p{index}' / f'seed_{seed}'
        if (directory / 'checkpoint.npz').exists():
            print(f'[{case}] table ADP p{index} seed {seed}: present, skipping')
            continue
        config = FittedValueConfig(seed=seed, grid_step=10., samples_per_period=8,
                                   price_index=index)
        print(f'[{case}] fitting table ADP p{index} seed {seed}')
        started = time.monotonic()
        FittedValueTrainer(scenario, config).train(directory)
        print(f'  done in {time.monotonic()-started:.0f}s')


def phase_train_ppo(case, scenario, study, updates, seeds=SEEDS):
    for seed in seeds:
        directory = study / case / 'joint_ppo' / f'seed_{seed}'
        if (directory / 'summary.json').exists():
            print(f'[{case}] Joint PPO seed {seed}: complete, skipping')
            continue
        if directory.exists():
            raise FileExistsError(f'{directory} holds an unfinished run; remove it first')
        print(f'[{case}] training Joint PPO seed {seed}, {updates} updates')
        started = time.monotonic()
        PaperPPOTrainer(scenario, PaperPPOConfig(seed=seed, updates=updates)).train(directory)
        print(f'  done in {(time.monotonic()-started)/60:.1f} min')


def evaluate_case(case, scenario, study, only=None, train_seeds=SEEDS):
    winners = read_json(study / case / 'selection.json')['family_winners']
    banks, seeds = conditions(scenario), seeds_for(case, 'testing')
    results = {}

    def record(method, condition, row):
        results.setdefault(method, {}).setdefault(condition, []).append(row)
        print(f'  {method:<18} {condition:<22} {row["mean_discounted_profit"]:+9.2f}')

    for family, spec in winners.items():
        if only and family not in only:
            continue
        setting = {k: v for k, v in spec.items() if k != 'mean_discounted_profit'}
        for condition, variant in banks.items():
            pricing, dispatch = rule_callbacks(variant, case, setting)
            record(family, condition, score(variant, seeds, pricing, dispatch, f'{family}/{condition}'))

    if 'table_adp' in LEARNED[case] and (not only or 'table_adp' in only):
        for seed in train_seeds:
            chosen = adp_price_index(case, scenario, study, 'table_adp')
            path = (study / case / 'table_adp' / f'p{chosen}' / f'seed_{seed}' / 'checkpoint.npz')
            if not path.exists():
                print(f'  table_adp seed {seed}: no checkpoint, skipped')
                continue
            policy = FittedValuePolicy.load_checkpoint(path)
            for condition, variant in banks.items():
                active = (policy if condition == 'original'
                          else policy.with_scenario(variant, allow_distribution_shift=True))
                record('table_adp', condition,
                       score(variant, seeds, active.prices, active.dispatch,
                             f'table_adp/{condition}/{seed}'))

    if not only or 'rollout_adp' in only:
        chosen = adp_price_index(case, scenario, study, 'rollout_adp')
        for seed in train_seeds:
            for condition, variant in banks.items():
                # One planner per condition: the config is fixed, only the world moves.
                active = RolloutADPPolicy(variant,
                                          RolloutADPConfig(seed=seed, price_index=chosen))
                record('rollout_adp', condition,
                       score(variant, seeds, active.prices, active.dispatch,
                             f'rollout_adp/{condition}/{seed}',
                             on_episode_start=active.reset_episode))

    if not only or 'joint_ppo' in only:
        for seed in train_seeds:
            path = study / case / 'joint_ppo' / f'seed_{seed}' / 'best_checkpoint.pt'
            summary = study / case / 'joint_ppo' / f'seed_{seed}' / 'summary.json'
            if not summary.exists():
                print(f'  joint_ppo seed {seed}: training incomplete, skipped')
                continue
            trainer = PaperPPOTrainer.load_checkpoint(path)
            trainer.set_evaluation_mode(False)
            for condition, variant in banks.items():
                record('joint_ppo', condition,
                       score(variant, seeds, trainer.prices, trainer.dispatch,
                             f'joint_ppo/{condition}/{seed}',
                             on_episode_start=trainer.on_episode_start))
    return results, seeds


def phase_evaluate(case, scenario, study, only=None, seeds=SEEDS, tag=None):
    print(f'[{case}] evaluating on {TESTING[case]} test days per condition')
    results, test_seeds = evaluate_case(case, scenario, study, only, seeds)
    comparisons = {}
    for condition in CONDITIONS:
        baseline = results.get(BASELINE, {}).get(condition)
        if not baseline:
            continue
        for method, rows in results.items():
            if method == BASELINE or condition not in rows:
                continue
            comparisons.setdefault(condition, {})[method] = paired_comparison(
                dict(rows[condition][0], label=method), dict(baseline[0], label=BASELINE))
    stripped = {method: {condition: [compact(r) for r in rows]
                         for condition, rows in by_condition.items()}
                for method, by_condition in results.items()}
    # Each arm writes its own file, so parallel arms never race. summarize()
    # merges them. An arm must include the baseline for its paired interval.
    path = study / case / (f'evaluation_{tag}.json' if tag else 'evaluation.json')
    merged = stripped
    write_json(path, {'case': case, 'scenario': asdict(scenario), 'test_seeds': test_seeds,
                      'action_seed': ACTION_SEED, 'training_seeds': list(seeds),
                      'results': merged, 'paired_vs_service_first': comparisons})
    return merged


def phase_summarize(case, scenario, study):
    results, comparisons = {}, {}
    files = sorted((study / case).glob('evaluation*.json'))
    if not files:
        raise FileNotFoundError(f'No evaluation files under {study / case}')
    for path in files:
        payload = read_json(path)
        for method, by_condition in payload['results'].items():
            for condition, rows in by_condition.items():
                results.setdefault(method, {}).setdefault(condition, []).extend(rows)
        for condition, rows in payload['paired_vs_service_first'].items():
            comparisons.setdefault(condition, {}).update(rows)
    # An arm may repeat a single-policy rule so it has its own baseline for the
    # paired interval. Those rows are identical, so keep one per distinct score.
    for method, by_condition in results.items():
        for condition, rows in by_condition.items():
            unique, seen = [], set()
            for row in rows:
                key = round(row['mean_discounted_profit'], 9)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            by_condition[condition] = unique
    print(f'[{case}] merged {len(files)} arm file(s): '
          + ', '.join(f.name for f in files))
    # Recompute every paired interval from the merged episodes, so arms that ran
    # without the baseline still get one. Stored intervals are the fallback.
    for condition in CONDITIONS:
        baseline = results.get(BASELINE, {}).get(condition)
        if not baseline or not baseline[0].get('episodes'):
            continue
        for method, rows in results.items():
            if method == BASELINE or condition not in rows or not rows[condition][0].get('episodes'):
                continue
            comparisons.setdefault(condition, {})[method] = paired_comparison(
                dict(rows[condition][0], label=method),
                dict(baseline[0], label=BASELINE))
    order = [BASELINE, 'constant_inventory', 'time_inventory', 'table_adp', 'rollout_adp', 'joint_ppo']
    present = [m for m in order if m in results] + [m for m in results if m not in order]
    print(f'\n[{case}] mean discounted profit per day, dollars '
          f'({TESTING[case]} test days per condition)\n')
    header = f'{"method":<20}' + ''.join(f'{c:>24}' for c in CONDITIONS) + f'{"seeds":>7}'
    print(header)
    print('-' * len(header))
    for method in present:
        line, runs = f'{method:<20}', 0
        for condition in CONDITIONS:
            rows = results[method].get(condition, [])
            runs = max(runs, len(rows))
            if not rows:
                line += f'{"-":>24}'
                continue
            means = np.array([r['mean_discounted_profit'] for r in rows])
            spread = (means.std(ddof=1) if len(means) > 1
                      else rows[0]['standard_error_discounted_profit'])
            line += f'{means.mean():>+14.2f} +-{spread:>7.2f}'
        print(line + f'{runs:>7}')
    print('\nSpread is across training seeds for multi-seed methods, and the '
          'within-bank standard error for the single-policy rules.')
    for condition in CONDITIONS:
        if condition not in comparisons:
            continue
        print(f'\n[{case}] paired difference vs {BASELINE}, {condition} '
              f'(first training seed, 95% bootstrap)')
        for method, row in comparisons[condition].items():
            low, high = row['paired_bootstrap_confidence_interval_95']
            flag = '' if low <= 0. <= high else '  significant'
            print(f'  {method:<18} {row["mean_difference"]:+9.2f}  '
                  f'[{low:+8.2f}, {high:+8.2f}]  win {row["candidate_win_fraction"]:.0%}{flag}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True,
                        choices=('select', 'select-adp', 'train-adp', 'train-ppo',
                                 'evaluate', 'summarize'))
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/pricing_reevaluation')
    parser.add_argument('--updates', type=int, default=PaperPPOConfig().updates)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--only', nargs='+', help='Evaluate only these methods')
    parser.add_argument('--tag', help='Write this arm to evaluation_<tag>.json')
    parser.add_argument('--price-index', type=int, help='Fit table ADP at this premium')
    args = parser.parse_args(argv)

    scenario = Scenario.from_json(ROOT / CASES[args.case])
    if scenario.mcv_pricing_mode != 'premium' or scenario.fcs_pricing_mode != 'base_price':
        parser.error('This re-evaluation requires the base-price and premium pricing modes')
    study = args.study_dir
    (study / args.case).mkdir(parents=True, exist_ok=True)

    if args.phase == 'select':
        phase_select(args.case, scenario, study)
    elif args.phase == 'select-adp':
        phase_select_adp(args.case, scenario, study)
    elif args.phase == 'train-adp':
        phase_train_adp(args.case, scenario, study, tuple(args.seeds), args.price_index)
    elif args.phase == 'train-ppo':
        phase_train_ppo(args.case, scenario, study, args.updates, tuple(args.seeds))
    elif args.phase == 'evaluate':
        phase_evaluate(args.case, scenario, study, args.only, tuple(args.seeds), args.tag)
    else:
        phase_summarize(args.case, scenario, study)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
