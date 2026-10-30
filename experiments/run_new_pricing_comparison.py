"""Method comparison under the LADWP base-price / premium-menu scenario.

The frozen fitted-value and Joint PPO studies cannot be rerun here: their
artifacts were produced on another machine, and their protocol guards pin the
old absolute price menu. This runner rebuilds an equivalent comparison end to
end on the current pricing model, reusing the project's own candidate grid,
trainers, and paired evaluation harness.

Phases, each writing JSON into --study-dir so they can run independently:
  select     best compact rule per family, chosen on the validation bank only
  train-adp  one table-ADP value function per training seed
  train-ppo  one Joint PPO controller per training seed
  evaluate   every method on the base bank and two stress banks
  summarize  print the comparison table

Selection never touches the test banks. Learned methods are trained on the base
scenario only and then evaluated under shift, so the stress columns measure
generalization rather than retuning.
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
from charging_env.benchmarks import compact, policies
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.rl.fitted_value import FittedValueConfig, FittedValuePolicy, FittedValueTrainer
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer

TRAINING_SEEDS = (7, 19, 37)
VALIDATION_SEEDS = [181000001 + 2 * i for i in range(128)]
TEST_SEEDS = [183000001 + 2 * i for i in range(512)]
HIGH_DEMAND_SEEDS = [185000001 + 2 * i for i in range(256)]
LOW_INVENTORY_SEEDS = [187000001 + 2 * i for i in range(256)]
ACTION_SEED = 189991
BASELINE = 'service_first'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def test_banks(scenario):
    """The base scenario and the two predeclared stress shifts, with their seeds."""
    high = replace(scenario, mean_requests_by_zone=tuple(
        1.5 * x for x in scenario.mean_requests_by_zone)).validate()
    low = replace(scenario, initial_energy_kwh=tuple(
        35. for _ in scenario.initial_energy_kwh)).validate()
    return {'base': (scenario, TEST_SEEDS),
            'high_demand': (high, HIGH_DEMAND_SEEDS),
            'low_inventory': (low, LOW_INVENTORY_SEEDS)}


def rule_callbacks(scenario, spec):
    """Pricing and dispatch callbacks for one compact-grid setting."""
    pricing = lambda obs, rng: policies.fixed_prices(obs, scenario, spec['price_index'])
    if spec['family'] == 'service_first':
        dispatch = lambda obs, rng: policies.service_first_dispatch(
            obs, scenario, spec['recharge_threshold'])
    else:
        dispatch = lambda obs, rng: policies.coordinated_dispatch(
            obs, scenario, compact.inventory_shadow(obs, scenario, spec),
            energy_fractions=compact.ENERGY_FRACTIONS)
    return pricing, dispatch


def score(scenario, seeds, pricing, dispatch, label, on_episode_start=None):
    result = evaluate_policy(scenario, seeds, pricing, dispatch, action_seed=ACTION_SEED,
                             label=label, on_episode_start=on_episode_start)
    if result['invalid_actions']:
        raise RuntimeError(f'{label} emitted {result["invalid_actions"]} invalid actions')
    return result


def phase_select(scenario, study):
    """Choose the best setting in each compact family on the validation bank."""
    specs = compact.candidates(scenario)
    print(f'Scoring {len(specs)} compact settings on {len(VALIDATION_SEEDS)} validation episodes')
    rows = []
    started = time.monotonic()
    for index, spec in enumerate(specs, start=1):
        pricing, dispatch = rule_callbacks(scenario, spec)
        result = score(scenario, VALIDATION_SEEDS, pricing, dispatch, spec['id'])
        rows.append(dict(spec, mean_discounted_profit=result['mean_discounted_profit']))
        if index % 10 == 0 or index == len(specs):
            print(f'  {index}/{len(specs)} settings ({time.monotonic()-started:.0f}s)')
    winners = {}
    for family in compact.FAMILIES:
        family_rows = [row for row in rows if row['family'] == family]
        winners[family] = max(family_rows, key=lambda row: row['mean_discounted_profit'])
        print(f'  best {family:<20} {winners[family]["id"]:<24} '
              f'{winners[family]["mean_discounted_profit"]:+8.3f}')
    write_json(study / 'selection.json', {
        'validation_seeds': VALIDATION_SEEDS, 'action_seed': ACTION_SEED,
        'candidates': rows, 'family_winners': winners,
        'note': 'Selected on validation episodes only; test banks were not consulted.'})
    return winners


def phase_train_adp(scenario, study, seeds=TRAINING_SEEDS):
    for seed in seeds:
        directory = study / 'adp' / f'seed_{seed}'
        if (directory / 'checkpoint.npz').exists():
            print(f'Table ADP seed {seed}: checkpoint present, skipping')
            continue
        # Fix the mobile price at the middle setting of the configured menu.
        config = FittedValueConfig(seed=seed, grid_step=10., samples_per_period=8,
                                   price_index=scenario.n_price_levels // 2)
        print(f'Fitting table ADP, training seed {seed}')
        started = time.monotonic()
        FittedValueTrainer(scenario, config).train(directory)
        print(f'  done in {time.monotonic()-started:.0f}s')


def phase_train_ppo(scenario, study, updates, seeds=TRAINING_SEEDS):
    for seed in seeds:
        directory = study / 'ppo' / f'seed_{seed}'
        if (directory / 'best_checkpoint.pt').exists():
            print(f'Joint PPO seed {seed}: checkpoint present, skipping')
            continue
        config = PaperPPOConfig(seed=seed, updates=updates)
        print(f'Training Joint PPO, training seed {seed}, {updates} updates')
        started = time.monotonic()
        PaperPPOTrainer(scenario, config).train(directory)
        print(f'  done in {(time.monotonic()-started)/60:.1f} min')


def evaluate_rules(scenario, study, winners, results):
    for family, spec in winners.items():
        settings = {k: v for k, v in spec.items() if k != 'mean_discounted_profit'}
        for case, (case_scenario, seeds) in test_banks(scenario).items():
            pricing, dispatch = rule_callbacks(case_scenario, settings)
            result = score(case_scenario, seeds, pricing, dispatch, f'{family}/{case}')
            results.setdefault(family, {}).setdefault(case, []).append(result)
            print(f'  {family:<20} {case:<14} {result["mean_discounted_profit"]:+8.3f}')


def evaluate_adp(scenario, study, results):
    for seed in TRAINING_SEEDS:
        path = study / 'adp' / f'seed_{seed}' / 'checkpoint.npz'
        if not path.exists():
            print(f'  table_adp seed {seed}: no checkpoint, skipped')
            continue
        policy = FittedValuePolicy.load_checkpoint(path)
        for case, (case_scenario, seeds) in test_banks(scenario).items():
            # The value table was fitted on the base scenario; shifted banks
            # reuse it deliberately, so the column measures generalization.
            active = (policy if case == 'base'
                      else policy.with_scenario(case_scenario, allow_distribution_shift=True))
            result = score(case_scenario, seeds, active.prices, active.dispatch,
                           f'table_adp/{case}/seed_{seed}')
            results.setdefault('table_adp', {}).setdefault(case, []).append(result)
            print(f'  table_adp seed {seed:<3} {case:<14} {result["mean_discounted_profit"]:+8.3f}')


def evaluate_ppo(scenario, study, results):
    for seed in TRAINING_SEEDS:
        path = study / 'ppo' / f'seed_{seed}' / 'best_checkpoint.pt'
        if not path.exists():
            print(f'  joint_ppo seed {seed}: no checkpoint, skipped')
            continue
        trainer = PaperPPOTrainer.load_checkpoint(path)
        trainer.set_evaluation_mode(False)
        for case, (case_scenario, seeds) in test_banks(scenario).items():
            result = score(case_scenario, seeds, trainer.prices, trainer.dispatch,
                           f'joint_ppo/{case}/seed_{seed}',
                           on_episode_start=trainer.on_episode_start)
            results.setdefault('joint_ppo', {}).setdefault(case, []).append(result)
            print(f'  joint_ppo seed {seed:<3} {case:<14} {result["mean_discounted_profit"]:+8.3f}')


def phase_evaluate(scenario, study):
    winners = read_json(study / 'selection.json')['family_winners']
    results = {}
    print('Evaluating selected rules')
    evaluate_rules(scenario, study, winners, results)
    print('Evaluating table ADP')
    evaluate_adp(scenario, study, results)
    print('Evaluating Joint PPO')
    evaluate_ppo(scenario, study, results)

    comparisons = {}
    for case in test_banks(scenario):
        baseline = results.get(BASELINE, {}).get(case)
        if not baseline:
            continue
        for method, cases in results.items():
            if method == BASELINE or case not in cases:
                continue
            # Compare the first training seed's run against the shared baseline;
            # across-seed spread is reported separately in the summary.
            comparisons.setdefault(case, {})[method] = paired_comparison(
                cases[case][0], baseline[0])
    write_json(study / 'evaluation.json', {
        'scenario': asdict(scenario), 'action_seed': ACTION_SEED,
        'training_seeds': list(TRAINING_SEEDS), 'results': results,
        'paired_vs_service_first': comparisons})
    return results


def phase_summarize(scenario, study):
    payload = read_json(study / 'evaluation.json')
    results, comparisons = payload['results'], payload['paired_vs_service_first']
    cases = list(test_banks(scenario))
    order = [BASELINE, 'constant_inventory', 'time_inventory', 'table_adp', 'joint_ppo']
    present = [m for m in order if m in results] + [m for m in results if m not in order]

    print(f'\nMean discounted profit per episode, dollars '
          f'(base n={len(TEST_SEEDS)}, stress n={len(HIGH_DEMAND_SEEDS)})\n')
    header = f'{"method":<20}' + ''.join(f'{case:>22}' for case in cases) + f'{"seeds":>7}'
    print(header)
    print('-' * len(header))
    for method in present:
        line = f'{method:<20}'
        runs = 0
        for case in cases:
            entries = results[method].get(case, [])
            runs = max(runs, len(entries))
            if not entries:
                line += f'{"-":>22}'
                continue
            means = np.array([e['mean_discounted_profit'] for e in entries])
            if len(means) > 1:
                line += f'{means.mean():>+12.2f} ±{means.std(ddof=1):>7.2f}'
            else:
                line += f'{means[0]:>+12.2f} ±{entries[0]["standard_error_discounted_profit"]:>7.2f}'
        print(line + f'{runs:>7}')
    print('\nFor learned methods the spread is across training seeds; for the '
          'single-policy rules it is the within-bank standard error.')

    for case in cases:
        if case not in comparisons:
            continue
        print(f'\nPaired difference vs {BASELINE}, {case} bank '
              f'(first training seed, 95% bootstrap interval)')
        for method, row in comparisons[case].items():
            low, high = row['paired_bootstrap_confidence_interval_95']
            flag = '' if low <= 0. <= high else '  significant'
            print(f'  {method:<20} {row["mean_difference"]:+8.2f}  '
                  f'[{low:+7.2f}, {high:+7.2f}]  win {row["candidate_win_fraction"]:.0%}{flag}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True,
                        choices=('select', 'train-adp', 'train-ppo', 'evaluate', 'summarize'))
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/new_pricing_comparison')
    parser.add_argument('--scenario', type=Path, default=ROOT / 'configs/utility_b3.json')
    parser.add_argument('--updates', type=int, default=PaperPPOConfig().updates,
                        help='Joint PPO updates per training seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TRAINING_SEEDS),
                        help='Training seeds to fit in this process, so seeds can run in parallel')
    args = parser.parse_args(argv)

    scenario = Scenario.from_json(args.scenario)
    if scenario.mcv_pricing_mode != 'premium' or scenario.fcs_pricing_mode != 'base_price':
        parser.error('This comparison requires the base-price station and premium mobile menus')
    study = args.study_dir
    study.mkdir(parents=True, exist_ok=True)

    if args.phase == 'select':
        phase_select(scenario, study)
    elif args.phase == 'train-adp':
        phase_train_adp(scenario, study, tuple(args.seeds))
    elif args.phase == 'train-ppo':
        phase_train_ppo(scenario, study, args.updates, tuple(args.seeds))
    elif args.phase == 'evaluate':
        phase_evaluate(scenario, study)
        phase_summarize(scenario, study)
    else:
        phase_summarize(scenario, study)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
