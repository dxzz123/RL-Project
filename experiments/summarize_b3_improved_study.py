"""Summarize a completed frozen b=3 study without running any environment.

Only saved artifacts are read. All three training seeds and both predeclared
success criteria are retained. The output is created exclusively and cannot
overwrite prior evidence. Invoke only after final benchmark evaluation finishes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 19, 37)
PRIMARY = 'rl_final_stochastic_mean'
PREVIOUS = 'previous_final_stochastic_mean'
COST_TERMS = {'procurement_cost', 'travel_cost', 'operating_cost', 'unmet_demand_cost'}
REVENUE_TERMS = {'mobile_service_revenue', 'fixed_service_revenue', 'discharge_revenue',
                 'terminal_inventory_value'}
BEHAVIOR_FIELDS = ('mean_charge_grid_kwh', 'mean_mobile_delivered_kwh',
    'mean_final_inventory_kwh', 'mean_mobile_price', 'mean_export_grid_kwh',
    'mean_fixed_delivered_kwh', 'mean_served_request_share', 'mean_outside_option_share',
    'pooled_selected_service_fulfillment', 'lower_tail_mean_10pct', 'loss_probability')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def text(self, path):
        path = path.resolve()
        self.hashes[str(path)] = digest(path)
        return path.read_text()

    def json(self, path):
        return json.loads(self.text(path))

    def csv(self, path):
        return list(csv.DictReader(self.text(path).splitlines()))

    def verify(self):
        changed = [path for path, value in self.hashes.items() if digest(Path(path)) != value]
        if changed:
            raise ValueError('Input artifacts changed while summarizing: ' + ', '.join(changed))


def close(actual, expected, message):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f'{message}: {actual} != {expected}')


def window_statistics(rows):
    values = [float(row['mean_discounted_profit']) for row in rows]
    updates = [int(row['update']) for row in rows]
    if updates != [375, 400, 425, 450, 475, 500] or not all(map(math.isfinite, values)):
        raise ValueError('Expected six finite stochastic validation scores at updates 375..500')
    return {'updates': updates, 'discounted_profit': values, 'mean': statistics.mean(values),
        'minimum': min(values), 'maximum': max(values), 'range': max(values) - min(values),
        'population_sd': statistics.pstdev(values), 'last_minus_first': values[-1] - values[0],
        'scope': 'Descriptive variation over six checkpoints on the same validation scenarios; not a convergence test or independent samples.'}


def reward_difference(candidate, baseline, *, discounted):
    key = 'mean_discounted_reward_terms' if discounted else 'mean_reward_terms'
    profit = 'mean_discounted_profit' if discounted else 'mean_profit'
    terms = candidate[key]
    if set(terms) != COST_TERMS | REVENUE_TERMS or set(terms) != set(baseline[key]):
        raise ValueError('Unexpected reward terms; update the explicit revenue/cost convention')
    differences = {term: terms[term] - baseline[key][term] for term in terms}
    contribution = {term: value * (-1 if term in COST_TERMS else 1) for term, value in differences.items()}
    profit_difference = candidate[profit] - baseline[profit]
    close(sum(contribution.values()), profit_difference, 'Business-reward decomposition does not close')
    return {'profit_difference': profit_difference, 'component_amount_difference': differences,
        'signed_profit_contribution': contribution, 'component_contribution_sum': sum(contribution.values()),
        'scope': 'Original business rewards. Training-only inventory potential and imitation loss are excluded.'}


def behavior(models):
    result = {'per_training_seed': {str(seed): {key: model[key] for key in BEHAVIOR_FIELDS}
                                   for seed, model in zip(SEEDS, models)},
              'mean_across_three_models': {key: statistics.mean(model[key] for model in models)
                                           for key in BEHAVIOR_FIELDS}}
    result['mode_actions_per_episode'] = {
        mode: statistics.mean(model['action_counts'][mode] / model['n_episodes'] for model in models)
        for mode in models[0]['action_counts']}
    result['price_scope'] = 'Benchmark mean_mobile_price averages all simulator quoted zone prices, including masked/unavailable nodes; it is not the active-quote or revenue-weighted price.'
    return result


def summarize(study, diagnostics):
    inputs = Inputs()
    protocol = inputs.json(study / 'protocol.json')
    benchmark_protocol = inputs.json(study / 'benchmarks/protocol.json')
    benchmark = inputs.json(study / 'benchmarks/summary.json')
    pilot_selection = inputs.json(study / 'pilot_selection.json')
    rule_selection = inputs.json(study / 'rule_selection/selection.json')
    rule_protocol = inputs.json(study / 'rule_selection/protocol.json')
    old_diagnostic = inputs.json(diagnostics / 'summary.json')
    audit = inputs.json(diagnostics / 'training_audit.json')
    if protocol['training_seeds'] != list(SEEDS) or benchmark_protocol['training_seeds'] != list(SEEDS):
        raise ValueError('The study must retain all three predeclared training seeds')
    if benchmark_protocol['study_protocol_sha256'] != digest(study / 'protocol.json'):
        raise ValueError('Final benchmark used a different root study protocol')
    if benchmark_protocol['primary_policy'] != PRIMARY or benchmark.get('test_tuning_performed') is not False:
        raise ValueError('Unexpected primary analysis or test-driven tuning declaration')
    if protocol['selected_bundle'] != pilot_selection['selected_bundle']:
        raise ValueError('Training bundle differs from the recorded pilot selection')
    if rule_selection['selection_protocol_sha256'] != digest(study / 'rule_selection/protocol.json'):
        raise ValueError('Rule-selection evidence differs from its frozen protocol')
    strongest = benchmark['strongest_validation_rule']
    if strongest != rule_selection['strongest_validation_family']:
        raise ValueError('The comparison rule was not selected on rule validation')

    training, pilots, warmstarts, late_rows = {}, {}, {}, {}
    validation_episodes = {'full_training': 0, 'pilots': 0}
    for group, runs, expected_updates in (
            ('full_training', [(str(seed), study / 'training' / f'seed_{seed}') for seed in SEEDS], 500),
            ('pilots', [(bundle, study / 'pilots' / bundle) for bundle in ('pure', 'guided')], 150)):
        for key, directory in runs:
            summary = inputs.json(directory / 'summary.json')
            metrics = inputs.csv(directory / 'training_metrics.csv')
            validation = inputs.csv(directory / 'validation_metrics.csv')
            if (summary['updates_this_run'] != expected_updates or summary['total_updates'] != expected_updates
                    or summary['training_physical_periods'] != expected_updates * 16 * 12
                    or summary['training_episodes'] != expected_updates * 16):
                raise ValueError('Incomplete or resumed training run: ' + str(directory))
            if not summary['all_metrics_finite'] or not summary['checkpoint_reload_matches'] or summary['invalid_evaluation_actions']:
                raise ValueError('Training verification failed: ' + str(directory))
            if [int(row['update']) for row in metrics] != list(range(1, expected_updates + 1)):
                raise ValueError('Training CSV does not contain the complete fresh run: ' + str(directory))
            if any(not math.isfinite(float(value)) for row in metrics for value in row.values()):
                raise ValueError('Training CSV contains nonfinite metrics: ' + str(directory))
            expected_validation = [(update, mode) for update in range(0, expected_updates + 1, 25)
                                   for mode in ('deterministic', 'stochastic')]
            if [(int(row['update']), row['policy']) for row in validation] != expected_validation:
                raise ValueError('Validation CSV does not contain the complete fixed schedule: ' + str(directory))
            # Each CSV row is one real evaluation; Trainer.train also repeats
            # the selected final evaluation once to verify checkpoint reload.
            validation_episodes[group] += (len(validation) + 1) * len(summary['eval_seeds'])
            if group == 'full_training':
                late = [row for row in validation if row['policy'] == 'stochastic' and 375 <= int(row['update']) <= 500]
                late_rows[key] = late
                training[key] = dict(summary, late_validation=window_statistics(late))
            else:
                pilots[key] = summary
            if (directory / 'warmstart.json').is_file():
                warm = inputs.json(directory / 'warmstart.json')
                if warm['demonstration_periods'] != 12 * warm['demonstration_episodes']:
                    raise ValueError('Inconsistent demonstration budget: ' + str(directory))
                warmstarts[f'{group}/{key}'] = warm
    if (protocol['selected_bundle'] == 'guided') != all(f'full_training/{seed}' in warmstarts for seed in SEEDS):
        raise ValueError('Full-run warm-start evidence does not match the selected bundle')
    aggregate_late = window_statistics([
        {'update': update, 'mean_discounted_profit': statistics.mean(
            float(next(row['mean_discounted_profit'] for row in late_rows[str(seed)] if int(row['update']) == update))
            for seed in SEEDS)} for update in (375, 400, 425, 450, 475, 500)])

    cases, case_counts, invalid = {}, {}, 0
    for name, case in benchmark['cases'].items():
        expected_seeds = benchmark_protocol['test_seeds'] if name == 'base' else benchmark_protocol['stress_seeds'][name]
        results = {}
        for path in sorted((study / 'benchmarks' / name).glob('*.json')):
            if path.name == 'summary.json':
                continue
            result = inputs.json(path)
            if result['evaluation_seeds'] != expected_seeds or len(result['episodes']) != len(expected_seeds):
                raise ValueError('Incomplete or inconsistent per-policy result: ' + str(path))
            results[path.stem] = result
        case_counts[name] = sum(len(result['episodes']) for result in results.values())
        if case_counts[name] != case['test_policy_episodes']:
            raise ValueError('Per-policy episode count differs from saved case summary: ' + name)
        invalid += sum(result['invalid_actions'] for result in results.values())
        new, old = case['model_averages'][PRIMARY], case['model_averages'][PREVIOUS]
        rule = results[strongest]
        row = {'new_final_stochastic_mean': new['mean_discounted_profit'],
            'previous_final_stochastic_mean': old['mean_discounted_profit'],
            'validation_selected_rule_mean': rule['mean_discounted_profit'],
            'rule_family': strongest,
            'new_final_mean_by_seed': new['mean_by_training_seed'],
            'previous_final_mean_by_seed': old['mean_by_training_seed'],
            'new_training_seed_sd': new['training_seed_sd'],
            'previous_training_seed_sd': old['training_seed_sd'],
            'paired_new_minus_previous': case['previous_to_final'],
            'paired_new_minus_rule': {
                'per_training_seed': {str(seed): case['comparisons_to_validation_best_rule'][f'rl_final_stochastic_{seed}'] for seed in SEEDS},
                'mean': case['comparisons_to_validation_best_rule'][PRIMARY]},
            'reward_decomposition': {
                baseline: {mode: reward_difference(new, other, discounted=discounted)
                    for mode, discounted in (('discounted', True), ('undiscounted', False))}
                for baseline, other in (('new_minus_previous', old), ('new_minus_rule', rule))},
            'test_policy_episodes': case_counts[name]}
        if name == 'base':
            row['behavior'] = {
                'new': behavior([results[f'rl_final_stochastic_{seed}'] for seed in SEEDS]),
                'previous': behavior([results[f'previous_final_stochastic_{seed}'] for seed in SEEDS]),
                'rule': {key: rule[key] for key in BEHAVIOR_FIELDS}}
            for group, average in (('new', new), ('previous', old)):
                for metric, value in row['behavior'][group]['mean_across_three_models'].items():
                    close(value, average[metric], 'Individual results disagree with model average')
            if 'warmstart_to_final' in case:
                row['warmstart_to_final_secondary'] = case['warmstart_to_final']
        cases[name] = row
    if set(cases) != {'base', 'high_demand', 'low_inventory'}:
        raise ValueError('All three predeclared evaluation cases must be complete')
    if sum(case_counts.values()) != benchmark['test_policy_episodes'] or invalid != benchmark['invalid_actions_all_evaluations']:
        raise ValueError('Global benchmark counts do not agree with individual saved evaluations')
    criteria = {}
    for name, comparison in (('consistent_previous_improvement', cases['base']['paired_new_minus_previous']),
                              ('consistent_rule_advantage', cases['base']['paired_new_minus_rule'])):
        criteria[name] = bool(all(value['mean_difference'] > 0 for value in comparison['per_training_seed'].values())
            and comparison['mean']['paired_bootstrap_confidence_interval_95'][0] > 0)
    if criteria != benchmark['success_criteria_results']:
        raise ValueError('Saved success outcomes disagree with the predeclared all-seed and interval criteria')

    full_periods = sum(row['training_physical_periods'] for row in training.values())
    pilot_periods = sum(row['training_physical_periods'] for row in pilots.values())
    bc_periods = {group: sum(row['demonstration_periods'] for key, row in warmstarts.items() if key.startswith(group + '/'))
                  for group in ('full_training', 'pilots')}
    if full_periods != 288000 or pilot_periods != 57600 or bc_periods['pilots'] != 3072:
        raise ValueError('Actual training/pilot budget differs from the declared study')
    if bc_periods['full_training'] != (9216 if protocol['selected_bundle'] == 'guided' else 0):
        raise ValueError('Actual full-run demonstration budget differs from the selected bundle')
    diagnostic_episodes = old_diagnostic['diagnostic_policy_episodes']
    audit_periods = audit['probe_method']['total_physical_periods']
    rule_episodes = rule_protocol['validation_policy_episodes']
    if (diagnostic_episodes, audit_periods, rule_episodes) != (640, 576, 7808):
        raise ValueError('Unexpected diagnostic, audit or rule-validation budget')
    posthoc_development_episodes = 0
    if (diagnostics / 'new_final/summary.json').is_file():
        posthoc_development_episodes = inputs.json(diagnostics / 'new_final/summary.json')['diagnostic_policy_episodes']
    budget = {'fresh_full_ppo_runs': 3, 'reused_new_training_runs': 0,
        'previous_control_checkpoints_reused': 3, 'full_ppo_physical_periods': full_periods,
        'pilot_ppo_physical_periods': pilot_periods, 'behavior_cloning_demonstration_periods': bc_periods,
        'ppo_plus_demonstration_physical_periods': full_periods + pilot_periods + sum(bc_periods.values()),
        'checkpoint_validation_and_reload_policy_episodes': validation_episodes,
        'independent_test_policy_episodes_by_case': case_counts,
        'independent_test_policy_episodes': sum(case_counts.values()),
        'rule_validation_policy_episodes': rule_episodes, 'development_diagnostic_policy_episodes': diagnostic_episodes,
        'frozen_training_audit_physical_periods': audit_periods,
        'additional_post_training_development_policy_episodes': posthoc_development_episodes,
        'scope': 'Simulation periods and executed policy episodes are distinct from unique scenario seeds. PPO counts exclude demonstrations and evaluation; previous controls are reused frozen weights. Unit-test simulations and any unrecorded manual checks are excluded.'}
    budget['accounted_simulation_physical_periods'] = (
        budget['ppo_plus_demonstration_physical_periods'] + audit_periods
        + 12 * (sum(validation_episodes.values()) + sum(case_counts.values()) + rule_episodes
                + diagnostic_episodes + posthoc_development_episodes))
    inputs.verify()
    return {'selected_bundle': protocol['selected_bundle'], 'pilot_selection': pilot_selection,
        'primary_policy': PRIMARY, 'strongest_validation_rule': strongest, 'cases': cases,
        'success_criteria': benchmark['success_criteria'], 'success_criteria_results': criteria,
        'training': training, 'late_validation_mean_across_three_seeds': aggregate_late,
        'warmstart_records': {key: {field: value for field, value in row.items() if field != 'environment_seeds'}
                              for key, row in warmstarts.items()},
        'budget': budget, 'invalid_test_actions': invalid,
        'limitations': ['Paired intervals condition on the fixed trained models; three seeds give limited retraining evidence.',
            'Late validation variation is descriptive and does not prove convergence.',
            'The selected training bundle changes multiple components; improvement cannot identify one component as its cause.',
            'Only final stochastic models determine the primary success criteria; deterministic, validation-best and warm-start comparisons remain secondary.'],
        'input_sha256': inputs.hashes, 'analysis_script_sha256': digest(Path(__file__))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/b3_improved_study')
    parser.add_argument('--diagnostic-dir', type=Path, default=ROOT / 'outputs/b3_gap_diagnostics')
    parser.add_argument('--output', type=Path, help='Defaults to <study-dir>/analysis.json; must not exist')
    args = parser.parse_args(argv)
    output = args.output or args.study_dir / 'analysis.json'
    if output.exists():
        parser.error('Analysis evidence already exists; choose a new output path')
    try:
        result = summarize(args.study_dir.resolve(), args.diagnostic_dir.resolve())
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print('Created ' + str(output.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
