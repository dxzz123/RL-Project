"""Audit a completed six-method Joint PPO study and print its saved results.

Run with --show to print an existing analysis without changing any file. The
default command independently audits and recalculates saved episode statistics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

SEEDS = (7, 19, 37)
PAPER, LEGACY, FITTED = 'paper_ppo_mean', 'legacy_ppo_mean', 'fitted_value_mean'
RULES = ('service_first', 'constant_inventory', 'time_inventory')
METHODS = (PAPER, LEGACY, FITTED, *RULES)
LABELS = {PAPER: 'Joint PPO', LEGACY: 'Earlier PPO', FITTED: 'Approximate DP',
          'service_first': 'Service first', 'constant_inventory': 'Constant inventory',
          'time_inventory': 'Time-dependent inventory'}


def _load_audit_dependencies():
    """Keep the saved-results viewer usable with the standard Python library."""
    global np, assert_close, audit_average, audit_individual, paired, reward_difference, verify_comparison
    if 'np' in globals():
        return
    import numpy as np
    from summarize_fitted_value_study import (
        assert_close, audit_average, audit_individual, paired, reward_difference,
        verify_comparison,
    )


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paper_reward_difference(candidate, baseline):
    result = reward_difference(candidate, baseline)
    result['interpretation'] = 'Positive entries increase the new Joint PPO profit. Cost reductions therefore appear positive; revenue reductions appear negative.'
    return result


def audit_operating_measures(result):
    """Check the operational explanation against saved individual episodes."""
    fields = {'mean_mobile_delivered_kwh': 'mobile_delivered_kwh',
              'mean_charge_grid_kwh': 'charge_grid_kwh',
              'mean_final_inventory_kwh': 'final_inventory_kwh',
              'mean_export_grid_kwh': 'export_grid_kwh',
              'mean_served_request_share': 'served_request_share',
              'mean_outside_option_share': 'outside_option_share',
              'mean_mobile_price': 'mean_mobile_price', 'mean_travel_kwh': 'travel_kwh'}
    for key, episode_key in fields.items():
        assert_close(result[key], np.mean([row[episode_key] for row in result['episodes']]),
                     result['label'] + '/operating/' + key)
    for mode, count in result['action_counts'].items():
        if count != sum(row['action_counts'][mode] for row in result['episodes']):
            raise ValueError('Saved action counts differ from the actual episode actions')
    if sum(result['action_counts'].values()) != 2*sum(row['periods'] for row in result['episodes']):
        raise ValueError('Action counts do not cover both vehicles in every physical period')


def audit_sources(study, root_protocol, evaluation_protocol):
    if evaluation_protocol['root_protocol_sha256'] != digest(study / 'protocol.json'):
        raise ValueError('Root protocol changed after final evaluation was frozen')
    from source_provenance import audit_frozen_sources, merge_source_hashes
    source_maps = [root_protocol['training_source_sha256'], evaluation_protocol['source_sha256']]
    for key, value in evaluation_protocol['checkpoint_sha256'].items():
        if digest(study / evaluation_protocol['checkpoint_paths'][key]) != value:
            raise ValueError('A frozen checkpoint changed: ' + key)
    for name, filename in (('selection', 'selection.json'), ('selection_protocol', 'protocol.json')):
        if digest(study / 'rule_selection' / filename) != evaluation_protocol[name + '_sha256']:
            raise ValueError('Validation-selection evidence changed')
    for seed in SEEDS:
        directory = study / 'training' / f'seed_{seed}'
        settings = read(directory / 'training_config.json')
        if settings != root_protocol['configuration_by_seed'][str(seed)]:
            raise ValueError('Recorded Joint PPO training settings differ from the root protocol')
        if digest(directory / 'summary.json') != evaluation_protocol['checkpoint_metadata'][f'paper_{seed}']['training_summary_sha256']:
            raise ValueError('Joint PPO training summary changed after evaluation was frozen')
        if digest(directory / 'protocol.json') != evaluation_protocol['checkpoint_metadata'][f'paper_{seed}']['training_protocol_sha256']:
            raise ValueError('Joint PPO training protocol changed after evaluation was frozen')
        training_protocol = read(directory / 'protocol.json')
        if training_protocol['training_config'] != settings:
            raise ValueError('Joint PPO training protocol and saved settings differ')
        source_maps.append(training_protocol['source_sha256'])
    status = audit_frozen_sources(ROOT, study, merge_source_hashes(*source_maps),
                                  root_protocol['training_source_sha256'])
    status.update(current_frozen_sources_verified=status['current_sources_verified'],
                  all_three_training_configurations_verified=True)
    return status


def monitor_record(directory, configuration):
    _load_audit_dependencies()
    with (directory / 'training_metrics.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if len(metrics) != 500 or not all(np.isfinite(float(value)) for row in metrics for value in row.values()):
        raise ValueError('Expected 500 finite training-metric rows')
    expected_optimizer_steps = configuration['epochs'] * int(np.ceil(
        configuration['episodes_per_update']*12/configuration['minibatch_size']))
    if any(int(row['optimizer_steps']) != expected_optimizer_steps or int(row['early_stopped']) != 0 for row in metrics):
        raise ValueError('The Joint PPO experiment must complete every configured PPO epoch without KL early stopping')
    summary = read(directory / 'summary.json')
    validation = read(directory / 'validation_history.json')
    stochastic = [row for row in validation if row.get('deterministic') is False]
    if not stochastic or int(stochastic[-1]['update']) != 500:
        raise ValueError('Missing final stochastic monitoring record')
    late = [float(row['mean_discounted_profit']) for row in stochastic if int(row['update']) >= 375]
    expected = [configuration['eval_seed_base']+2*i for i in range(configuration['eval_episodes'])]
    if any(row['evaluation_seeds'] != expected for row in validation):
        raise ValueError('Monitoring history contains scenarios outside the frozen monitor split')
    if [int(row['update']) for row in validation] != [0, *range(25, 501, 25)]:
        raise ValueError('Monitoring history differs from the frozen 25-update schedule')
    expected_monitoring = (2+configuration['updates']//configuration['eval_interval'])*configuration['eval_episodes']
    if (summary['monitoring_evaluation_episodes'] != expected_monitoring
            or summary['checkpoint_replay_episodes'] != configuration['eval_episodes']
            or summary['total_evaluation_policy_episodes'] != expected_monitoring+configuration['eval_episodes']):
        raise ValueError('Monitoring/replay episode counters differ from the actual training schedule')
    assert_close(float(stochastic[-1]['mean_discounted_profit']), summary['final_mean_discounted_profit'], 'Final monitored profit')
    return {'initial_stochastic_profit': float(stochastic[0]['mean_discounted_profit']),
        'final_stochastic_profit': float(stochastic[-1]['mean_discounted_profit']),
        'late_stochastic_validation_mean': float(np.mean(late)),
        'late_stochastic_validation_min': float(np.min(late)),
        'late_stochastic_validation_max': float(np.max(late)),
        'late_stochastic_validation_sd': float(np.std(late)),
        'late_validation_updates': [int(row['update']) for row in stochastic if int(row['update']) >= 375],
        'monitoring_policy_episodes': summary['monitoring_evaluation_episodes'],
        'reload_verification_episodes': summary['checkpoint_replay_episodes'],
        'optimizer_steps_total': sum(int(row['optimizer_steps']) for row in metrics),
        'optimizer_step_scope': 'Each counted minibatch update steps all four independent optimizers once.',
        'early_stopped_updates': sum(int(row['early_stopped']) for row in metrics),
        'late_training_mean': {key: float(np.mean([float(row[key]) for row in metrics[-125:]]))
            for key in ('pricing_entropy', 'dispatch_entropy', 'pricing_approx_kl', 'dispatch_approx_kl',
                        'pricing_explained_variance', 'dispatch_explained_variance',
                        'pricing_value_rmse', 'dispatch_value_rmse')},
        'scope': 'Monitoring scenarios support diagnostics only. The predeclared primary is the final 500-update checkpoint, regardless of the best monitored score.'}


def summarize(study):
    _load_audit_dependencies()
    root_protocol, protocol = read(study / 'protocol.json'), read(study / 'benchmarks/protocol.json')
    summary = read(study / 'benchmarks/summary.json')
    if summary['primary_policy'] != PAPER or summary['test_tuning_performed'] is not False:
        raise ValueError('Expected the untouched Joint PPO primary comparison')
    inputs = [study / 'protocol.json', *sorted((study / 'benchmarks').rglob('*.json')),
        *sorted((study / 'rule_selection').rglob('*.json')),
        *sorted((study / 'training').rglob('*.json')), *sorted((study / 'training').rglob('*.csv')),
        *sorted((study / 'training').rglob('*.pt')), *sorted((study / 'legacy').rglob('*.pt')),
        *sorted((study / 'fitted').rglob('*')), *sorted((study / 'provenance').rglob('*'))]
    test_log = study / 'logs/unit_tests.log'
    if test_log.is_file():
        inputs.append(test_log)
    hashes = {str(path.relative_to(study)): digest(path) for path in inputs if path.is_file()}
    source_provenance = audit_sources(study, root_protocol, protocol)
    selection, winners = read(study / 'rule_selection/selection.json'), {}
    validation_episodes, validation_invalid = 0, 0
    for spec in root_protocol['rule_candidates']:
        result = read(study / 'rule_selection/validation' / (spec['id'] + '.json'))
        audit_individual(result, root_protocol['splits']['validation'])
        audit_operating_measures(result)
        if result['policy_specification'] != spec:
            raise ValueError('Validation record differs from its frozen heuristic settings')
        row = dict(spec, mean_discounted_profit=result['mean_discounted_profit'])
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
        validation_episodes += result['n_episodes']
        validation_invalid += result['invalid_actions']
    strongest = max(winners, key=lambda name: winners[name]['mean_discounted_profit'])
    if winners != selection['family_winners'] or strongest != summary['strongest_validation_rule']:
        raise ValueError('The primary heuristic differs from independent validation selection')
    if validation_episodes != 10496:
        raise ValueError('Unexpected rule-validation policy-episode count')
    cases, executions, invalid = {}, 0, 0
    for case in ('base', 'high_demand', 'low_inventory'):
        saved, case_dir = summary['cases'][case], study / 'benchmarks' / case
        expected = root_protocol['splits'][case]
        if saved['strongest_validation_rule'] != strongest:
            raise ValueError('The validation-selected primary baseline changed across cases')
        individual_names = [*RULES, *[f'{method}_{seed}' for method in ('paper', 'legacy', 'fitted') for seed in SEEDS]]
        individuals = {name: read(case_dir / f'{name}.json') for name in individual_names}
        for result in individuals.values():
            audit_individual(result, expected)
            audit_operating_measures(result)
        averages = saved['model_averages']
        recomputed = {}
        for method, label in (('paper', PAPER), ('legacy', LEGACY), ('fitted', FITTED)):
            recomputed[label] = audit_average(averages[label], [individuals[f'{method}_{seed}'] for seed in SEEDS])
        combined = dict(individuals, **averages)
        comparisons = {}
        for name, comparison in saved['comparisons_to_validation_best_rule'].items():
            calculated = paired(combined[name], individuals[strongest])
            verify_comparison(comparison, calculated, case + '/versus_rule/' + name)
            comparisons[name] = calculated
        versus = {}
        for method, label, field in (('legacy', LEGACY, 'paper_vs_legacy'),
                                     ('fitted', FITTED, 'paper_vs_fitted')):
            calculated = {'mean': paired(recomputed[PAPER], recomputed[label]), 'per_training_seed': {}}
            verify_comparison(saved[field]['mean'], calculated['mean'], case + '/' + field + '/mean')
            for seed in SEEDS:
                result = paired(individuals[f'paper_{seed}'], individuals[f'{method}_{seed}'])
                verify_comparison(saved[field]['per_training_seed'][str(seed)], result, case + '/' + field + '/' + str(seed))
                calculated['per_training_seed'][str(seed)] = result
            versus[method] = calculated
        if [row['policy'] for row in saved['table']] != list(METHODS):
            raise ValueError('Expected exactly the six predeclared summary rows')
        for row in saved['table']:
            assert_close(row['mean_discounted_profit'], combined[row['policy']]['mean_discounted_profit'], case + '/table/' + row['policy'])
        behaviors = {name: {key.removeprefix('mean_'): combined[name][key] for key in
            ('mean_mobile_delivered_kwh', 'mean_charge_grid_kwh', 'mean_final_inventory_kwh',
             'mean_export_grid_kwh', 'mean_served_request_share', 'mean_outside_option_share', 'mean_mobile_price')}
            for name in METHODS}
        for name in METHODS:
            member_names = ({PAPER: [f'paper_{seed}' for seed in SEEDS],
                             LEGACY: [f'legacy_{seed}' for seed in SEEDS],
                             FITTED: [f'fitted_{seed}' for seed in SEEDS]}.get(name, [name]))
            members = [individuals[key] for key in member_names]
            for key in ('mean_mobile_delivered_kwh', 'mean_charge_grid_kwh', 'mean_final_inventory_kwh',
                        'mean_export_grid_kwh', 'mean_served_request_share', 'mean_outside_option_share', 'mean_mobile_price'):
                assert_close(combined[name][key], np.mean([member[key] for member in members]), name + '/operating-average/' + key)
            behaviors[name]['action_fractions'] = {mode: float(np.mean([
                member['action_counts'][mode]/sum(member['action_counts'].values()) for member in members]))
                for mode in ('WAIT', 'SERVE', 'REPOSITION', 'RECHARGE', 'DISCHARGE')}
            behaviors[name]['travel_kwh'] = float(np.mean([member['mean_travel_kwh'] for member in members]))
            behaviors[name]['aggregation'] = 'Mean of actual per-model operating measures; no averaged or ensemble trajectory is executed.'
        descriptive_best = max(RULES, key=lambda name: individuals[name]['mean_discounted_profit'])
        cases[case] = {'mean_discounted_profits': {name: combined[name]['mean_discounted_profit'] for name in METHODS},
            'paper_by_training_seed': averages[PAPER]['mean_by_training_seed'],
            'legacy_by_training_seed': averages[LEGACY]['mean_by_training_seed'],
            'fitted_by_training_seed': averages[FITTED]['mean_by_training_seed'],
            'paper_training_seed_sd': averages[PAPER]['training_seed_sd'],
            'new_vs_validation_best_rule': {'mean': comparisons[PAPER],
                'per_training_seed': {str(seed): comparisons[f'paper_{seed}'] for seed in SEEDS}},
            'new_vs_legacy': versus['legacy'], 'new_vs_fitted': versus['fitted'],
            'new_vs_each_selected_rule': {name: paired(averages[PAPER], individuals[name]) for name in RULES},
            'reward_decomposition': {'versus_validation_best_rule': paper_reward_difference(averages[PAPER], individuals[strongest]),
                'versus_legacy': paper_reward_difference(averages[PAPER], averages[LEGACY]),
                'versus_fitted': paper_reward_difference(averages[PAPER], averages[FITTED])},
            'behaviors': behaviors, 'descriptive_highest_profit_heuristic': descriptive_best,
            'new_vs_descriptive_best_heuristic': paired(averages[PAPER], individuals[descriptive_best]),
            'descriptive_scope': 'The observed highest-profit heuristic is a descriptive result. It does not replace the primary baseline chosen on validation.',
            'test_policy_episodes': len(expected)*12,
            'invalid_actions': sum(result['invalid_actions'] for result in individuals.values())}
        executions += cases[case]['test_policy_episodes']
        invalid += cases[case]['invalid_actions']
        if saved['test_policy_episodes'] != len(expected)*12 or saved['invalid_actions_all_evaluations'] != cases[case]['invalid_actions']:
            raise ValueError('Saved case execution or invalid-action counts differ')
    def positive_all(result):
        return bool(result['mean']['paired_bootstrap_confidence_interval_95'][0] > 0
            and all(item['paired_bootstrap_confidence_interval_95'][0] > 0 for item in result['per_training_seed'].values()))
    criteria = {'consistent_heuristic_advantage': positive_all(cases['base']['new_vs_validation_best_rule']),
        'consistent_legacy_ppo_improvement': positive_all(cases['base']['new_vs_legacy']),
        'consistent_fitted_value_advantage': positive_all(cases['base']['new_vs_fitted'])}
    if (criteria != summary['success_criteria_results'] or executions != 12288
            or executions != summary['test_policy_episodes'] or invalid != summary['invalid_actions_all_evaluations']):
        raise ValueError('Saved consistency criteria or overall test counters differ')
    training = {str(seed): read(study / 'training' / f'seed_{seed}' / 'summary.json') for seed in SEEDS}
    monitor = {str(seed): monitor_record(study / 'training' / f'seed_{seed}',
        root_protocol['configuration_by_seed'][str(seed)]) for seed in SEEDS}
    budgets = {'new_training_updates_per_seed': 500, 'new_training_episodes_per_seed': 8000,
        'new_training_physical_periods_per_seed': 96000, 'new_training_episodes_total': 24000,
        'new_training_physical_periods_total': 288000,
        'rule_validation_policy_episodes': validation_episodes, 'invalid_rule_validation_actions': validation_invalid,
        'test_policy_episodes': executions, 'invalid_test_actions': invalid,
        'base_independent_scenarios': 512, 'stress_independent_scenarios_per_case': 256,
        'monitoring_policy_episodes': sum(row['monitoring_policy_episodes'] for row in monitor.values()),
        'checkpoint_reload_verification_episodes': sum(row['reload_verification_episodes'] for row in monitor.values()),
        'optimizer_steps_total': sum(row['optimizer_steps_total'] for row in monitor.values()),
        'individual_optimizer_steps_total': 4*sum(row['optimizer_steps_total'] for row in monitor.values()),
        'optimizer_step_scope': 'optimizer_steps_total counts PPO minibatch updates. Each update steps four separate actor/critic optimizers; individual_optimizer_steps_total counts those four calls separately.',
        'control_scope': 'Previously trained legacy PPO and fitted-value controls were copied without retraining. Fitted value uses generative state queries and known physics; coordinated heuristics use known physics. This is not an equal-information or equal-compute contest.'}
    smoke_dir = study / 'provenance/development_smoke'
    if (smoke_dir / 'summary.json').is_file():
        smoke = read(smoke_dir / 'summary.json')
        records = {}
        for name in ('main_target', 'optional_a1_target'):
            path = smoke_dir / name / 'summary.json'
            if not path.is_file():
                raise ValueError('Recorded development smoke summary is missing its target result: ' + name)
            row = read(path)
            selected_fields = ('update_count', 'episode_count', 'environment_periods',
                'all_metrics_finite', 'checkpoint_reload_matches', 'invalid_evaluation_actions',
                'monitoring_evaluation_episodes', 'checkpoint_replay_episodes', 'total_evaluation_policy_episodes')
            if any(row[key] != smoke[name][key] for key in selected_fields):
                raise ValueError('Aggregated smoke counters differ from the underlying target result')
            if (row['all_metrics_finite'] is not True or row['checkpoint_reload_matches'] is not True
                    or row['invalid_evaluation_actions'] != 0):
                raise ValueError('A recorded development execution check failed')
            records[name] = {key: row[key] for key in selected_fields}
        budgets['separate_development_smoke'] = {
            'current_source_runs': records,
            'current_source_training_episodes': sum(row['episode_count'] for row in records.values()),
            'current_source_physical_periods': sum(row['environment_periods'] for row in records.values()),
            'current_source_evaluation_policy_episodes': sum(row['total_evaluation_policy_episodes'] for row in records.values()),
            'earlier_preflight': smoke.get('earlier_preflight'),
            'scope': smoke['scope'],
            'evidence_scope': 'Execution checks only. Their profits were not used to select the main method, settings, checkpoint or reported controller, and they are excluded from primary training and final test budgets.'}
    test_record = {'available': False}
    if test_log.is_file():
        text = test_log.read_text(encoding='utf-8')
        runs = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', text)
        successful_runs = re.findall(r'^OK\s*$', text, re.MULTILINE)
        test_record = {'available': True, 'log': 'logs/unit_tests.log', 'sha256': digest(test_log),
            'tests': sum(int(count) for count, _ in runs) if runs else None,
            'elapsed_seconds': sum(float(seconds) for _, seconds in runs) if runs else None,
            'runs': [{'tests': int(count), 'elapsed_seconds': float(seconds)} for count, seconds in runs],
            'passed': bool(runs and len(successful_runs) == len(runs)
                           and not re.search(r'^FAILED\b', text, re.MULTILINE)),
            'scope': 'Sum over recorded disjoint test-suite runs; each recorded run must end in OK.'}
    result = {'study': root_protocol['study'], 'primary_baseline': strongest, 'cases': cases,
        'success_criteria_results': criteria, 'budgets': budgets,
        'monitoring': monitor, 'training_summaries': training, 'unit_tests': test_record,
        'source_provenance': source_provenance, 'audit_source_sha256': digest(Path(__file__)),
        'source_audit_helper_sha256': digest(ROOT / 'experiments/source_provenance.py'),
        'audit_helper_sha256': digest(ROOT / 'experiments/summarize_fitted_value_study.py'),
        'independent_audit': 'Saved individual profits, all primary paired bootstrap comparisons, actual individual-model risk averages, business-reward ledgers, tables, validation choices, counts, consistency criteria, frozen sources and checkpoints independently verified. Source snapshots are strictly verified when present.',
        'uncertainty_scope': 'Conditional scenario/action sampling intervals for fixed models, excluding uncertainty over arbitrary future retraining. Three training seeds give limited retraining evidence.',
        'input_sha256': hashes}
    if audit_sources(study, root_protocol, protocol) != source_provenance:
        raise ValueError('Source provenance changed during the independent audit')
    if any(digest(study / name) != value for name, value in hashes.items()):
        raise ValueError('Saved study evidence changed during analysis')
    output = study / 'analysis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def show_results(result):
    """Print the compact saved comparison without running any policy."""
    print('Saved comparison: mean discounted business profit per episode')
    print(f'{"Method":<28} {"Base":>10} {"High demand":>12} {"Low inventory":>14}')
    for name in METHODS:
        values = [result['cases'][case]['mean_discounted_profits'][name]
                  for case in ('base', 'high_demand', 'low_inventory')]
        print(f'{LABELS[name]:<28} {values[0]:>10.2f} {values[1]:>12.2f} {values[2]:>14.2f}')
    print('\nBase-case Joint PPO minus each frozen comparison:')
    for field, label in (('new_vs_validation_best_rule', 'Validation-selected rule'),
                         ('new_vs_legacy', 'Earlier PPO'), ('new_vs_fitted', 'Approximate DP')):
        row = result['cases']['base'][field]['mean']
        low, high = row['paired_bootstrap_confidence_interval_95']
        print(f'{label:<28} {row["mean_difference"]:+.2f}  paired 95% interval [{low:+.2f}, {high:+.2f}]')
    print('Intervals describe scenario/action uncertainty for these fixed models; they exclude retraining uncertainty.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/paper_ppo_study')
    parser.add_argument('--show', action='store_true', help='Print saved six-method profits and primary paired intervals')
    args = parser.parse_args(argv)
    if args.show:
        path = args.study_dir.resolve() / 'analysis.json'
        if not path.is_file():
            parser.error('No saved analysis.json exists. Run this command once without --show to audit the completed study first.')
        show_results(read(path))
    else:
        summarize(args.study_dir.resolve())
        print(args.study_dir.resolve() / 'analysis.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
