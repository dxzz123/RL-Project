"""Independently audit and explain a completed frozen fitted-value comparison.

This post-processing command reads saved episodes only. It does not execute a
controller, select settings, fit values, or access any new evaluation scenario.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

SEEDS = (7, 19, 37)
METHODS = ('fitted_value_mean', 'ppo_mean', 'service_first', 'constant_inventory', 'time_inventory')
COST_TERMS = {'procurement_cost', 'travel_cost', 'operating_cost', 'unmet_demand_cost'}
REVENUE_TERMS = {'mobile_service_revenue', 'fixed_service_revenue', 'discharge_revenue',
                 'terminal_inventory_value'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_close(actual, expected, label):
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-12):
        raise ValueError(f'{label} differs from the independently recomputed value: {actual!r} versus {expected!r}')


def profits(result):
    rows = result['episodes']
    seeds = [row['seed'] for row in rows]
    if len(set(seeds)) != len(seeds) or seeds != result['evaluation_seeds']:
        raise ValueError('Episode seeds must be unique and match their saved evaluation order')
    values = np.asarray([row['discounted_profit'] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Saved episode profits contain nonfinite values')
    return seeds, values


def paired(candidate, baseline):
    """Bootstrap saved paired profits independently of the evaluation helper."""
    seeds, values = profits(candidate)
    baseline_seeds, baseline_values = profits(baseline)
    if set(seeds) != set(baseline_seeds):
        raise ValueError('A paired comparison has mismatched scenario seeds')
    lookup = dict(zip(baseline_seeds, baseline_values))
    delta = values - np.asarray([lookup[seed] for seed in seeds])
    rng = np.random.default_rng(271828)
    # One independent calculation over the complete bootstrap index array. The
    # runner uses small chunks; both consume exactly the same RNG sequence.
    bootstrap = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    return {'mean_difference': float(delta.mean()),
            'paired_bootstrap_confidence_interval_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'candidate_win_fraction': float(np.mean(delta > 0.)),
            'tie_fraction': float(np.mean(delta == 0.)), 'n_episodes': len(delta),
            'bootstrap_samples': 5000, 'bootstrap_seed': 271828,
            'uncertainty_scope': 'Scenario/action sampling conditional on these fixed trained models; excludes retraining uncertainty.'}


def verify_comparison(saved, computed, label):
    for key in ('mean_difference', 'paired_bootstrap_confidence_interval_95',
                'candidate_win_fraction', 'tie_fraction', 'n_episodes',
                'bootstrap_samples', 'bootstrap_seed'):
        assert_close(saved[key], computed[key], label + '/' + key)


def audit_individual(result, expected_seeds):
    seeds, values = profits(result)
    if seeds != expected_seeds or result['n_episodes'] != len(seeds):
        raise ValueError('An individual policy result uses the wrong case split or episode count')
    if sum(row['invalid_actions'] for row in result['episodes']) != result['invalid_actions']:
        raise ValueError('Saved invalid-action counts do not add up')
    assert_close(result['mean_discounted_profit'], values.mean(), result['label'] + '/mean_profit')
    assert_close(result['profit_p10'], np.quantile(values, .1), result['label'] + '/p10')
    assert_close(result['lower_tail_mean_10pct'], np.sort(values)[:math.ceil(.1*len(values))].mean(), result['label'] + '/tail')
    assert_close(result['loss_probability'], np.mean(values < 0.), result['label'] + '/loss_probability')
    terms = result['mean_discounted_reward_terms']
    if set(terms) != COST_TERMS | REVENUE_TERMS:
        raise ValueError('Unexpected business-reward ledger fields')
    for key in terms:
        assert_close(terms[key], np.mean([row['discounted_reward_terms'][key] for row in result['episodes']]),
                     result['label'] + '/ledger/' + key)
    reconstructed = sum(terms[key] for key in REVENUE_TERMS) - sum(terms[key] for key in COST_TERMS)
    assert_close(reconstructed, values.mean(), result['label'] + '/ledger_total')


def audit_average(saved, individuals):
    seeds = individuals[0]['evaluation_seeds']
    values = np.mean([profits(item)[1] for item in individuals], axis=0)
    assert_close(profits(saved)[1], values, saved['label'] + '/scene_average')
    assert_close(saved['mean_discounted_profit'], values.mean(), saved['label'] + '/mean')
    means = [item['mean_discounted_profit'] for item in individuals]
    assert_close(saved['training_seed_sd'], np.std(means, ddof=1), saved['label'] + '/training_seed_sd')
    for seed, item in zip(SEEDS, individuals):
        if item['training_seed'] != seed:
            raise ValueError('Model average has reordered or missing training seeds')
        assert_close(saved['mean_by_training_seed'][str(seed)], item['mean_discounted_profit'], saved['label'] + '/seed_mean')
    for key in ('profit_p10', 'lower_tail_mean_10pct', 'loss_probability',
                'mean_mobile_delivered_kwh', 'mean_charge_grid_kwh', 'mean_final_inventory_kwh'):
        assert_close(saved[key], np.mean([item[key] for item in individuals]), saved['label'] + '/' + key)
    for key in saved['mean_discounted_reward_terms']:
        assert_close(saved['mean_discounted_reward_terms'][key],
                     np.mean([item['mean_discounted_reward_terms'][key] for item in individuals]),
                     saved['label'] + '/ledger/' + key)
    return {'label': saved['label'], 'evaluation_seeds': seeds,
            'episodes': [{'seed': seed, 'discounted_profit': float(value)} for seed, value in zip(seeds, values)]}


def reward_difference(candidate, baseline):
    terms = {key: (candidate['mean_discounted_reward_terms'][key] - baseline['mean_discounted_reward_terms'][key])
             * (-1. if key in COST_TERMS else 1.) for key in COST_TERMS | REVENUE_TERMS}
    change = candidate['mean_discounted_profit'] - baseline['mean_discounted_profit']
    assert_close(sum(terms.values()), change, 'Reward-decomposition total')
    return {'discounted_profit_difference': change, 'signed_contributions': terms,
            'interpretation': 'Positive entries increase fitted-value profit. Cost reductions therefore appear positive; revenue reductions appear negative.'}


def audit_sources(study, root_protocol, evaluation_protocol):
    if evaluation_protocol['root_protocol_sha256'] != digest(study / 'protocol.json'):
        raise ValueError('Root protocol hash differs from the frozen evaluation protocol')
    for name, value in evaluation_protocol['source_sha256'].items():
        if digest(ROOT / name) != value:
            raise ValueError('Frozen source differs: ' + name)
    selection = study / 'rule_selection'
    if (digest(selection / 'selection.json') != evaluation_protocol['selection_sha256']
            or digest(selection / 'protocol.json') != evaluation_protocol['selection_protocol_sha256']):
        raise ValueError('Rule-selection evidence differs from its frozen hash')
    for name, value in evaluation_protocol['checkpoint_sha256'].items():
        if digest(study / evaluation_protocol['checkpoint_paths'][name]) != value:
            raise ValueError('Frozen checkpoint differs: ' + name)
    training_protocols = {}
    for seed in SEEDS:
        path = study / 'fitted' / f'seed_{seed}' / 'protocol.json'
        protocol = read(path)
        if protocol['training_config'] != root_protocol['fitted_configuration_by_seed'][str(seed)]:
            raise ValueError('Training configuration differs from its root protocol')
        for name, value in protocol['source_sha256'].items():
            if digest(ROOT / name) != value:
                raise ValueError('Current source differs from the executed training protocol: ' + name)
        training_protocols[str(seed)] = protocol
    status = {'current_training_sources_verified': True,
              'all_three_root_configurations_verified': True,
              'snapshot_available': False, 'snapshot_verified': False,
              'scope': 'Current source files match every frozen training protocol; no additional executed-source snapshot is available for this repeat study.'}
    snapshot = study / 'provenance/executed_training_source'
    if not (snapshot / 'manifest.json').is_file():
        return status
    manifest = read(snapshot / 'manifest.json')
    if (manifest['root_protocol_sha256'] != digest(study / 'protocol.json')
            or digest(snapshot / 'root_protocol.json') != digest(study / 'protocol.json')):
        raise ValueError('Executed-source snapshot refers to a different root protocol')
    for name, entry in manifest['sources'].items():
        value = digest(snapshot / entry['snapshot_relative_path'])
        if value != entry['snapshot_sha256'] or value != entry['frozen_protocol_sha256']:
            raise ValueError('Executed training-source snapshot differs: ' + name)
    for seed in SEEDS:
        protocol_path = study / 'fitted' / f'seed_{seed}' / 'protocol.json'
        entry = manifest['training_protocols'][str(seed)]
        if digest(protocol_path) != entry['sha256'] or digest(snapshot / entry['snapshot_relative_path']) != entry['sha256']:
            raise ValueError('Executed training-protocol snapshot differs for seed ' + str(seed))
        protocol = training_protocols[str(seed)]
        for name, value in protocol['source_sha256'].items():
            if manifest['sources'][name]['frozen_protocol_sha256'] != value:
                raise ValueError('Training-source provenance differs from executed snapshot')
    status.update(snapshot_available=True, snapshot_verified=True,
                  scope='Current source files match every frozen training protocol; the additional executed-source snapshot, its root protocol and all three training protocols also match their recorded hashes.')
    return status


def summarize(study):
    summary_path = study / 'benchmarks/summary.json'
    summary = read(summary_path)
    root_protocol, protocol = read(study / 'protocol.json'), read(study / 'benchmarks/protocol.json')
    if summary['test_tuning_performed'] is not False or summary['primary_policy'] != 'fitted_value_mean':
        raise ValueError('Expected the untouched, fitted-value primary evaluation')
    # Hash only the completed evidence this analysis reads, so later plots and
    # publication checks can be added without changing its scientific inputs.
    inputs = [study / 'protocol.json', *sorted((study / 'benchmarks').rglob('*.json')),
              *sorted((study / 'rule_selection').rglob('*.json')),
              *sorted((study / 'fitted').rglob('*.json')),
              *sorted((study / 'fitted').rglob('*.csv')),
              *sorted((study / 'fitted').rglob('*.npz')),
              *sorted((study / 'ppo').rglob('*.pt')),
              *sorted((study / 'provenance/executed_training_source').rglob('*')),
              *sorted((study / 'provenance/development_smoke').rglob('*.json'))]
    test_log = study / 'logs/unit_tests.log'
    if test_log.is_file():
        inputs.append(test_log)
    hashes = {str(path.relative_to(study)): digest(path) for path in inputs if path.is_file()}
    source_provenance = audit_sources(study, root_protocol, protocol)
    selection = read(study / 'rule_selection/selection.json')
    validation_episodes, validation_invalid, winners = 0, 0, {}
    for spec in root_protocol['rule_candidates']:
        record = read(study / 'rule_selection/validation' / (spec['id'] + '.json'))
        audit_individual(record, root_protocol['splits']['validation'])
        if record['policy_specification'] != spec:
            raise ValueError('Rule-validation record changed its frozen candidate settings')
        validation_episodes += record['n_episodes']
        validation_invalid += record['invalid_actions']
        row = dict(spec, mean_discounted_profit=record['mean_discounted_profit'])
        if spec['family'] not in winners or row['mean_discounted_profit'] > winners[spec['family']]['mean_discounted_profit']:
            winners[spec['family']] = row
    if (validation_episodes != 10496 or selection['family_winners'] != winners
            or selection['strongest_validation_family'] != max(winners, key=lambda key: winners[key]['mean_discounted_profit'])):
        raise ValueError('Validation episode budget or selected family winners differ from independent recomputation')
    cases, executions, invalid = {}, 0, 0
    strongest = summary['strongest_validation_rule']
    for case in ('base', 'high_demand', 'low_inventory'):
        case_dir, saved = study / 'benchmarks' / case, summary['cases'][case]
        count = 512 if case == 'base' else 256
        if saved['strongest_validation_rule'] != strongest:
            raise ValueError('The validation-selected primary baseline changed across cases')
        expected_seeds = root_protocol['splits'][case]
        individuals = {name: read(case_dir / f'{name}.json') for name in
                       ['service_first', 'constant_inventory', 'time_inventory',
                        *[f'{method}_{seed}' for method in ('fitted', 'ppo') for seed in SEEDS]]}
        for item in individuals.values():
            audit_individual(item, expected_seeds)
        if sum(len(item['episodes']) for item in individuals.values()) != 9*count:
            raise ValueError('Actual policy-episode executions differ from the nine-controller case budget')
        averages = saved['model_averages']
        recomputed = {}
        for method, mean_label in (('fitted', 'fitted_value_mean'), ('ppo', 'ppo_mean')):
            recomputed[mean_label] = audit_average(averages[mean_label], [individuals[f'{method}_{seed}'] for seed in SEEDS])
        combined = dict(individuals, **averages)
        pairs = {}
        for name, comparison in saved['comparisons_to_validation_best_rule'].items():
            computed = paired(combined[name], individuals[strongest])
            verify_comparison(comparison, computed, case + '/vs_rule/' + name)
            pairs[name] = computed
        versus_ppo = {'per_training_seed': {}, 'mean': paired(recomputed['fitted_value_mean'], recomputed['ppo_mean'])}
        verify_comparison(saved['fitted_vs_ppo']['mean'], versus_ppo['mean'], case + '/vs_PPO/mean')
        for seed in SEEDS:
            comparison = paired(individuals[f'fitted_{seed}'], individuals[f'ppo_{seed}'])
            verify_comparison(saved['fitted_vs_ppo']['per_training_seed'][str(seed)], comparison, case + '/vs_PPO/' + str(seed))
            versus_ppo['per_training_seed'][str(seed)] = comparison
        for row in saved['table']:
            assert_close(row['mean_discounted_profit'], combined[row['policy']]['mean_discounted_profit'], case + '/table/' + row['policy'])
        if [row['policy'] for row in saved['table']] != list(METHODS):
            raise ValueError('The compact table must retain exactly the five predeclared rows')
        descriptive_best = max(('service_first', 'constant_inventory', 'time_inventory'),
                               key=lambda name: individuals[name]['mean_discounted_profit'])
        behaviors = {}
        for name in METHODS:
            item = combined[name]
            behaviors[name] = {key.removeprefix('mean_'): item[key] for key in
                ('mean_mobile_delivered_kwh', 'mean_charge_grid_kwh', 'mean_final_inventory_kwh',
                 'mean_export_grid_kwh', 'mean_served_request_share', 'mean_outside_option_share', 'mean_mobile_price')}
        cases[case] = {'mean_discounted_profits': {name: combined[name]['mean_discounted_profit'] for name in METHODS},
            'fitted_by_training_seed': averages['fitted_value_mean']['mean_by_training_seed'],
            'ppo_by_training_seed': averages['ppo_mean']['mean_by_training_seed'],
            'fitted_training_seed_sd': averages['fitted_value_mean']['training_seed_sd'],
            'fitted_vs_validation_best_rule': {'mean': pairs['fitted_value_mean'],
                'per_training_seed': {str(seed): pairs[f'fitted_{seed}'] for seed in SEEDS}},
            'fitted_vs_ppo': versus_ppo,
            'fitted_vs_each_selected_rule': {name: paired(averages['fitted_value_mean'], individuals[name])
                                           for name in ('service_first', 'constant_inventory', 'time_inventory')},
            'percent_gain_over_validation_best_rule': 100*pairs['fitted_value_mean']['mean_difference']/individuals[strongest]['mean_discounted_profit'],
            'percent_gain_over_ppo': 100*versus_ppo['mean']['mean_difference']/averages['ppo_mean']['mean_discounted_profit'],
            'reward_decomposition': {'versus_validation_best_rule': reward_difference(averages['fitted_value_mean'], individuals[strongest]),
                                     'versus_ppo': reward_difference(averages['fitted_value_mean'], averages['ppo_mean'])},
            'behaviors': behaviors,
            'descriptive_highest_profit_heuristic': descriptive_best,
            'fitted_vs_descriptive_best_heuristic': paired(averages['fitted_value_mean'], individuals[descriptive_best]),
            'descriptive_scope': 'Highest observed heuristic is a descriptive stress-test result, not a newly selected controller; the primary baseline remains the rule chosen on base validation.',
            'test_policy_episodes': 9*count,
            'invalid_actions': sum(item['invalid_actions'] for item in individuals.values())}
        executions += 9*count
        invalid += cases[case]['invalid_actions']
        if saved['test_policy_episodes'] != 9*count or saved['invalid_actions_all_evaluations'] != cases[case]['invalid_actions']:
            raise ValueError('Case policy-execution or invalid-action totals differ')
    base = cases['base']
    rule_results, ppo_results = base['fitted_vs_validation_best_rule'], base['fitted_vs_ppo']
    def positive_all(result):
        return bool(result['mean']['paired_bootstrap_confidence_interval_95'][0] > 0
            and all(item['paired_bootstrap_confidence_interval_95'][0] > 0 for item in result['per_training_seed'].values()))
    criteria = {'consistent_heuristic_advantage': positive_all(rule_results),
                'consistent_ppo_advantage': positive_all(ppo_results)}
    if criteria != summary['success_criteria_results'] or executions != summary['test_policy_episodes'] or invalid != summary['invalid_actions_all_evaluations']:
        raise ValueError('Overall criteria or evaluation counters differ from independent calculations')
    training = {str(seed): read(study / 'fitted' / f'seed_{seed}' / 'summary.json') for seed in SEEDS}
    budget_fields = ('state_draw_queries', 'candidate_outcomes', 'unique_exogenous_draw_seeds', 'training_wall_seconds')
    budgets = {'fitted_by_seed': {seed: {key: row[key] for key in budget_fields} for seed, row in training.items()},
        'fitted_totals': {key: sum(row[key] for row in training.values()) for key in budget_fields},
        'fitted_on_policy_training_episodes': 0,
        'fitted_budget_scope': 'State/draw queries use 96 unique period/draw seeds per model reused across 1296 fleet states per layer; candidate outcomes are exact measured counts. Concurrent wall times are per-process elapsed measurements and their sum is not overall elapsed study time.',
        'ppo_training_physical_periods_per_model': 96000, 'ppo_updates_per_model': 500,
        'ppo_scope': 'Previously trained frozen controls copied without retraining; not an equal-information or equal-compute comparison.',
        'rule_validation_policy_episodes': 82*128, 'test_policy_episodes': executions,
        'invalid_rule_validation_actions': validation_invalid,
        'base_independent_scenarios': 512, 'stress_independent_scenarios_per_case': 256,
        'invalid_test_actions': invalid}
    smoke_dir = study / 'provenance/development_smoke'
    if (smoke_dir / 'summary.json').is_file() and (smoke_dir / 'execution_evaluation.json').is_file():
        smoke_training, smoke_execution = read(smoke_dir / 'summary.json'), read(smoke_dir / 'execution_evaluation.json')
        holdouts = {seed for seeds in root_protocol['splits'].values() for seed in seeds}
        if set(smoke_execution['evaluation_seeds']) & holdouts:
            raise ValueError('Development smoke scenarios overlap the frozen comparison')
        budgets['separate_development_smoke'] = {
            'state_draw_queries': smoke_training['state_draw_queries'],
            'candidate_outcomes': smoke_training['candidate_outcomes'],
            'unique_exogenous_draw_seeds': smoke_training['unique_exogenous_draw_seeds'],
            'evaluation_episodes': smoke_execution['n_episodes'],
            'invalid_actions': smoke_execution['invalid_actions'],
            'scope': 'Coarse-grid execution check before the full fixed study; excluded from the three full-model and primary test budgets.'}
    test_record = {'available': False}
    if test_log.is_file():
        log = test_log.read_text(encoding='utf-8')
        match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', log)
        test_record = {'available': True, 'log': 'logs/unit_tests.log', 'sha256': digest(test_log),
                       'tests': int(match.group(1)) if match else None,
                       'elapsed_seconds': float(match.group(2)) if match else None,
                       'passed': bool(match and re.search(r'^OK\s*$', log, re.MULTILINE))}
    result = {'study': root_protocol['study'], 'primary_baseline': strongest,
        'cases': cases, 'success_criteria_results': criteria, 'budgets': budgets,
        'unit_tests': test_record,
        'audit_source_sha256': digest(Path(__file__)),
        'source_provenance': source_provenance,
        'independent_audit': 'All primary and saved rule comparisons recomputed directly from individual episode profits with an independent bootstrap calculation; tables, actual-model risk averages, reward ledgers, scenario pairing, counts, criteria, current training sources and checkpoints verified. An additional executed-source snapshot is strictly verified when present; source_provenance records whether it is available.',
        'uncertainty_scope': 'Conditional scene/action sampling intervals for fixed policies. Three training seeds provide limited evidence about future retraining. No test-driven controller selection or refitting.',
        'input_sha256': hashes}
    if audit_sources(study, root_protocol, protocol) != source_provenance:
        raise ValueError('Source provenance changed during the independent audit')
    if any(digest(study / name) != value for name, value in hashes.items()):
        raise ValueError('Saved evidence changed while the independent audit ran')
    output = study / 'analysis.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/fitted_value_study')
    args = parser.parse_args(argv)
    print(summarize(args.study_dir.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
