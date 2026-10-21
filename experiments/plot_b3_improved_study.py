"""Plot a completed b=3 improvement study as one six-panel PNG, never a PDF.

The plotter consumes saved evidence only. It does not train, simulate, select,
or bootstrap policies, and refuses to render before final benchmarking ends.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'charging_plot_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Reuse the previous figure's interval definitions and read-only data checks.
from plot_b3_study import (
    check_episodes, finite_array, interval_point, load_json, normal_interval,
    require, style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 19, 37)
SEED_COLORS = {7: '#0072B2', 19: '#D55E00', 37: '#009E73'}
NEW_COLOR, OLD_COLOR, RULE_COLOR = '#8B3E8F', '#C67B32', '#64717D'
NEW = 'rl_final_stochastic_mean'
OLD = 'previous_final_stochastic_mean'
RULES = ('random', 'fixed_wait', 'fixed_service', 'fixed_myopic',
         'fixed_inventory', 'adaptive_inventory')
LABELS = {
    'random': 'Random feasible', 'fixed_wait': 'Fixed price + WAIT',
    'fixed_service': 'Fixed price + service first',
    'fixed_myopic': 'Fixed price + myopic',
    'fixed_inventory': 'Fixed price + inventory',
    'adaptive_inventory': 'Adaptive price + inventory',
    OLD: 'Previous RL · three-model mean', NEW: 'New RL · three-model mean',
}
CASES = ('base', 'high_demand', 'low_inventory')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(directory, relative):
    require(isinstance(relative, str), 'Source paths must be relative strings')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,
            f'Invalid relative source path: {relative}')
    resolved = (directory / path).resolve()
    require(resolved.is_relative_to(directory.resolve()),
            f'Source path leaves its provenance directory: {relative}')
    return resolved


def verify_current_sources(hashes):
    for name, expected in hashes.items():
        require(digest(source_path(ROOT, name)) == expected,
                f'Current source differs from frozen provenance: {name}')


def verify_training_sources(study, root_protocol):
    """Prefer the exact executed snapshot over subsequently hardened runners."""
    expected = root_protocol['training_source_sha256']
    snapshot = study / 'provenance' / 'executed_training_source'
    if not snapshot.exists():
        verify_current_sources(expected)
        return
    manifest = load_json(snapshot / 'manifest.json')
    require(manifest['source_count'] == len(expected)
            and set(manifest['sources']) == set(expected),
            'Executed training snapshot has missing or extra frozen sources')
    protocol_hash = digest(study / 'protocol.json')
    require(manifest['study_protocol_sha256'] == protocol_hash,
            'Executed training snapshot belongs to a different frozen protocol')
    protocol_record = manifest['root_protocol_snapshot']
    require(protocol_record['sha256'] == protocol_hash
            and digest(source_path(snapshot, protocol_record['path'])) == protocol_hash,
            'Executed training snapshot root protocol is missing or changed')
    for name, expected_hash in expected.items():
        record = manifest['sources'][name]
        require(record['snapshot_relative_path'] == name
                and record['frozen_protocol_sha256'] == expected_hash
                and record['snapshot_sha256'] == expected_hash,
                f'Training snapshot manifest differs from frozen provenance: {name}')
        path = source_path(snapshot, record['snapshot_relative_path'])
        require(path.stat().st_size == record['size_bytes'] and digest(path) == expected_hash,
                f'Executed training snapshot bytes are missing or changed: {name}')


def curve(run, expected_updates, expected_scenario):
    summary = load_json(run / 'summary.json')
    settings = load_json(run / 'training_config.json')
    scenario = load_json(run / 'scenario.json')
    require(scenario == expected_scenario, f'Scenario mismatch: {run}')
    require(summary['total_updates'] == expected_updates
            and summary['training_physical_periods'] == expected_updates * 16 * scenario['horizon'],
            f'Incomplete or incorrect training budget: {run}')
    require(settings['episodes_per_update'] == 16 and settings['dispatch_final_activation'] == 'silu',
            f'Wrong training preset: {run}')
    require(summary['checkpoint_reload_matches'] and summary['all_metrics_finite']
            and summary['invalid_evaluation_actions'] == 0, f'Failed completion checks: {run}')
    with (run / 'validation_metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = [row for row in csv.DictReader(stream) if row['policy'] == 'stochastic']
    updates = finite_array([float(row['update']) for row in rows], str(run))
    periods = finite_array([float(row['environment_periods']) for row in rows], str(run))
    profits = finite_array([float(row['mean_discounted_profit']) for row in rows], str(run))
    require(updates.tolist() == list(range(0, expected_updates + 1, 25)),
            f'Missing/reordered validation checkpoints: {run}')
    require(np.array_equal(periods, updates * 16 * scenario['horizon']),
            f'Wrong validation period accounting: {run}')
    require(int(summary['best_update']) in updates, f'Missing validation-best point: {run}')
    return dict(updates=updates, periods=periods, profits=profits, summary=summary, settings=settings)


def comparison_check(comparison, expected_mean, expected_count):
    interval = finite_array(comparison['paired_bootstrap_confidence_interval_95'], 'paired CI')
    require(interval.shape == (2,) and interval[0] <= interval[1], 'Invalid paired interval')
    require(comparison['n_episodes'] == expected_count, 'Wrong paired comparison count')
    require(np.isclose(comparison['mean_difference'], expected_mean, rtol=1e-10, atol=1e-10),
            'Paired difference is inconsistent with saved policy means')


def load_study(study):
    benchmark = study / 'benchmarks'
    # This file is written last, after all cases and frozen-source checks pass.
    summary = load_json(benchmark / 'summary.json')
    protocol = load_json(benchmark / 'protocol.json')
    root_protocol = load_json(study / 'protocol.json')
    selection = load_json(study / 'pilot_selection.json')
    require(summary['test_tuning_performed'] is False, 'Test-tuning status must be false')
    require(summary['invalid_actions_all_evaluations'] == 0, 'Invalid benchmark actions recorded')
    scenario = protocol['scenario']
    require(scenario['charging_utility_intercept'] == 3 and scenario['horizon'] == 12,
            'This figure requires the 12-period b=3 scenario')
    require(root_protocol['scenario'] == scenario, 'Root and benchmark scenarios differ')
    require(tuple(protocol['training_seeds']) == SEEDS, 'Expected all seeds 7, 19, 37')
    require(selection['test_scenarios_accessed'] is False, 'Pilot selection accessed test scenes')
    selected = root_protocol['selected_bundle']
    require(selected in ('pure', 'guided') and selected == selection['selected_bundle'],
            'Pilot selection and frozen training bundle differ')
    require(len(protocol['validation_seeds']) == 128 and len(protocol['test_seeds']) == 512,
            'Expected 128 rule-validation and 512 base-test scenes')
    splits = [protocol['validation_seeds'], protocol['test_seeds']]
    for case in CASES[1:]:
        require(len(protocol['stress_seeds'][case]) == 256, 'Expected 256 scenes per stress case')
        splits.append(protocol['stress_seeds'][case])
    require(all(len(values) == len(set(values)) for values in splits), 'Duplicate scene seeds')
    require(all(not set(splits[i]) & set(splits[j]) for i in range(4) for j in range(i)),
            'Rule validation and all three test cases must have disjoint seeds')
    require(protocol['study_protocol_sha256'] == digest(study / 'protocol.json'),
            'Frozen root study protocol no longer matches the benchmark')
    for source_map in (root_protocol['training_source_sha256'], protocol['source_sha256']):
        require(bool(source_map) and all(isinstance(value, str) and len(value) == 64
                                        for value in source_map.values()),
                'Missing frozen source-hash provenance')
    verify_training_sources(study, root_protocol)
    verify_current_sources(protocol['source_sha256'])
    expected_models = 12 if selected == 'guided' else 9
    require(len(protocol['checkpoint_metadata']) == expected_models
            and len(protocol['checkpoint_sha256']) == expected_models,
            'Missing or extra frozen model provenance')
    for seed in SEEDS:
        for variant in ('final', 'previous_final'):
            record = protocol['checkpoint_metadata'][f'{variant}_{seed}']
            require(record['training_updates'] == 500 and record['training_physical_periods'] == 96000
                    and record['training_episodes'] == 8000 and record['training_seed'] == seed,
                    f'Incorrect frozen {variant} budget for seed {seed}')

    pilots, full, demonstration_periods = {}, {}, 0
    for bundle in ('pure', 'guided'):
        pilots[bundle] = curve(study / 'pilots' / bundle, 150, scenario)
        pilot = pilots[bundle]
        score = float(pilot['profits'][np.isin(pilot['updates'], [100, 125, 150])].mean())
        require(np.isclose(score, selection['pilot_scores'][bundle]['selection_score']),
                f'Pilot selection score does not match saved curve: {bundle}')
    expected_selected = ('guided' if selection['pilot_scores']['guided']['selection_score']
                         > selection['pilot_scores']['pure']['selection_score'] + 1. else 'pure')
    require(selected == expected_selected, 'Selected bundle differs from the declared >$1.00 criterion')
    for seed in SEEDS:
        run = study / 'training' / f'seed_{seed}'
        full[seed] = curve(run, 500, scenario)
        settings = full[seed]['settings']
        require(settings == root_protocol['configuration_by_seed'][str(seed)],
                f'Full training settings differ from the frozen protocol: seed {seed}')
        require(len(full[seed]['summary']['eval_seeds']) == 64, 'Expected 64 training-validation scenes')
        require(all(not set(full[seed]['summary']['eval_seeds']) & set(values) for values in splits),
                'Training-validation scenes overlap rule validation or testing')
        if selected == 'guided':
            warmstart = load_json(run / 'warmstart.json')
            require(warmstart['demonstration_episodes'] == root_protocol['warmstart']['episodes']
                    and warmstart['demonstration_periods'] == warmstart['demonstration_episodes'] * scenario['horizon'],
                    f'Inconsistent demonstration budget: seed {seed}')
            demonstration_periods += warmstart['demonstration_periods']

    strongest = summary['strongest_validation_rule']
    require(strongest in RULES and strongest != 'random', 'Invalid validation-selected strongest rule')
    evaluations = {}
    policy_episodes = 0
    for case in CASES:
        result = summary['cases'][case]
        seeds = protocol['test_seeds'] if case == 'base' else protocol['stress_seeds'][case]
        require(result['strongest_validation_rule'] == strongest, 'Rule changes across test cases')
        require(result['invalid_actions_all_evaluations'] == 0, f'Invalid actions in {case}')
        rows = {row['policy']: row for row in result['table']}
        for name in (NEW, OLD):
            aggregate = result['model_averages'][name]
            check_episodes(aggregate, seeds, f'{case}/{name}')
            require(set(aggregate['mean_by_training_seed']) == {str(seed) for seed in SEEDS},
                    f'Missing model seed means: {case}/{name}')
            require(np.isclose(np.mean(list(aggregate['mean_by_training_seed'].values())),
                               rows[name]['mean_discounted_profit']), 'Inconsistent seed mean')
        new = result['model_averages'][NEW]
        old = result['model_averages'][OLD]
        rule_mean = rows[strongest]['mean_discounted_profit']
        comparison_check(result['previous_to_final']['mean'],
                         new['mean_discounted_profit'] - old['mean_discounted_profit'], len(seeds))
        comparison_check(result['comparisons_to_validation_best_rule'][NEW],
                         new['mean_discounted_profit'] - rule_mean, len(seeds))
        for seed in SEEDS:
            new_mean = new['mean_by_training_seed'][str(seed)]
            comparison_check(result['previous_to_final']['per_training_seed'][str(seed)],
                             new_mean - old['mean_by_training_seed'][str(seed)], len(seeds))
            comparison_check(result['comparisons_to_validation_best_rule'][f'rl_final_stochastic_{seed}'],
                             new_mean - rule_mean, len(seeds))
        expected_policy_count = 6 + 3 * (5 if selected == 'guided' else 4) if case == 'base' else 15
        require(result['test_policy_episodes'] == len(seeds) * expected_policy_count,
                f'Wrong actual policy-episode count in {case}')
        policy_episodes += result['test_policy_episodes']
        if case == 'base':
            for name in (*RULES, OLD, NEW):
                evaluation = (result['model_averages'][name] if name in (OLD, NEW)
                              else load_json(benchmark / 'base' / f'{name}.json'))
                check_episodes(evaluation, seeds, name)
                require(np.isclose(evaluation['mean_discounted_profit'], rows[name]['mean_discounted_profit']),
                        f'Base table differs from episode evidence: {name}')
                evaluations[name] = evaluation
    require(policy_episodes == summary['test_policy_episodes'], 'Wrong total evaluated policy-episode count')
    return summary, protocol, root_protocol, selection, pilots, full, evaluations, demonstration_periods


def paired_panel(ax, letter, title, comparisons):
    style_axis(ax, letter, title)
    for y, seed in enumerate(SEEDS):
        row = comparisons['per_training_seed'][str(seed)]
        interval_point(ax, row['mean_difference'], row['paired_bootstrap_confidence_interval_95'],
                       y, SEED_COLORS[seed])
    mean = comparisons['mean']
    interval_point(ax, mean['mean_difference'], mean['paired_bootstrap_confidence_interval_95'],
                   3, NEW_COLOR, marker='D', size=7)
    ax.axvline(0, color='#455A6C', linestyle='--', linewidth=1.)
    ax.set_yticks(range(4), [f'Seed {seed}' for seed in SEEDS] + ['Three-model mean'])
    ax.set_ylim(3.5, -.5)
    ax.grid(axis='x')
    ax.set_xlabel('Paired difference in discounted profit ($ / episode)')


def render(study, output, data, dpi):
    summary, protocol, root_protocol, selection, pilots, full, evaluations, demo_periods = data
    selected = selection['selected_bundle']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': '#243444', 'axes.labelcolor': '#243444',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'grid.color': '#E5EAF0', 'grid.linewidth': .7})
    fig, axes = plt.subplots(3, 2, figsize=(17.6, 16.8))
    fig.subplots_adjust(left=.115, right=.975, top=.865, bottom=.14, hspace=.57, wspace=.43)
    fig.suptitle('SA-PPO–MAPPO · targeted training improvements at b = 3', x=.115, y=.973,
                 ha='left', fontsize=22, fontweight='bold')
    bundle = 'imitation-assisted RL' if selected == 'guided' else 'pure RL'
    fig.text(.115, .942, f'Selected bundle: {bundle} · training seeds 7, 19, 37 · '
             '500 PPO updates / 96,000 physical periods per model', fontsize=11, color='#526170')
    budget = (f' + {demo_periods // 3:,} teacher-demonstration periods per new model'
              if selected == 'guided' else '; no teacher demonstrations in final models')
    fig.text(.115, .918, 'Final-training budget: 288,000 PPO periods across three models' + budget,
             fontsize=10.5, color='#526170')
    fig.text(.115, .895, 'Independent tests: 512 base + 256 high-demand + 256 low-inventory scenes; '
             '128 separate scenes select rules', fontsize=10.5, color='#526170')

    ax = axes[0, 0]
    style_axis(ax, 'A', 'Development pilots · one seed, validation only')
    colors = {'pure': '#536E8A', 'guided': '#8B3E8F'}
    for name in ('pure', 'guided'):
        data = pilots[name]
        ax.plot(data['updates'], data['profits'], color=colors[name], marker='o', linewidth=1.8,
                markersize=4, label='Pure RL' if name == 'pure' else 'Imitation-assisted RL')
    ax.axvspan(100, 150, color='#D8E1EA', alpha=.32, zorder=0)
    ax.set_xlabel('PPO updates (192 physical periods per update)')
    ax.set_ylabel('Validation discounted profit ($ / episode)')
    ax.set_xlim(-2, 152)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.grid(True)
    ax.legend(loc='center right', bbox_to_anchor=(.98, .55), frameon=True,
              facecolor='white', edgecolor='#D2DBE3', framealpha=1., fontsize=9)
    scores = selection['pilot_scores']
    ax.text(.97, .06,
            f'Mean at updates 100 / 125 / 150\n'
            f'Pure: {scores["pure"]["selection_score"]:.2f} USD  |  '
            f'Guided: {scores["guided"]["selection_score"]:.2f} USD\n'
            f'Selected: {selected}; guided requires > 1.00 USD gain',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.6,
            bbox=dict(facecolor='white', edgecolor='#D2DBE3', alpha=.96, boxstyle='round,pad=.45'))

    ax = axes[0, 1]
    style_axis(ax, 'B', 'Selected bundle · full stochastic validation curves')
    for seed in SEEDS:
        data = full[seed]
        ax.plot(data['periods'] / 1000, data['profits'], color=SEED_COLORS[seed],
                linewidth=1.6, marker='o', markersize=2.8, label=f'Seed {seed}')
        best = np.flatnonzero(data['updates'] == int(data['summary']['best_update']))[0]
        ax.scatter(data['periods'][best] / 1000, data['profits'][best], marker='*', s=105,
                   color=SEED_COLORS[seed], edgecolor='white', linewidth=.65, zorder=5)
    ax.set_xlim(-1, 97)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_xlabel('PPO physical periods (thousands)')
    ax.set_ylabel('Validation discounted profit ($ / episode)')
    ax.grid(True)
    handles = [Line2D([], [], color=SEED_COLORS[seed], linewidth=1.7, label=f'Seed {seed}') for seed in SEEDS]
    handles.append(Line2D([], [], color='#526170', marker='*', linestyle='None', markersize=9,
                          label='Validation best'))
    ax.legend(handles=handles, frameon=False, loc='lower right', ncol=2, fontsize=8.5)
    ax.text(0, -.24, '64 fixed validation scenes per point; unsmoothed. '
            + ('Period 0 is after imitation.' if selected == 'guided' else 'Period 0 is random initialization.'),
            transform=ax.transAxes, fontsize=8.2, color='#526170')

    ax = axes[1, 0]
    style_axis(ax, 'C', 'Base test · conditional 95% mean intervals')
    policies = (*RULES, OLD, NEW)
    strongest = summary['strongest_validation_rule']
    for y, name in enumerate(policies):
        result = evaluations[name]
        profits = np.asarray([row['discounted_profit'] for row in result['episodes']])
        color = OLD_COLOR if name == OLD else NEW_COLOR if name == NEW else RULE_COLOR
        interval_point(ax, result['mean_discounted_profit'], normal_interval(profits), y, color,
                       marker='D' if name in (OLD, NEW) else 'o')
        if name in (OLD, NEW):
            for seed in SEEDS:
                ax.plot(result['mean_by_training_seed'][str(seed)], y + .2, marker='|',
                        linestyle='None', markersize=8, color=SEED_COLORS[seed])
    ax.set_yticks(range(len(policies)), [LABELS[name] + (' *' if name == strongest else '') for name in policies])
    ax.set_ylim(len(policies) - .4, -.5)
    ax.axvline(0, color='#AAB4BE', linewidth=.8)
    ax.grid(axis='x')
    ax.set_xlabel('Mean discounted profit ($ / episode)')
    ax.text(0, -.21, '* Selected on rule validation; colored ticks show individual training seeds.',
            transform=ax.transAxes, fontsize=8.2, color='#526170')

    base = summary['cases']['base']
    paired_panel(axes[1, 1], 'D', 'Base paired 95% intervals · new − previous RL',
                 base['previous_to_final'])
    rule_comparison = {
        'per_training_seed': {str(seed): base['comparisons_to_validation_best_rule'][f'rl_final_stochastic_{seed}']
                              for seed in SEEDS},
        'mean': base['comparisons_to_validation_best_rule'][NEW],
    }
    paired_panel(axes[2, 0], 'E', 'Base paired 95% intervals · new RL − selected rule', rule_comparison)

    ax = axes[2, 1]
    style_axis(ax, 'F', 'Frozen stochastic policies · all three test cases')
    x, width = np.arange(3), .24
    for i, (name, color, label) in enumerate(((strongest, RULE_COLOR, 'Selected rule'),
                                             (OLD, OLD_COLOR, 'Previous RL mean'),
                                             (NEW, NEW_COLOR, 'New RL mean'))):
        means = []
        for case in CASES:
            rows = {row['policy']: row for row in summary['cases'][case]['table']}
            means.append(rows[name]['mean_discounted_profit'])
        positions = x + (i - 1) * width
        ax.bar(positions, means, width=width, color=color, label=label,
               edgecolor='white', linewidth=.6, zorder=3)
        if name in (OLD, NEW):
            for seed_index, seed in enumerate(SEEDS):
                points = [summary['cases'][case]['model_averages'][name]['mean_by_training_seed'][str(seed)]
                          for case in CASES]
                ax.scatter(positions + (seed_index - 1) * .048, points,
                           color=SEED_COLORS[seed], edgecolor='white', linewidth=.55, s=26, zorder=5)
    ax.axhline(0, color='#AAB4BE', linewidth=.8)
    ax.set_xticks(x, ['Base', 'Demand ×1.5', 'Initial energy\n35 kWh / vehicle'])
    ax.set_ylabel('Mean discounted profit ($ / episode)')
    ax.grid(axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(-.015, -.23), ncol=3,
              frameon=False, fontsize=8.1, handlelength=1.2, columnspacing=1.1)

    footer = (
        'A: pilot selection uses the shaded three-checkpoint window; guided starts after behavior cloning. '
        'B: validation-best stars are descriptive; primary tests use all three final models.\n'
        'C: normal mean intervals. D/E: saved paired-bootstrap intervals; values above zero favor the new policy. '
        'All intervals condition on fixed models, not independent retraining.\n'
        'RL means average separately executed models; no ensemble policy is evaluated. Colored ticks/dots '
        'show seed variation. F omits intervals; cross-case differences are descriptive.\n'
        f'All profit is the original environment reward. Selected rule: {LABELS[strongest]}. '
        'Rules and models are frozen before testing; no test-driven selection or claim of global optimality.'
    )
    fig.text(.115, .032, footer, ha='left', va='bottom', fontsize=8.5, color='#526170', linespacing=1.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp.png')
    metadata = {
        'Title': 'SA-PPO-MAPPO targeted b=3 training improvements',
        'Description': 'Pilot provenance; final models; conditional sampling intervals, not retraining uncertainty.',
        'Source': str(study), 'Selected bundle': selected,
        'Benchmark summary SHA256': digest(study / 'benchmarks' / 'summary.json'),
        'Benchmark protocol SHA256': digest(study / 'benchmarks' / 'protocol.json'),
        'Pilot selection SHA256': digest(study / 'pilot_selection.json'),
        'Frozen model count': str(len(protocol['checkpoint_sha256'])),
        'Training source count': str(len(root_protocol['training_source_sha256'])),
        'Benchmark source count': str(len(protocol['source_sha256'])),
    }
    try:
        fig.savefig(temporary, format='png', dpi=dpi, bbox_inches='tight', pad_inches=.24, metadata=metadata)
        temporary.replace(output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs' / 'b3_improved_study')
    parser.add_argument('--output', type=Path, help='PNG output; default: STUDY_DIR/b3_improved_study.png')
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    output = (args.output or study / 'b3_improved_study.png').resolve()
    if output.suffix.lower() != '.png' or args.dpi < 72:
        parser.error('Use PNG output and at least 72 DPI')
    try:
        data = load_study(study)
        render(study, output, data, args.dpi)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f'Complete and verify the study before plotting: {error}')
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
