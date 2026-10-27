"""Render one PNG from the completed SiLU, b=3, three-seed study.

This is a read-only evidence consumer: it never trains or evaluates a policy,
selects a checkpoint, or tunes a rule. Matplotlib is an optional dependency.
Run only after training and the independent benchmark both finish:

    python experiments/archive/plot_b3_study.py
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
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (7, 19, 37)
UPDATES = 500
PERIODS = 96000
SEED_COLORS = {7: '#0072B2', 19: '#D55E00', 37: '#009E73'}
RULE_COLOR = '#606B77'
MEAN_COLOR = '#8B3E8F'
RULES = ('random', 'fixed_wait', 'fixed_service', 'fixed_myopic',
         'fixed_inventory', 'adaptive_inventory')
PRIMARY = 'rl_final_stochastic_mean'
SECONDARY = 'rl_final_deterministic_mean'
LABELS = {
    'random': 'Random feasible',
    'fixed_wait': 'Fixed price + WAIT',
    'fixed_service': 'Fixed price + service first',
    'fixed_myopic': 'Fixed price + myopic',
    'fixed_inventory': 'Fixed price + inventory',
    'adaptive_inventory': 'Adaptive price + inventory',
    PRIMARY: 'Final RL · stochastic',
    SECONDARY: 'Final RL · deterministic',
}
CASE_LABELS = ('Base', 'Demand ×1.5', 'Initial energy\n35 kWh / vehicle')
CASES = ('base', 'high_demand', 'low_inventory')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_array(values, label):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), f'Empty/nonfinite {label}')
    return values


def check_episodes(evaluation, expected_seeds, label):
    episodes = evaluation['episodes']
    seeds = [int(row['seed']) for row in episodes]
    require(len(seeds) == len(set(seeds)), f'Duplicate evaluation seeds in {label}')
    require(set(seeds) == set(expected_seeds), f'Wrong evaluation split for {label}')
    profits = finite_array([row['discounted_profit'] for row in episodes], label)
    require(np.isclose(profits.mean(), evaluation['mean_discounted_profit'],
                       rtol=1e-10, atol=1e-10), f'Inconsistent mean in {label}')
    require(evaluation.get('invalid_actions', 0) == 0, f'Invalid actions in {label}')
    return profits


def normal_interval(values):
    """Conditional mean interval, using one paired model-average per scene.

    For the primary aggregate, each value is the average of three separately
    executed models on the same scene. Neither pooled model-episodes nor the
    three seed means are treated as independent scene observations.
    """
    require(len(values) >= 2, 'At least two independent scenes are required')
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - half_width, mean + half_width


def load_study(study):
    benchmark = study / 'benchmarks'
    # The top-level summary is written only after all benchmark cases finish.
    summary = load_json(benchmark / 'summary.json')
    protocol = load_json(benchmark / 'protocol.json')
    require(summary.get('test_tuning_performed') is False, 'Test-tuning status must be false')
    require(protocol['scenario'].get('charging_utility_intercept') == 3,
            'This plot is specific to the b=3 customer model')
    require(tuple(protocol['training_seeds']) == SEEDS, 'Expected training seeds 7, 19, 37')
    require(len(protocol['test_seeds']) == 512 and len(protocol['stress_seeds']) == 256,
            'This predeclared study requires 512 base and 256 stress scenes')
    split_sets = [set(protocol[key]) for key in ('validation_seeds', 'test_seeds', 'stress_seeds')]
    require(all(not split_sets[i] & split_sets[j] for i in range(3) for j in range(i)),
            'Benchmark validation, base test and stress splits overlap')

    curves = {}
    for seed in SEEDS:
        run = study / 'training' / f'seed_{seed}'
        training_summary = load_json(run / 'summary.json')
        config = load_json(run / 'training_config.json')
        scenario = load_json(run / 'scenario.json')
        require(training_summary['total_updates'] == UPDATES
                and training_summary['training_physical_periods'] == PERIODS,
                f'Seed {seed} has not completed exactly 500 updates / 96,000 periods')
        require(config.get('dispatch_final_activation') == 'silu'
                and scenario.get('charging_utility_intercept') == 3,
                f'Seed {seed} is not the SiLU, b=3 run')
        require(training_summary['all_metrics_finite']
                and training_summary['checkpoint_reload_matches']
                and training_summary['invalid_evaluation_actions'] == 0,
                f'Seed {seed} failed a saved completion check')
        require(not set(training_summary['eval_seeds']) & (split_sets[1] | split_sets[2]),
                f'Training-validation scenes overlap held-out tests for seed {seed}')
        with (run / 'validation_metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if row['policy'] == 'stochastic']
        updates = finite_array([float(row['update']) for row in rows], f'seed {seed} updates')
        periods = finite_array([float(row['environment_periods']) for row in rows],
                               f'seed {seed} periods')
        profits = finite_array([float(row['mean_discounted_profit']) for row in rows],
                               f'seed {seed} validation profits')
        expected_updates = list(range(0, UPDATES + 1, int(config['eval_interval'])))
        if expected_updates[-1] != UPDATES:
            expected_updates.append(UPDATES)
        require(updates.tolist() == expected_updates,
                f'Seed {seed} is missing or reorders validation checkpoints')
        require(periods[0] == 0 and periods[-1] == PERIODS and np.all(np.diff(periods) > 0),
                f'Seed {seed} has incomplete validation-period coverage')
        curves[seed] = {'periods': periods, 'profits': profits,
                        'validation_episodes': len(training_summary['eval_seeds'])}

    strongest = summary['strongest_validation_rule']
    require(strongest in RULES and strongest != 'random', 'Unknown strongest validation rule')
    base = summary['cases']['base']
    evaluations = {}
    for name in (*RULES, PRIMARY, SECONDARY):
        evaluation = (base['model_averages'][name] if name.startswith('rl_')
                      else load_json(benchmark / 'base' / f'{name}.json'))
        check_episodes(evaluation, protocol['test_seeds'], name)
        evaluations[name] = evaluation

    for case in CASES:
        result = summary['cases'][case]
        require(result['strongest_validation_rule'] == strongest,
                f'Rule was changed for case {case}')
        require(result['invalid_actions_all_evaluations'] == 0, f'Invalid actions in {case}')
        expected_seeds = protocol['test_seeds' if case == 'base' else 'stress_seeds']
        aggregate = result['model_averages'][PRIMARY]
        check_episodes(aggregate, expected_seeds, f'{case}/{PRIMARY}')
        require(set(aggregate['mean_by_training_seed']) == {str(seed) for seed in SEEDS},
                f'Missing individual training-seed means for {case}')
        seed_means = finite_array(list(aggregate['mean_by_training_seed'].values()),
                                 f'{case} individual means')
        require(np.isclose(seed_means.mean(), aggregate['mean_discounted_profit']),
                f'Incorrect three-model mean for {case}')
        rows = {row['policy']: row for row in result['table']}
        require(np.isfinite(rows[strongest]['mean_discounted_profit']),
                f'Nonfinite rule profit in {case}')

    # Use the saved paired-bootstrap analysis; the plotter does not resample.
    comparisons = base['comparisons_to_validation_best_rule']
    for name in [f'rl_final_stochastic_{seed}' for seed in SEEDS] + [PRIMARY]:
        comparison = comparisons[name]
        interval = finite_array(comparison['paired_bootstrap_confidence_interval_95'],
                                f'{name} paired interval')
        require(interval.shape == (2,) and interval[0] <= interval[1],
                f'Invalid paired interval for {name}')
        require(comparison['n_episodes'] == len(protocol['test_seeds'])
                and np.isfinite(comparison['mean_difference']),
                f'Incomplete paired comparison for {name}')
    return summary, protocol, curves, evaluations


def style_axis(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', fontsize=12, fontweight='bold', pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#9CA8B2')
    ax.tick_params(labelsize=9, colors='#3F4C58')
    ax.set_axisbelow(True)


def interval_point(ax, mean, interval, y, color, marker='o', size=6):
    # Drawing segments directly also handles a percentile CI not enclosing its estimate.
    ax.hlines(y, interval[0], interval[1], color=color, linewidth=1.8, zorder=3)
    ax.vlines(interval, y - .075, y + .075, color=color, linewidth=1.4, zorder=3)
    ax.plot(mean, y, marker=marker, markersize=size, color=color, linestyle='None', zorder=4)


def render(study, output, summary, protocol, curves, evaluations, dpi):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'text.color': '#243444', 'axes.labelcolor': '#243444',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'grid.color': '#E5EAF0', 'grid.linewidth': .7,
    })
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 12.2))
    fig.subplots_adjust(left=.075, right=.97, bottom=.18, top=.855,
                        wspace=.43, hspace=.48)
    fig.suptitle('SA-PPO–MAPPO · SiLU dispatch activation + charging utility b = 3',
                 x=.075, y=.968, ha='left', fontsize=21, fontweight='bold')
    fig.text(.075, .931,
             'Training seeds 7, 19, 37 · 500 updates / 96,000 physical periods per seed',
             fontsize=11, color='#526170')
    fig.text(.075, .904,
             f'Rules selected on {len(protocol["validation_seeds"])} validation scenes · '
             'Independent tests: N = 512 base; N = 256 per stress case',
             fontsize=10.5, color='#526170')

    # A: every saved point including initialization, with no smoothing.
    ax = axes[0, 0]
    style_axis(ax, 'A', 'Stochastic validation learning curves')
    for seed in SEEDS:
        ax.plot(curves[seed]['periods'] / 1000, curves[seed]['profits'],
                color=SEED_COLORS[seed], marker='o', markersize=3.1,
                linewidth=1.7, label=f'Seed {seed}')
    ax.axhline(0, color='#AAB4BE', linewidth=.8)
    ax.set_xlim(0, PERIODS / 1000)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_xlabel('Training physical periods (thousands)')
    ax.set_ylabel('Mean discounted profit ($ / episode)')
    ax.grid(True)
    ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=8.5,
              handlelength=1.6, columnspacing=1.2)
    inset = ax.inset_axes([.47, .12, .49, .43])
    for seed in SEEDS:
        curve = curves[seed]
        late = curve['periods'] >= PERIODS / 2
        inset.plot(curve['periods'][late] / 1000, curve['profits'][late],
                   color=SEED_COLORS[seed], marker='o', markersize=2.1, linewidth=1.25)
    inset.set_title('48k–96k periods · expanded y scale', fontsize=7.6, pad=4)
    inset.set_xlim(48, 96)
    inset.set_xticks([48, 72, 96])
    inset.tick_params(labelsize=7)
    inset.grid(True, linewidth=.5)
    inset.spines[['top', 'right']].set_visible(False)

    # B: all six selected rule families plus the primary and secondary RL modes.
    ax = axes[0, 1]
    style_axis(ax, 'B', 'Base test profit · conditional 95% intervals')
    policies = (*RULES, PRIMARY, SECONDARY)
    strongest = summary['strongest_validation_rule']
    for y, name in enumerate(policies):
        evaluation = evaluations[name]
        profits = np.asarray([row['discounted_profit'] for row in evaluation['episodes']])
        color = MEAN_COLOR if name == PRIMARY else ('#B86B38' if name == SECONDARY else RULE_COLOR)
        interval_point(ax, evaluation['mean_discounted_profit'], normal_interval(profits), y,
                       color, marker='D' if name == PRIMARY else 'o')
        if name.startswith('rl_'):
            for seed in SEEDS:
                ax.plot(evaluation['mean_by_training_seed'][str(seed)], y + .19,
                        marker='|', color=SEED_COLORS[seed], markersize=8,
                        linestyle='None', zorder=4)
    ax.set_yticks(range(len(policies)),
                  [LABELS[name] + (' *' if name == strongest else '') for name in policies])
    ax.set_ylim(len(policies) - .45, -.55)
    ax.axvline(0, color='#AAB4BE', linewidth=.8)
    ax.grid(axis='x')
    ax.set_xlabel('Mean discounted profit ($ / episode)')
    ax.text(0, -.21, '* Strongest rule selected on validation; colored ticks = individual RL seeds.',
            transform=ax.transAxes, fontsize=8.2, color='#526170')

    # C: three independently trained models, not a CI based on only three seeds.
    ax = axes[1, 0]
    style_axis(ax, 'C', 'Final stochastic RL − selected rule · base')
    comparison_names = [f'rl_final_stochastic_{seed}' for seed in SEEDS] + [PRIMARY]
    for y, name in enumerate(comparison_names):
        result = summary['cases']['base']['comparisons_to_validation_best_rule'][name]
        interval_point(ax, result['mean_difference'],
                       result['paired_bootstrap_confidence_interval_95'], y,
                       SEED_COLORS[SEEDS[y]] if y < 3 else MEAN_COLOR,
                       marker='o' if y < 3 else 'D', size=6.5)
    ax.axvline(0, color='#455A6C', linewidth=1.1, linestyle='--')
    ax.set_yticks(range(4), [f'Seed {seed}' for seed in SEEDS] + ['Mean of three models'])
    ax.set_ylim(3.5, -.5)
    ax.grid(axis='x')
    ax.set_xlabel('Paired difference in discounted profit ($ / episode)')
    ax.text(0, -.20, 'Saved 95% paired-bootstrap intervals; values above zero favor RL.',
            transform=ax.transAxes, fontsize=8.2, color='#526170')

    # D: frozen policies in all three scenarios; no stress-specific reselection.
    ax = axes[1, 1]
    style_axis(ax, 'D', 'Frozen final stochastic policies across scenarios')
    width = .145
    x = np.arange(len(CASES))
    series = [(strongest, RULE_COLOR, 'Selected rule')]
    series.extend((str(seed), SEED_COLORS[seed], f'RL seed {seed}') for seed in SEEDS)
    series.append((PRIMARY, MEAN_COLOR, 'RL three-model mean'))
    for index, (name, color, label) in enumerate(series):
        means = []
        for case in CASES:
            result = summary['cases'][case]
            rows = {row['policy']: row for row in result['table']}
            aggregate = result['model_averages'][PRIMARY]
            means.append(rows[strongest]['mean_discounted_profit'] if name == strongest else
                         aggregate['mean_discounted_profit'] if name == PRIMARY else
                         aggregate['mean_by_training_seed'][name])
        ax.bar(x + (index - 2) * width, means, width=width, color=color,
               label=label, edgecolor='white', linewidth=.5, zorder=3)
    ax.axhline(0, color='#AAB4BE', linewidth=.8)
    ax.grid(axis='y')
    ax.set_xticks(x, CASE_LABELS)
    ax.set_ylabel('Mean discounted profit ($ / episode)')
    ax.legend(loc='upper left', bbox_to_anchor=(-.01, -.20), frameon=False,
              ncol=3, fontsize=8.1, columnspacing=1.1, handlelength=1.25)

    validation_counts = sorted({curves[seed]['validation_episodes'] for seed in SEEDS})
    count_label = str(validation_counts[0]) if len(validation_counts) == 1 else '/'.join(map(str, validation_counts))
    footer = (
        f'A: {count_label} fixed validation scenes per checkpoint; unsmoothed points include initialization. '
        'Validation monitors/selects policies and is separate from testing.\n'
        'B: normal mean intervals. C: paired-bootstrap difference intervals. Both condition on the frozen '
        'policies and reflect scene/action sampling, not retraining uncertainty.\n'
        'RL means average three separately executed models; no ensemble policy was executed. '
        'Individual seed points/bars expose training variation. D omits intervals.\n'
        f'Selected rule: {LABELS[strongest]}. Rules are tuned only on validation, then frozen for all tests; '
        'no claim of global optimality or calibrated customer preferences.'
    )
    fig.text(.075, .037, footer, ha='left', va='bottom', fontsize=8.5,
             color='#526170', linespacing=1.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp.png')
    metadata = {
        'Title': 'SA-PPO-MAPPO SiLU b=3 completed study',
        'Description': 'Conditional sampling intervals; individual training-seed results; no ensemble.',
        'Source': str(study),
        'Benchmark summary SHA256': hashlib.sha256((study / 'benchmarks' / 'summary.json').read_bytes()).hexdigest(),
        'Benchmark protocol SHA256': hashlib.sha256((study / 'benchmarks' / 'protocol.json').read_bytes()).hexdigest(),
    }
    try:
        fig.savefig(temporary, format='png', dpi=dpi, bbox_inches='tight',
                    pad_inches=.22, metadata=metadata)
        temporary.replace(output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs' / 'silu_b3_study')
    parser.add_argument('--output', type=Path,
                        help='PNG path; default: STUDY_DIR/silu_b3_study.png')
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    output = (args.output or study / 'silu_b3_study.png').resolve()
    if output.suffix.lower() != '.png':
        parser.error('Only PNG output is supported')
    if args.dpi < 72:
        parser.error('Use at least 72 DPI')
    try:
        summary, protocol, curves, evaluations = load_study(study)
        render(study, output, summary, protocol, curves, evaluations, args.dpi)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f'Complete and verify the study before plotting: {error}')
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
