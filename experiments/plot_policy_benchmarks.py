"""Plot frozen-policy benchmark evidence after run_policy_benchmarks completes.

Run: python experiments/plot_policy_benchmarks.py
Optional plotting dependency: matplotlib. No policy fitting or selection occurs.
"""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    'random', 'fixed_wait', 'fixed_service', 'fixed_myopic', 'fixed_inventory',
    'adaptive_inventory', 'rl_final_stochastic_mean', 'rl_final_deterministic_mean',
)
LABELS = {
    'random': 'Random feasible',
    'fixed_wait': 'Fixed price + WAIT',
    'fixed_service': 'Fixed price + service first',
    'fixed_myopic': 'Fixed price + myopic',
    'fixed_inventory': 'Fixed price + inventory',
    'adaptive_inventory': 'Adaptive price + inventory',
    'rl_final_stochastic_mean': 'SA-PPO–MAPPO, stochastic',
    'rl_final_deterministic_mean': 'SA-PPO–MAPPO, deterministic',
}
SHORT_LABELS = {
    'random': 'Random', 'fixed_wait': 'WAIT', 'fixed_service': 'Service first',
    'fixed_myopic': 'Myopic', 'fixed_inventory': 'Inventory',
    'adaptive_inventory': 'Adaptive',
    'rl_final_stochastic_mean': 'RL stochastic',
    'rl_final_deterministic_mean': 'RL deterministic',
}
COLORS = {
    'random': '#9A9FA9', 'fixed_wait': '#6D7787', 'fixed_service': '#4D8DB3',
    'fixed_myopic': '#237F85', 'fixed_inventory': '#166C4E',
    'adaptive_inventory': '#7960A5', 'rl_final_stochastic_mean': '#D97819',
    'rl_final_deterministic_mean': '#B7443D',
}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def conditional_interval(evaluation):
    """Normal mean interval across scenes; fixed-model RL means stay paired.

    An RL aggregate contains one average return per scene, averaged over the
    three separately executed models. Its interval conditions on this fixed trio
    and is not an interval for a randomly retrained model or an ensemble policy.
    """
    values = np.asarray([r['discounted_profit'] for r in evaluation['episodes']], dtype=float)
    if len(values) < 2:
        return None
    mean = float(values.mean())
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean - margin, mean + margin


def style_axis(ax, panel, title):
    ax.set_title(f'{panel}  {title}', loc='left', fontsize=11, fontweight='bold', pad=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#B9C3CC')
    ax.tick_params(colors='#3E4A58', labelsize=9)
    ax.set_axisbelow(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/policy_benchmarks')
    parser.add_argument('--docs-copy', type=Path, default=ROOT/'docs/assets/policy_benchmarks.png')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    path = output/'summary.json'
    if not path.is_file():
        parser.error('Complete run_policy_benchmarks.py before plotting; summary.json is absent')
    summary, protocol = load_json(path), load_json(output/'protocol.json')
    base = summary['cases']['base']
    rows = {row['policy']: row for row in base['table']}
    strongest = summary['strongest_validation_rule']
    evaluations = {name: (base['model_averages'][name] if name.startswith('rl_')
                           else load_json(output/'base'/f'{name}.json')) for name in POLICIES}
    n_base, n_stress = len(protocol['test_seeds']), len(protocol['stress_seeds'])
    n_validation = len(protocol['validation_seeds'])
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelcolor': '#243444', 'text.color': '#243444',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.8),
                             gridspec_kw={'width_ratios': [1.08, 1.]})
    fig.subplots_adjust(left=.215, right=.97, top=.87, bottom=.195, hspace=.47, wspace=.34)
    fig.suptitle('Frozen policy benchmarks', x=.215, y=.965, ha='left',
                 fontsize=23, fontweight='bold')
    fig.text(.215, .926,
             f'Rules tuned on {n_validation} validation scenarios; fresh common test scenarios '
             f'({n_base} base, {n_stress} per stress case)', fontsize=11, color='#5A6775')

    # A: show test mean and conditional sampling uncertainty for every policy.
    ax = axes[0, 0]
    style_axis(ax, 'A', 'Base profit · conditional 95% intervals')
    y = np.arange(len(POLICIES))
    for index, name in enumerate(POLICIES):
        mean = rows[name]['mean_discounted_profit']
        interval = conditional_interval(evaluations[name])
        error = None if interval is None else [[mean-interval[0]], [interval[1]-mean]]
        ax.errorbar(mean, index, xerr=error, fmt='o', color=COLORS[name],
                    markersize=6.5 if name.startswith('rl_') else 5.5,
                    elinewidth=1.8, capsize=3, zorder=3)
        if name.startswith('rl_'):
            means = list(evaluations[name]['mean_by_training_seed'].values())
            ax.scatter(means, [index+.18]*len(means), marker='|', s=40,
                       color=COLORS[name], alpha=.75, zorder=4)
    ax.set_yticks(y, [LABELS[name] for name in POLICIES])
    ax.invert_yaxis()
    ax.set_ylim(len(POLICIES)-.45, -.55)
    ax.axvline(0, color='#AAB6BF', lw=.9)
    ax.grid(axis='x', color='#E6EBEF')
    ax.set_xlabel('Discounted profit per episode ($)')
    ax.legend(handles=[Line2D([], [], marker='|', color='#6A737C', linestyle='None',
                              markersize=8, label='Individual training-seed mean (RL)')],
              frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0., -.27))

    # B: paired differences are the appropriate head-to-head statistical test.
    ax = axes[0, 1]
    style_axis(ax, 'B', f'Paired difference vs {SHORT_LABELS[strongest]}')
    comparisons = base['comparisons_to_validation_best_rule']
    for index, name in enumerate(POLICIES):
        if name == strongest:
            ax.plot(0, index, marker='D', markersize=5, color=COLORS[name], zorder=4)
            continue
        comparison = comparisons[name]
        mean = comparison['mean_difference']
        interval = comparison['paired_bootstrap_confidence_interval_95']
        error = None if interval is None else [[mean-interval[0]], [interval[1]-mean]]
        ax.errorbar(mean, index, xerr=error, fmt='o', color=COLORS[name],
                    markersize=6 if name.startswith('rl_') else 5,
                    elinewidth=1.8, capsize=3, zorder=3)
    ax.axvline(0, color='#384F62', lw=1.1, ls='--')
    ax.set_yticks(y, [SHORT_LABELS[name] for name in POLICIES])
    ax.invert_yaxis()
    ax.set_ylim(len(POLICIES)-.45, -.55)
    ax.grid(axis='x', color='#E6EBEF')
    ax.set_xlabel('Policy − validation-selected rule ($ / episode)')
    ax.text(0., -.25, '95% paired bootstrap intervals · below zero = lower profit',
            transform=ax.transAxes, fontsize=8.5, color='#5A6775')

    # C: reward and actual service coverage are distinct objectives.
    ax = axes[1, 0]
    style_axis(ax, 'C', 'Profit and service coverage')
    annotations = {
        'random': ((6, 2), 'left'), 'fixed_wait': ((6, -14), 'left'),
        'fixed_service': ((8, -13), 'left'),
        'fixed_myopic': ((-8, -13), 'right'),
        'fixed_inventory': ((-7, 12), 'right'),
        'adaptive_inventory': ((8, 10), 'left'),
        'rl_final_stochastic_mean': ((5, -22), 'left'),
        'rl_final_deterministic_mean': ((-9, 12), 'right'),
    }
    for name in POLICIES:
        x = 100 * rows[name]['mean_served_request_share']
        profit = rows[name]['mean_discounted_profit']
        ax.scatter(x, profit, s=64 if name.startswith('rl_') else 44,
                   marker='s' if name.startswith('rl_') else 'o',
                   color=COLORS[name], edgecolor='white', linewidth=.8, zorder=4)
        offset, alignment = annotations[name]
        ax.annotate(SHORT_LABELS[name], (x, profit), xytext=offset,
                    textcoords='offset points', fontsize=8, ha=alignment, color=COLORS[name])
    ax.axhline(0, color='#AAB6BF', lw=.9)
    ax.grid(color='#E6EBEF')
    coverages = [100 * rows[name]['mean_served_request_share'] for name in POLICIES]
    ax.set_xlim(max(0., min(coverages)-.85), max(coverages)+1.85)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin-2, ymax+4)
    ax.set_xlabel('Served requests / all arrivals (%)')
    ax.set_ylabel('Discounted profit per episode ($)')

    # D: policies and validation-tuned settings are frozen in all stress cases.
    ax = axes[1, 1]
    style_axis(ax, 'D', 'Frozen-policy stress tests')
    cases = ('base', 'high_demand', 'low_inventory')
    case_colors = ('#617B94', '#E8A152', '#73AC93')
    case_labels = ('Base', 'Demand ×1.5', 'Initial energy 35 kWh / MCV')
    width = .23
    for case_index, case in enumerate(cases):
        case_rows = {row['policy']: row for row in summary['cases'][case]['table']}
        values = [case_rows[name]['mean_discounted_profit'] for name in POLICIES]
        ax.bar(y+(case_index-1)*width, values, width=width, color=case_colors[case_index],
               label=case_labels[case_index], linewidth=0, zorder=3)
    ax.axhline(0, color='#AAB6BF', lw=.9)
    ax.grid(axis='y', color='#E6EBEF')
    ax.set_xticks(y, [SHORT_LABELS[name] for name in POLICIES], rotation=42, ha='right')
    ax.set_ylabel('Discounted profit per episode ($)')
    ax.legend(loc='lower right', frameon=True, framealpha=.94,
              edgecolor='none', fontsize=8, handlelength=1.3)

    fig.text(.215, .09,
             'RL points average three separately executed final models (training seeds 7, 19, 37); '
             'they are not an ensemble policy.', fontsize=9, color='#586775')
    fig.text(.215, .066,
             'Intervals condition on the fixed policies/models and reflect test-scene/action sampling; '
             'they do not quantify uncertainty from retraining.', fontsize=9, color='#586775')
    fig.text(.215, .042,
             'All rules use current observations; coordinated rules use the known physical model. '
             'No benchmark is claimed to be globally optimal.', fontsize=9, color='#586775')
    png, pdf = output/'comparison.png', output/'comparison.pdf'
    fig.savefig(png, dpi=220, bbox_inches='tight', pad_inches=.24)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.24)
    plt.close(fig)
    args.docs_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png, args.docs_copy)
    print(f'Wrote {png}\nWrote {pdf}\nCopied {args.docs_copy}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
