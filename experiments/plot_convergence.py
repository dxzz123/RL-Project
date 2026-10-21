"""Plot the fixed three-seed convergence experiment with matplotlib.

Run from the repository root after all three runs complete:
    python experiments/plot_convergence.py

Use --partial to inspect currently available rows. Partial figures are named
convergence_partial and never copied into docs/assets. Install matplotlib in
the active Python environment if it is not already available.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

# Keep matplotlib's cache writable without changing the user's global config.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'charging_plot_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


SEEDS = (7, 19, 37)
COLORS = ('#0072B2', '#D55E00', '#009E73')
EXPECTED_PERIODS = 96000
CONTROL_PERIODS = 48000


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No data rows in {path}')
    return rows


def values(rows, key):
    result = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite values in {key}')
    return result


def trailing_mean(data, window=20):
    if len(data) < window:
        return np.asarray([], dtype=float)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--partial', action='store_true', help='Render explicitly marked preview figures')
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = root / 'outputs' / 'convergence_study'
    datasets = {}
    for seed in SEEDS:
        run = output / f'seed_{seed}'
        summary_path = run / 'summary.json'
        if not args.partial:
            if not summary_path.exists():
                parser.error(f'Run seed {seed} is incomplete: missing {summary_path}; use --partial for preview')
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary['total_updates'] < 500 or summary['training_physical_periods'] < EXPECTED_PERIODS:
                parser.error(f'Run seed {seed} has not completed the declared 96,000-period budget')
        validation = read_csv(run / 'validation_metrics.csv')
        training = read_csv(run / 'training_metrics.csv')
        datasets[seed] = (validation, training)
        if not args.partial and max(values(validation, 'environment_periods')) < EXPECTED_PERIODS:
            parser.error(f'Run seed {seed} is missing final validation')
    control = json.loads((root / 'outputs' / 'baseline_experiment' / 'validation_curve.json').read_text(encoding='utf-8'))
    control_x = values(control, 'periods')
    if control_x.max() != CONTROL_PERIODS:
        parser.error('Expected original-configuration control to end at 48,000 periods')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.edgecolor': '#8b929a',
        'xtick.color': '#4b5563', 'ytick.color': '#4b5563',
        'grid.color': '#e6e9ed', 'grid.linewidth': 0.7,
        'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    stochastic_ax, deterministic_ax, service_ax, critic_ax = axes.flatten()
    profit_values, ev_values = [], []
    for seed, color in zip(SEEDS, COLORS):
        validation, training = datasets[seed]
        for policy, ax in [('stochastic', stochastic_ax), ('deterministic', deterministic_ax)]:
            rows = [r for r in validation if r['policy'] == policy]
            x, y = values(rows, 'environment_periods'), values(rows, 'mean_discounted_profit')
            ax.plot(x, y, color=color, linewidth=1.8, marker='o', markersize=3.1,
                    label=f'Candidate, seed {seed}')
            profit_values.extend(y)
        stochastic_rows = [r for r in validation if r['policy'] == 'stochastic']
        service_ax.plot(values(stochastic_rows, 'environment_periods'),
                        values(stochastic_rows, 'mean_served_request_share'),
                        color=color, linewidth=1.8, marker='o', markersize=3.1)
        ev = (values(training, 'pricing_explained_variance')
              + values(training, 'dispatch_explained_variance')) / 2.
        smoothed = trailing_mean(ev, 20)
        critic_ax.plot(values(training, 'environment_periods')[19:], smoothed,
                       color=color, linewidth=1.7)
        ev_values.extend(smoothed)

    for policy, ax in [('stochastic', stochastic_ax), ('deterministic', deterministic_ax)]:
        y = values(control, policy)
        ax.plot(control_x, y, color='#20252b', linestyle='--', linewidth=1.6,
                label='Original configuration, seed 7 (48k periods)')
        profit_values.extend(y)
        ax.axhline(0, color='#77818d', linewidth=0.75, zorder=0)
    profit_range = max(profit_values)-min(profit_values)
    for ax in (stochastic_ax, deterministic_ax):
        ax.set_ylim(min(profit_values)-.06*profit_range, max(profit_values)+.08*profit_range)
        ax.set_ylabel('Mean discounted profit ($ / episode)')

    stochastic_ax.set_title('A  Stochastic policy · validation profit', loc='left', pad=11)
    deterministic_ax.set_title('B  Deterministic policy · validation profit', loc='left', pad=11)
    service_ax.set_title('C  Stochastic policy · requests served', loc='left', pad=11)
    critic_ax.set_title('D  Critic fit · training rollouts', loc='left', pad=11)
    service_ax.set_ylabel('Mean share of all requests served')
    service_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    service_ax.set_ylim(bottom=0)
    critic_ax.set_ylabel('Explained variance (20-update trailing mean)')
    if ev_values:
        critic_ax.set_ylim(min(-.03, min(ev_values)-.03), max(1.03, max(ev_values)+.03))
    critic_ax.axhline(0, color='#77818d', linewidth=0.75, zorder=0)
    for ax in axes.flatten():
        ax.set_xlim(0, EXPECTED_PERIODS)
        ax.set_xticks([0, 24000, 48000, 72000, 96000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: '0' if value == 0 else f'{value/1000:.0f}k'))
        ax.grid(True, axis='both')
        ax.set_axisbelow(True)
    for ax in (service_ax, critic_ax):
        ax.set_xlabel('Training physical periods')

    title = 'SA-PPO–MAPPO · synthetic toy-city training'
    if args.partial:
        title += '  [PARTIAL PREVIEW]'
    fig.suptitle(title, x=.075, y=.975, ha='left', fontsize=16, fontweight='bold', color='#18212c')
    handles = [Line2D([], [], color=color, linewidth=2, marker='o', markersize=3,
                       label=f'Candidate, seed {seed}') for seed, color in zip(SEEDS, COLORS)]
    handles.append(Line2D([], [], color='#20252b', linestyle='--', linewidth=1.6,
                          label='Original configuration, seed 7 (48k periods)'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.068, .935),
               ncol=4, frameon=False, fontsize=9, handlelength=2.6, columnspacing=1.6)
    caption = (
        'A–C: 32 fixed validation episodes per checkpoint; points are unsmoothed; no confidence bands. '
        'Control stochastic action RNG differs.\n'
        'C includes mobile and fixed-station service. D averages the two post-fit critics against rollout GAE targets; '
        'only D uses a trailing mean.\n'
        'Validation is used for monitoring/selection. Independent test results and convergence limits are reported separately.'
    )
    fig.text(.075, .028, caption, ha='left', va='bottom', fontsize=8.6, color='#4b5563', linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.855, hspace=.36, wspace=.24)
    stem = 'convergence_partial' if args.partial else 'convergence'
    output.mkdir(parents=True, exist_ok=True)
    png, pdf = output / f'{stem}.png', output / f'{stem}.pdf'
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    if not args.partial:
        docs_image = root / 'docs' / 'assets' / 'training_convergence.png'
        docs_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, docs_image)
    print(json.dumps({'partial': args.partial, 'png': str(png), 'pdf': str(pdf)}, indent=2))


if __name__ == '__main__':
    main()
