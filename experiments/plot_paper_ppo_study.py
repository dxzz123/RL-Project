"""Plot saved Joint PPO comparisons and monitoring records without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'paper_ppo_mean': 'Joint PPO',
    'legacy_ppo_mean': 'Earlier PPO / MAPPO',
    'fitted_value_mean': 'Fitted-value dispatch',
    'service_first': 'Service first',
    'constant_inventory': 'Constant inventory value',
    'time_inventory': 'Time-dependent inventory value',
}
COLORS = dict(zip(LABELS, ['#087f8c', '#db8536', '#209b77', '#9ba6b2', '#647c95', '#9863a4']))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plot(study):
    inputs = {}

    def read(relative):
        path = study / relative
        inputs[str(relative)] = digest(path)
        return json.loads(path.read_text())

    analysis = read('analysis.json')
    summary = read('benchmarks/summary.json')
    protocol = read('benchmarks/protocol.json')
    if summary['test_policy_episodes'] != 12288 or summary['test_tuning_performed']:
        raise ValueError('Expected the completed fixed 12,288-episode comparison')
    for label, rel in protocol['checkpoint_paths'].items():
        value = digest(study / rel)
        if value != protocol['checkpoint_sha256'][label]:
            raise ValueError('Frozen checkpoint changed: ' + label)
        inputs[rel] = value
    cases, methods = analysis['cases'], list(LABELS)
    if any(set(case['mean_discounted_profits']) != set(methods) for case in cases.values()):
        raise ValueError('The plot requires exactly the six main methods')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#a8b3bf', 'text.color': '#233446',
                         'axes.labelcolor': '#233446', 'xtick.color': '#536579',
                         'ytick.color': '#536579'})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.6))
    fig.subplots_adjust(left=.18, right=.97, top=.83, bottom=.25, hspace=.55, wspace=.46)
    fig.text(.08, .96, 'Joint PPO versus earlier learners and three rules', fontsize=19, weight='bold')
    fig.text(.08, .925, 'b = 3 | Joint PPO: A2 dispatch critic target, joint fleet ratio, interleaved GAE | Training seeds 7, 19, 37', fontsize=11)
    fig.text(.08, .895, 'Final 500-update stochastic policies; all settings frozen before 512 independent base and two 256-scenario stress tests', fontsize=10.5, color='#5f7185')

    ax = axes[0, 0]
    base = cases['base']
    values = [base['mean_discounted_profits'][name] for name in methods]
    ax.barh(np.arange(6), values, color=[COLORS[name] for name in methods], height=.58)
    short_labels = ['Joint PPO', 'Earlier PPO / MAPPO', 'Fitted-value dispatch',
                    'Service first', 'Constant inventory\nvalue', 'Time-dependent\ninventory value']
    ax.set_yticks(np.arange(6), short_labels)
    ax.invert_yaxis()
    spread = max(max(values)-min(values), 20.)
    ax.set_xlim(min(0., min(values)-.07*spread), max(values)+.20*spread)
    for index, value in enumerate(values):
        ax.text(value+.015*spread, index, f'{value:.2f}', va='center', fontsize=10)
    for name, field in [('paper_ppo_mean', 'paper_by_training_seed'),
                        ('legacy_ppo_mean', 'legacy_by_training_seed'),
                        ('fitted_value_mean', 'fitted_by_training_seed')]:
        seed_means = list(base[field].values())
        ax.scatter(seed_means, np.full(3, methods.index(name))+.19, color='#223347', s=16, zorder=4)
    ax.set_xlabel('Mean discounted business profit per episode')
    ax.set_title('A  Six-method base test', loc='left', weight='bold', fontsize=11.5)
    ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)

    ax = axes[0, 1]
    comparisons = base['new_vs_legacy']
    entries = [(f'Seed {seed}', comparisons['per_training_seed'][str(seed)]) for seed in (7, 19, 37)]
    entries.append(('Three-model mean', comparisons['mean']))
    for index, (label, item) in enumerate(entries):
        value = item['mean_difference']
        low, high = item['paired_bootstrap_confidence_interval_95']
        color = COLORS['paper_ppo_mean'] if index < 3 else '#9863a4'
        ax.errorbar(value, index, xerr=[[value-low], [high-value]], fmt='D' if index == 3 else 'o',
                    capsize=4, color=color, markersize=6)
    ax.axvline(0., color='#7f90a3', linestyle='--', lw=1)
    ax.set_yticks(np.arange(4), [label for label, _ in entries]); ax.invert_yaxis()
    ax.set_xlabel('Paired profit gain over earlier PPO')
    ax.set_title('B  New minus earlier PPO: paired 95% intervals', loc='left', weight='bold', fontsize=11.5)
    ax.text(0, -.24, 'Earlier PPO: matching seed labels; 5,000 paired bootstrap draws',
            transform=ax.transAxes, fontsize=9, color='#5f7185')
    ax.set_axisbelow(True); ax.grid(axis='x', alpha=.15)

    ax = axes[1, 0]
    locations, width = np.arange(3), .125
    case_order = ('base', 'high_demand', 'low_inventory')
    handles = []
    for index, name in enumerate(methods):
        bars = ax.bar(locations+(index-2.5)*width,
                      [cases[case]['mean_discounted_profits'][name] for case in case_order],
                      width=width, color=COLORS[name], label=LABELS[name])
        handles.append(bars)
    ax.set_xticks(locations, ['Base', 'Demand x1.5', 'Initial energy\n35 kWh / vehicle'])
    ax.set_ylabel('Mean discounted profit per episode')
    ax.set_title('C  Independent stress cases', loc='left', weight='bold', fontsize=11.5)
    ax.set_axisbelow(True); ax.grid(axis='y', alpha=.15)

    ax = axes[1, 1]
    for seed, color in zip((7, 19, 37), ['#087f8c', '#416eaa', '#d4873c']):
        history = read(f'training/seed_{seed}/validation_history.json')
        if isinstance(history, dict):
            history = history['evaluations']
        xs = [row['update'] for row in history]
        ys = [row['mean_discounted_profit'] for row in history]
        ax.plot(xs, ys, label=f'Seed {seed}', lw=1.8, color=color)
    ax.set_xlabel('Training update')
    ax.set_ylabel('Mean discounted monitoring profit')
    ax.set_title('D  Separate validation monitoring (64 scenarios)', loc='left', weight='bold', fontsize=11.5)
    ax.legend(frameon=False); ax.grid(alpha=.15)
    fig.legend(handles, [LABELS[name] for name in methods], frameon=False,
               ncol=3, loc='center', bbox_to_anchor=(.56, .12), fontsize=10)
    fig.text(.08, .075, 'A: dots show individual training-seed means. B: intervals condition on fixed trained models; they exclude arbitrary retraining uncertainty.', fontsize=9, color='#5f7185')
    fig.text(.08, .055, 'C: stress cases use different scenario seeds from base. D: monitoring is separate from all tests; the final checkpoint remains primary.', fontsize=9, color='#5f7185')
    fig.text(.08, .035, 'Controllers differ in observation and action sets. Fitted value has stronger model access; equal rollout budgets do not imply equal information or computation.', fontsize=9, color='#5f7185')
    output = study / 'paper_ppo_comparison.png'
    fig.savefig(output, dpi=180, facecolor='white'); plt.close(fig)
    for name, expected in inputs.items():
        if digest(study/name) != expected:
            raise RuntimeError('An input changed while plotting: '+name)
    (study/'plot_provenance.json').write_text(json.dumps({
        'input_sha256': inputs, 'plot_script_sha256': digest(Path(__file__)),
        'output_sha256': digest(output),
        'scope': 'Saved independent comparison; monitoring is shown without selecting checkpoints or refitting.'}, indent=2)+'\n')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT/'outputs/paper_ppo_study')
    args = parser.parse_args(argv)
    print(make_plot(args.study_dir.resolve()))


if __name__ == '__main__':
    main()
