"""Plot the completed five-method fitted-value study; never refit or select models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

from charging_env.rl.fitted_value import FittedValuePolicy

LABELS = {
    'fitted_value_mean': 'Learned continuation value',
    'ppo_mean': 'SA-PPO / MAPPO',
    'service_first': 'Service first',
    'constant_inventory': 'Constant inventory value',
    'time_inventory': 'Time-dependent inventory value',
}
COLORS = {'fitted_value_mean': '#008c70', 'ppo_mean': '#df7c28',
          'service_first': '#8a97a7', 'constant_inventory': '#647589', 'time_inventory': '#8b4f9c'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=ROOT / 'outputs/fitted_value_study')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    summary_path = study / 'benchmarks/summary.json'
    protocol_path = study / 'benchmarks/protocol.json'
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if summary['test_policy_episodes'] != 9216 or summary['test_tuning_performed']:
        raise ValueError('Plot requires the complete frozen 9,216-episode study')
    inputs = {str(summary_path): digest(summary_path), str(protocol_path): digest(protocol_path)}
    for key, rel in protocol['checkpoint_paths'].items():
        checkpoint = study / rel
        if digest(checkpoint) != protocol['checkpoint_sha256'][key]:
            raise ValueError('Frozen checkpoint changed: ' + key)
        inputs[str(checkpoint)] = digest(checkpoint)
    methods = list(LABELS)
    cases = summary['cases']
    if any([row['policy'] for row in case['table']] != methods for case in cases.values()):
        raise ValueError('Plot must contain exactly the five predefined methods')
    base = cases['base']
    strongest = summary['strongest_validation_rule']
    base_rows = {row['policy']: row for row in base['table']}
    episode_profits = {}
    for name in methods:
        if name in base['model_averages']:
            result = base['model_averages'][name]
        else:
            path = study / 'benchmarks/base' / f'{name}.json'
            inputs[str(path)] = digest(path)
            result = json.loads(path.read_text())
        episode_profits[name] = np.array([row['discounted_profit'] for row in result['episodes']])
        if len(episode_profits[name]) != 512:
            raise ValueError('Base chart requires 512 shared scenes')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#a8b3bf', 'text.color': '#233446',
                         'axes.labelcolor': '#233446', 'xtick.color': '#536579', 'ytick.color': '#536579'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 12.3))
    fig.subplots_adjust(left=.17, right=.97, top=.83, bottom=.25, hspace=.52, wspace=.46)
    fig.text(.09, .96, 'Learned continuation value versus PPO and three key rules', fontsize=19, weight='bold')
    fig.text(.09, .925, 'b = 3 | Fixed 0.60 mobile price for fitted value | Training seeds 7, 19, 37 | 512 independent base scenes', fontsize=11)
    fig.text(.09, .895, 'Three rules selected on 128 validation scenes; all weights and choices frozen before base and stress tests', fontsize=10.5, color='#5f7185')
    ax = axes[0, 0]
    y = np.arange(len(methods))
    means = [base_rows[name]['mean_discounted_profit'] for name in methods]
    errors = [1.96 * episode_profits[name].std(ddof=1) / np.sqrt(512) for name in methods]
    ax.barh(y, means, color=[COLORS[name] for name in methods], height=.55, alpha=.88)
    ax.errorbar(means, y, xerr=errors, fmt='none', ecolor='#253e53', capsize=3, lw=1.2)
    ax.set_yticks(y, ['Learned continuation\nvalue', 'SA-PPO / MAPPO', 'Service first', 'Constant inventory\nvalue', 'Time-dependent\ninventory value']); ax.invert_yaxis()
    ax.set_xlim(0, max(means) + 15)
    ax.set_xlabel('Mean discounted business profit per episode')
    ax.set_title('A  Base-test means and conditional 95% intervals', loc='left', weight='bold', fontsize=11.5)
    for i, value in enumerate(means): ax.text(value+3.3, i, f'{value:.2f}', va='center', fontsize=9.5)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for name in ('fitted_value_mean', 'ppo_mean'):
        values = base['model_averages'][name]['mean_by_training_seed']
        index = methods.index(name)
        ax.scatter(list(values.values()), np.full(3, index)+.19, color='#1d3041', s=15, zorder=5)
    ax = axes[0, 1]
    comparisons = base['comparisons_to_validation_best_rule']
    keys = ['fitted_7', 'fitted_19', 'fitted_37', 'fitted_value_mean']
    for i, key in enumerate(keys):
        item = comparisons[key]
        mean = item['mean_difference']; low, high = item['paired_bootstrap_confidence_interval_95']
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='D' if i==3 else 'o',
                    color='#008c70' if i<3 else '#8b4f9c', capsize=4, ms=6)
    ax.axvline(0, color='#6c7f90', linestyle='--', lw=1)
    ax.set_yticks(range(4), ['Seed 7', 'Seed 19', 'Seed 37', 'Three-model mean']); ax.invert_yaxis()
    ax.set_xlabel('Paired profit gain over the selected rule')
    ax.set_title('B  Paired gains over the validation-selected rule', loc='left', weight='bold', fontsize=11.5)
    ax.grid(axis='x', alpha=.15)
    ax.text(0, -.24, 'Baseline: '+LABELS[strongest]+'; 5,000 paired bootstrap draws', transform=ax.transAxes, fontsize=9, color='#5f7185')
    ax = axes[1, 0]
    x = np.arange(3); width=.15
    for offset, name in enumerate(methods):
        values = [next(row['mean_discounted_profit'] for row in cases[case]['table'] if row['policy']==name)
                  for case in ('base', 'high_demand', 'low_inventory')]
        ax.bar(x+(offset-2)*width, values, width=width, color=COLORS[name], label=LABELS[name], alpha=.9)
    ax.set_xticks(x, ['Base', 'Demand x1.5', 'Initial energy\n35 kWh / vehicle'])
    ax.set_ylabel('Mean discounted profit per episode')
    ax.set_title('C  Base and independent stress cases', loc='left', weight='bold', fontsize=11.5)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.56, .115), ncol=3, frameon=False, fontsize=9)
    ax = axes[1, 1]
    policy = FittedValuePolicy.load_checkpoint(study / protocol['checkpoint_paths']['fitted_7'])
    energy = np.linspace(policy.scenario.reserve_kwh, policy.scenario.battery_capacity_kwh, 151)
    locations = np.broadcast_to(np.array(policy.scenario.initial_locations), (len(energy), 2))
    batteries = np.column_stack((energy, np.full(len(energy), 55.)))
    reference = batteries.copy(); reference[:, 0] = policy.scenario.reserve_kwh
    for period, color in [(0, '#008c70'), (6, '#4678b5'), (11, '#df7c28')]:
        increment = policy.continuation(period, locations, batteries) - policy.continuation(period, locations, reference)
        ax.plot(energy, increment, color=color, lw=2, label=f'Period {period+1}')
    ax.set_xlabel('Vehicle 0 stored energy (kWh); vehicle 1 fixed at 55 kWh')
    ax.set_ylabel('Additional fitted value above reserve')
    ax.set_title('D  Learned energy value at different times', loc='left', weight='bold', fontsize=11.5)
    ax.legend(frameon=False); ax.grid(alpha=.15)
    fig.text(.09, .075, 'A: dots show individual training-seed means. Intervals condition on fixed models, not arbitrary retraining. B uses the frozen validation winner.', fontsize=9, color='#5f7185')
    fig.text(.09, .055, 'C: each stress case uses 256 scenes, with different seeds from the base case. Cross-case changes are descriptive. D: predeclared seed 7, initial locations.', fontsize=9, color='#5f7185')
    fig.text(.09, .035, 'Fitted value uses a known one-period model and arbitrary state queries; PPO uses episode rollouts. Training budgets and information are different.', fontsize=9, color='#5f7185')
    output = study / 'fitted_value_comparison.png'
    fig.savefig(output, dpi=180, facecolor='white'); plt.close(fig)
    for name, expected in inputs.items():
        if digest(Path(name)) != expected: raise RuntimeError('An input changed while plotting: '+name)
    relative_inputs = {str(Path(name).relative_to(study)): value for name, value in inputs.items()}
    (study/'plot_provenance.json').write_text(json.dumps({'input_sha256': relative_inputs, 'plot_script_sha256': digest(Path(__file__)),
        'output_sha256': digest(output), 'source': 'Saved frozen test results; panel D uses saved training tables only.'}, indent=2)+'\n')
    print('Wrote '+str(output))


if __name__ == '__main__':
    main()
