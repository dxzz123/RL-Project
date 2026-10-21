"""Summarize learning diagnostics after the three planned runs finish."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'outputs/convergence_study'


def main():
    results = {}
    for seed in (7, 19, 37):
        run = STUDY/f'seed_{seed}'
        summary = json.loads((run/'summary.json').read_text())
        training = [{k: float(v) for k, v in r.items()}
                    for r in csv.DictReader((run/'training_metrics.csv').open())]
        validation = list(csv.DictReader((run/'validation_metrics.csv').open()))
        stats = {'summary': summary, 'late_validation': {}}
        for policy in ('deterministic', 'stochastic'):
            rows = [r for r in validation if r['policy'] == policy and int(r['update']) >= 375]
            y = np.array([float(r['mean_discounted_profit']) for r in rows])
            x = np.array([float(r['environment_periods'])/10000 for r in rows])
            stats['late_validation'][policy] = {
                'updates': [int(r['update']) for r in rows],
                'values': y.tolist(), 'minimum': float(y.min()), 'maximum': float(y.max()),
                'mean': float(y.mean()), 'sd_across_checkpoints': float(y.std(ddof=1)),
                'descriptive_slope_per_10000_periods': float(np.polyfit(x, y, 1)[0]),
                'note': 'Repeated measurements on fixed validation scenes; descriptive stability only, no trend significance test.',
            }
        for part, rows in [('early_10_updates', training[:10]), ('last_50_updates', training[-50:])]:
            stats[part] = {key: float(np.mean([r[key] for r in rows])) for key in (
                'mean_training_discounted_profit', 'pricing_explained_variance',
                'dispatch_explained_variance', 'pricing_value_rmse', 'dispatch_value_rmse',
                'pricing_approx_kl', 'dispatch_approx_kl', 'pricing_entropy', 'dispatch_entropy')}
        stats['early_stop_count'] = int(sum(r['early_stopped'] for r in training))
        stats['all_training_metrics_finite'] = all(np.isfinite(v) for r in training for v in r.values())
        results[str(seed)] = stats
    (STUDY/'training_stability.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({s: {'best_update': d['summary']['best_update'],
                         'late_validation': d['late_validation'],
                         'last_50_updates': d['last_50_updates']} for s, d in results.items()}, indent=2))


if __name__ == '__main__':
    main()
