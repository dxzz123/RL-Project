"""Describe completed fixed-policy tests and late validation, without tuning."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(7,19,37)
PRIMARY='rl_final_stochastic_mean'


def read_json(path):
    return json.loads(path.read_text())


def consistency(per_seed, aggregate):
    points=[per_seed[str(seed)]['mean_difference'] for seed in SEEDS]
    all_positive=all(value>0 for value in points)
    mean_lower_positive=aggregate['paired_bootstrap_confidence_interval_95'][0]>0
    return {
        'all_three_point_differences_positive':all_positive,
        'aggregate_paired_interval_lower_positive':mean_lower_positive,
        'predeclared_consistency_criterion_passed':all_positive and mean_lower_positive,
        'all_three_individual_interval_lowers_positive':all(
            per_seed[str(seed)]['paired_bootstrap_confidence_interval_95'][0]>0 for seed in SEEDS),
        'per_training_seed':per_seed,'aggregate_comparison':aggregate,
        'scope':'Conditional on these three fixed trained models; not general retraining guarantees.',
    }


def summarize(study):
    benchmark=read_json(study/'benchmarks/summary.json')
    protocol=read_json(study/'benchmarks/protocol.json')
    root_protocol=read_json(study/'protocol.json')
    if benchmark['test_tuning_performed'] or len(protocol['test_seeds'])!=512:
        raise ValueError('Expected the completed frozen 512-scene independent study')
    training={}
    for seed in SEEDS:
        run=study/'training'/f'seed_{seed}'
        summary=read_json(run/'summary.json')
        if (summary['total_updates']!=500 or summary['training_physical_periods']!=96000
                or not summary['all_metrics_finite'] or not summary['checkpoint_reload_matches']):
            raise ValueError('Training did not finish correctly: '+str(seed))
        with (run/'validation_metrics.csv').open(newline='') as stream:
            rows=[row for row in csv.DictReader(stream) if row['policy']=='stochastic']
        late=[row for row in rows if int(row['update'])>=375]
        if [int(row['update']) for row in late]!=[375,400,425,450,475,500]:
            raise ValueError('Expected six predeclared late validation checkpoints')
        values=np.array([float(row['mean_discounted_profit']) for row in late])
        with (run/'training_metrics.csv').open(newline='') as stream:
            updates=list(csv.DictReader(stream))
        training[str(seed)]={
            'final_validation_mean':summary['eval_after_mean_discounted_profit'],
            'validation_best_update':summary['best_update'],
            'validation_best_mean':summary['best_validation_mean_discounted_profit'],
            'late_validation_updates':[int(row['update']) for row in late],
            'late_validation_profit':values.tolist(),
            'late_validation_min':float(values.min()),'late_validation_max':float(values.max()),
            'late_validation_sample_sd':float(values.std(ddof=1)),
            'late_validation_first_to_last_change':float(values[-1]-values[0]),
            'last50_training_profit_mean':float(np.mean([
                float(row['mean_training_discounted_profit']) for row in updates[-50:]])),
            'training_periods':summary['training_physical_periods'],
            'invalid_evaluation_actions':summary['invalid_evaluation_actions'],
        }
    cases={}
    for case,result in benchmark['cases'].items():
        comparisons=result['comparisons_to_validation_best_rule']
        per_seed={str(seed):comparisons[f'rl_final_stochastic_{seed}'] for seed in SEEDS}
        cases[case]={
            'rule':result['strongest_validation_rule'],
            'primary_profit':result['model_averages'][PRIMARY]['mean_discounted_profit'],
            'primary_training_seed_sample_sd':result['model_averages'][PRIMARY]['training_seed_sd'],
            'primary_profit_by_seed':result['model_averages'][PRIMARY]['mean_by_training_seed'],
            'versus_rule':consistency(per_seed,comparisons[PRIMARY]),
            'invalid_actions_all_evaluations':result['invalid_actions_all_evaluations'],
        }
    initial=benchmark['cases']['base']['initial_to_final']
    return {
        'training':training,'cases':cases,
        'initial_to_final':consistency(initial['per_training_seed'],initial['mean']),
        'budget':{'total_training_periods':288000,
                  'reused_training_periods':96000 if root_protocol.get('reuse_seed_7') else 0,
                  'new_training_periods':192000 if root_protocol.get('reuse_seed_7') else 288000,
                  'rule_validation_policy_episodes':61*128,
                  'test_policy_episodes':benchmark['test_policy_episodes']},
        'late_validation_scope':'Six correlated fixed-validation scores, descriptive stability only; not a convergence test.',
        'selection_policy':protocol['selection'],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,default=ROOT/'outputs/silu_b3_study')
    args=parser.parse_args(argv)
    study=args.study_dir.resolve()
    result=summarize(study)
    with (study/'analysis.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print('Wrote '+str(study/'analysis.json'))


if __name__=='__main__':
    main()
