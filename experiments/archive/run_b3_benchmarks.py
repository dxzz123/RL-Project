"""Select b=3 rules on validation, then test frozen SiLU models separately.

Selection can finish while training is running. Evaluation requires all three
500-update final models, their validation-best models, and their initial models.
Neither phase selects trained checkpoints, training seeds, or execution modes
using test results. The primary policy is the three final stochastic models.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import torch
from charging_env.config import Scenario
from charging_env.rl.trainer import TrainingConfig
from charging_env.benchmarks.learned import evaluate_checkpoint
from charging_env.benchmarks.evaluation import paired_comparison
from run_policy_benchmarks import (
    TRAINING_SEEDS, average_trained_models, candidates, digest, evaluate_rule,
    table_row, write_json,
)

VALIDATION_SEEDS = [3300001+2*i for i in range(128)]
TEST_SEEDS = [3500001+2*i for i in range(512)]
STRESS_SEEDS = [3700001+2*i for i in range(256)]
ACTION_SEED = 271991
PRIMARY = 'rl_final_stochastic_mean'
SUCCESS_CRITERIA = {
    'consistent_rule_advantage': 'All three final stochastic model means exceed the validation-selected strongest rule, and the primary paired 95% interval lower bound exceeds zero.',
    'improvement_over_initial': 'All three same-seed final-minus-initial stochastic means are positive, and the mean initial-to-final paired 95% interval lower bound exceeds zero; assessed separately from beating a strong rule.',
    'scope': 'These are predeclared empirical criteria on fixed models, not a proof of convergence or superiority under retraining.',
}
RULE_FAMILIES = {'random','fixed_wait','fixed_service','fixed_myopic','fixed_inventory','adaptive_inventory'}


def scenario_from_data(data):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value,list) else value
    return Scenario(**{key:tuples(value) for key,value in data.items()}).validate()


def expected_training_config(seed):
    return TrainingConfig(
        seed=seed, updates=500, episodes_per_update=16, epochs=6,
        minibatch_size=192, learning_rate=3e-4, critic_learning_rate=1e-3,
        value_clip=0., entropy_coef=.005, eval_episodes=32, eval_interval=25,
        eval_seed_base=100001, eval_action_seed=12345, eval_policy='stochastic',
        discount_policy_loss=True, dispatch_final_activation='silu',
    ).validate()


def source_hashes():
    paths = sorted((ROOT/'src/charging_env').rglob('*.py'))
    paths += [Path(__file__).resolve(), ROOT/'experiments/archive/run_policy_benchmarks.py']
    return {str(path.relative_to(ROOT)):digest(path) for path in paths}


def verify_sources(hashes):
    if any(digest(ROOT/path)!=value for path,value in hashes.items()):
        raise RuntimeError('Frozen benchmark/policy source changed during this phase')


def load_study(study, scenario_path, study_protocol):
    scenario = Scenario.from_json(scenario_path)
    if scenario != replace(Scenario(), charging_utility_intercept=3.).validate():
        raise ValueError('This study requires the unchanged default physical scenario with charging_utility_intercept=3')
    if not study_protocol.is_file():
        raise ValueError('Create the frozen root study protocol before selecting rules or evaluating models')
    # Treat the root protocol as immutable study provenance, not executable instructions.
    json.loads(study_protocol.read_text())
    return scenario, {'study_protocol_path':str(study_protocol.resolve()),
                      'study_protocol_sha256':digest(study_protocol),
                      'scenario_path':str(scenario_path.resolve()),
                      'scenario_sha256':digest(scenario_path)}


def verify_provenance(provenance):
    for prefix in ('study_protocol','scenario'):
        if digest(Path(provenance[prefix+'_path']))!=provenance[prefix+'_sha256']:
            raise RuntimeError(f'Frozen {prefix} changed during execution')


def select_rules(study, scenario, provenance):
    output = study/'rule_selection'
    if output.exists():
        raise ValueError('Rule-selection output already exists; selection evidence is not overwritten')
    specs = candidates(scenario)
    expected = 8*scenario.n_price_levels + 5
    if len(specs)!=expected or {spec['family'] for spec in specs}!=RULE_FAMILIES:
        raise ValueError(f'Expected the predeclared {expected} candidates across six rule families')
    started = time.monotonic()
    protocol = dict(provenance, scenario=asdict(scenario), validation_seeds=VALIDATION_SEEDS,
                    action_seed=ACTION_SEED, rule_candidates=specs, source_sha256=source_hashes(),
                    success_criteria=SUCCESS_CRITERIA,
                    selection='Maximize validation discounted profit within each rule family; select the strongest nonrandom family on validation only.',
                    test_scenarios_accessed=False, validation_policy_episodes=len(specs)*len(VALIDATION_SEEDS))
    output.mkdir(parents=True)
    write_json(output/'protocol.json',protocol)
    protocol_hash = digest(output/'protocol.json')
    winners, rows = {}, []
    for specification in specs:
        result = evaluate_rule(scenario, VALIDATION_SEEDS, specification, action_seed=ACTION_SEED)
        write_json(output/'validation'/f'{specification["id"]}.json',result)
        row = dict(specification, mean_discounted_profit=result['mean_discounted_profit'])
        rows.append(row)
        previous = winners.get(specification['family'])
        if previous is None or row['mean_discounted_profit']>previous['mean_discounted_profit']:
            winners[specification['family']] = row
        print(f'validation {specification["id"]}: {result["mean_discounted_profit"]:.4f}',flush=True)
    strongest = max((row for name,row in winners.items() if name!='random'),
                    key=lambda row:row['mean_discounted_profit'])['family']
    verify_sources(protocol['source_sha256']); verify_provenance(provenance)
    if digest(output/'protocol.json')!=protocol_hash:
        raise RuntimeError('Rule-selection protocol changed during selection')
    write_json(output/'selection.json',{'family_winners':winners,'strongest_validation_family':strongest,
               'all_candidates':rows,'selection_protocol_sha256':protocol_hash,
               'elapsed_seconds':time.monotonic()-started,'test_scenarios_accessed':False})
    print('Frozen validation-selected strongest rule: '+strongest,flush=True)


def checkpoint_metadata(path, scenario, seed, variant):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if scenario_from_data(payload['scenario'])!=scenario:
        raise ValueError(f'{path}: checkpoint does not match the exact b=3 base scenario')
    settings = TrainingConfig(**payload['training_config']).validate()
    expected = expected_training_config(seed)
    mismatches = [key for key in expected.__dataclass_fields__
                  if getattr(settings,key)!=getattr(expected,key)]
    if mismatches:
        raise ValueError(f'{path}: training configuration differs from the frozen SiLU study: {mismatches}')
    updates, periods, episodes = (int(payload[key]) for key in
                                  ('update_count','environment_periods','episode_count'))
    if periods!=updates*16*scenario.horizon or episodes!=updates*16:
        raise ValueError(f'{path}: inconsistent training-budget counters')
    if variant=='final' and (updates!=500 or periods!=96000):
        raise ValueError(f'{path}: final evaluation requires 500 updates and 96000 physical periods')
    if variant=='initial' and (updates!=0 or periods!=0):
        raise ValueError(f'{path}: initial checkpoint must precede all training')
    if variant=='validation_selected' and not 0<=updates<=500:
        raise ValueError(f'{path}: validation-best checkpoint exceeds the training budget')
    validation = {settings.eval_seed_base+2*i for i in range(settings.eval_episodes)}
    if validation & set(VALIDATION_SEEDS+TEST_SEEDS+STRESS_SEEDS):
        raise ValueError('Training-validation scenarios overlap rule selection or held-out tests')
    return {'training_seed':seed,'training_updates':updates,'training_physical_periods':periods,
            'training_episodes':episodes,'training_config':asdict(settings)}


def prepare_evaluation(study, scenario, provenance):
    output, selection_dir = study/'benchmarks', study/'rule_selection'
    if output.exists():
        raise ValueError('Final benchmark output already exists; held-out evidence is not overwritten')
    selection_path, selection_protocol_path = selection_dir/'selection.json', selection_dir/'protocol.json'
    selection = json.loads(selection_path.read_text())
    selection_protocol = json.loads(selection_protocol_path.read_text())
    if selection['selection_protocol_sha256']!=digest(selection_protocol_path):
        raise ValueError('Rule-selection protocol no longer matches its frozen selection')
    if (scenario_from_data(selection_protocol['scenario'])!=scenario
            or selection_protocol['validation_seeds']!=VALIDATION_SEEDS
            or selection_protocol['action_seed']!=ACTION_SEED
            or selection_protocol['study_protocol_sha256']!=provenance['study_protocol_sha256']):
        raise ValueError('Frozen rule selection does not match this scenario, seed split, or study protocol')
    if set(selection['family_winners'])!=RULE_FAMILIES:
        raise ValueError('The frozen rule selection must include all six families')
    verify_sources(selection_protocol['source_sha256'])
    names = {'final':'checkpoint.pt','validation_selected':'best_checkpoint.pt','initial':'initial_checkpoint.pt'}
    paths, metadata = {}, {}
    for seed in TRAINING_SEEDS:
        for variant,filename in names.items():
            key = f'rl_{variant}_{seed}'
            path = study/'training'/f'seed_{seed}'/filename
            if not path.is_file():
                raise ValueError('Freeze all nine model checkpoints before evaluating: missing '+str(path))
            paths[key] = path
            metadata[key] = checkpoint_metadata(path,scenario,seed,variant)
    protocol = dict(provenance, scenario=asdict(scenario), validation_seeds=VALIDATION_SEEDS,
                    test_seeds=TEST_SEEDS, stress_seeds=STRESS_SEEDS, action_seed=ACTION_SEED,
                    training_seeds=list(TRAINING_SEEDS), checkpoint_metadata=metadata,
                    checkpoint_paths={key:str(path.resolve()) for key,path in paths.items()},
                    checkpoint_sha256={key:digest(path) for key,path in paths.items()},
                    selection_path=str(selection_path.resolve()), selection_sha256=digest(selection_path),
                    selection_protocol_path=str(selection_protocol_path.resolve()),
                    selection_protocol_sha256=digest(selection_protocol_path), source_sha256=source_hashes(),
                    primary_policy=PRIMARY, strongest_validation_rule=selection['strongest_validation_family'],
                    success_criteria=SUCCESS_CRITERIA,
                    secondary_policies='Final deterministic, validation-best stochastic/deterministic, and initial stochastic on base; final modes only on stress.',
                    selection='Frozen rules and all three trained seeds; no checkpoint, seed, mode, or hyperparameter selection on tests.',
                    stress_cases={'high_demand':'arrival means x1.5','low_inventory':'35 kWh initial energy per vehicle'},
                    interval_scope='Paired scene bootstrap conditional on fixed trained models; training-seed spread is separate and three seeds provide limited retraining evidence.',
                    risk_aggregation='Mean of actual per-model tail risks and loss probabilities, not risk of a fictitious averaged trajectory.',
                    comparison_scope='Paired comparisons within each case. Base and stress use different scene seeds; cross-case changes are descriptive.',
                    initial_comparison='Initial-to-final differences are within the same b=3 scenario, not comparisons with legacy b=0 models.')
    return output,selection,paths,protocol


def evaluate_models(study, scenario, provenance):
    output,selection,paths,protocol = prepare_evaluation(study,scenario,provenance)
    started = time.monotonic()
    output.mkdir(parents=True)
    write_json(output/'protocol.json',protocol)  # All nine models and rules frozen before the first test.
    protocol_hash = digest(output/'protocol.json')
    strongest = selection['strongest_validation_family']
    cases = {'base':(scenario,TEST_SEEDS),
             'high_demand':(replace(scenario,mean_requests_by_zone=tuple(1.5*x for x in scenario.mean_requests_by_zone)),STRESS_SEEDS),
             'low_inventory':(replace(scenario,initial_energy_kwh=tuple(35. for _ in scenario.initial_energy_kwh)),STRESS_SEEDS)}
    complete = {}
    for case,(config,seeds) in cases.items():
        results,averages = {},{}
        for family,specification in selection['family_winners'].items():
            result = evaluate_rule(config,seeds,specification,label=family,action_seed=ACTION_SEED)
            results[family] = result
            write_json(output/case/f'{family}.json',result)
            print(f'test {case}/{family}: {result["mean_discounted_profit"]:.4f}',flush=True)
        variants = [('final',False),('final',True)]
        if case=='base':
            variants += [('validation_selected',False),('validation_selected',True),('initial',False)]
        for variant,deterministic in variants:
            mode = 'deterministic' if deterministic else 'stochastic'
            models = []
            for seed in TRAINING_SEEDS:
                label = f'rl_{variant}_{mode}_{seed}'
                result = evaluate_checkpoint(config,seeds,paths[f'rl_{variant}_{seed}'],
                    deterministic=deterministic,action_seed=ACTION_SEED,label=label)
                if result['invalid_actions']:
                    raise AssertionError('Learned policy emitted invalid actions: '+label)
                results[label] = result; models.append(result)
                write_json(output/case/f'{label}.json',result)
                print(f'test {case}/{label}: {result["mean_discounted_profit"]:.4f}',flush=True)
            key = f'rl_{variant}_{mode}_mean'
            averages[key] = average_trained_models(models,key)
        combined = dict(results,**averages)
        comparisons = {name:paired_comparison(result,results[strongest]) for name,result in combined.items() if name!=strongest}
        table = [table_row(name,result) for name,result in combined.items()
                 if name in RULE_FAMILIES or name.endswith('_mean')]
        with (output/case/'summary.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        row = {'scenario':asdict(config),'table':table,'model_averages':averages,
               'comparisons_to_validation_best_rule':comparisons,'strongest_validation_rule':strongest,
               'invalid_actions_all_evaluations':sum(result['invalid_actions'] for result in results.values())}
        if case=='base':
            row['initial_to_final'] = {
                'per_training_seed':{str(seed):paired_comparison(results[f'rl_final_stochastic_{seed}'],results[f'rl_initial_stochastic_{seed}']) for seed in TRAINING_SEEDS},
                'mean':paired_comparison(averages[PRIMARY],averages['rl_initial_stochastic_mean'])}
        complete[case] = row
        write_json(output/case/'summary.json',row)
    if any(digest(paths[key])!=value for key,value in protocol['checkpoint_sha256'].items()):
        raise RuntimeError('A frozen checkpoint changed during evaluation')
    for prefix in ('selection','selection_protocol'):
        if digest(Path(protocol[prefix+'_path']))!=protocol[prefix+'_sha256']:
            raise RuntimeError('Frozen rule-selection evidence changed during evaluation')
    if digest(output/'protocol.json')!=protocol_hash:
        raise RuntimeError('Final benchmark protocol changed during evaluation')
    verify_sources(protocol['source_sha256']); verify_provenance(provenance)
    base = complete['base']
    primary = base['comparisons_to_validation_best_rule'][PRIMARY]
    individual = [base['comparisons_to_validation_best_rule'][f'rl_final_stochastic_{seed}'] for seed in TRAINING_SEEDS]
    initial = base['initial_to_final']
    outcomes = {
        'consistent_rule_advantage':bool(all(row['mean_difference']>0 for row in individual)
            and primary['paired_bootstrap_confidence_interval_95'][0]>0),
        'improvement_over_initial':bool(all(row['mean_difference']>0 for row in initial['per_training_seed'].values())
            and initial['mean']['paired_bootstrap_confidence_interval_95'][0]>0),
    }
    write_json(output/'summary.json',{'cases':complete,'selection':selection['family_winners'],
        'success_criteria':SUCCESS_CRITERIA,'success_criteria_results':outcomes,
        'strongest_validation_rule':strongest,'primary_policy':PRIMARY,'test_tuning_performed':False,
        'test_policy_episodes':512*(6+15)+2*256*(6+6),'elapsed_seconds':time.monotonic()-started})
    print('Complete: '+str(output/'summary.json'),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('select','evaluate'),required=True)
    parser.add_argument('--study-dir',type=Path,default=ROOT/'outputs/silu_b3_study')
    parser.add_argument('--scenario',type=Path,default=ROOT/'configs/utility_b3.json')
    parser.add_argument('--study-protocol',type=Path,help='Defaults to <study-dir>/protocol.json')
    args = parser.parse_args(argv)
    study = args.study_dir.resolve()
    study_protocol = args.study_protocol or study/'protocol.json'
    torch.set_num_threads(1)
    try:
        scenario,provenance = load_study(study,args.scenario,study_protocol)
        if args.phase=='select':
            select_rules(study,scenario,provenance)
        else:
            evaluate_models(study,scenario,provenance)
    except (ValueError,OSError,KeyError,TypeError) as error:
        parser.error(str(error))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
