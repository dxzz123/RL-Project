"""Development pilots and a frozen three-seed b=3 improvement study."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from charging_env.config import Scenario
from charging_env.rl.trainer import Trainer, TrainingConfig

SEEDS = (7, 19, 37)
PILOT_UPDATES = 150
UPDATES = 500
# Index 1 is the medium 1.4x premium under the premium pricing menu.
WARMSTART = dict(episodes=256, epochs=20, minibatch_size=256, learning_rate=1e-3,
                 price_index=1, shadow_price=.30)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_for(seed, *, pilot=False):
    return TrainingConfig(
        seed=seed, updates=UPDATES, episodes_per_update=16, epochs=6,
        minibatch_size=192, hidden_size=64, learning_rate=3e-4,
        critic_learning_rate=1e-3, value_clip=0., entropy_coef=.005,
        gae_lambda=1., eval_episodes=64, eval_interval=25,
        eval_seed_base=4300001, eval_action_seed=439991, eval_policy='stochastic',
        discount_policy_loss=True, dispatch_final_activation='silu',
        dispatch_agent_id=True, encoder_layer_norm=True, inventory_potential=.30,
        dispatch_entropy_coef=.02, dispatch_entropy_final_coef=.003,
        dispatch_entropy_anneal_updates=500, separate_network_grad_clip=True,
        training_seed_base=12000000 if pilot else 10000000+20000*SEEDS.index(seed),
    ).validate()


def initialize(study, previous):
    if study.exists():
        raise FileExistsError('Use a new study directory')
    scenario=Scenario.from_json(ROOT/'configs/utility_b3.json')
    for seed in SEEDS:
        source=previous/f'seed_{seed}'/'checkpoint.pt'
        if not source.is_file():
            raise FileNotFoundError(source)
    study.mkdir(parents=True)
    previous_hashes={}
    for seed in SEEDS:
        source=previous/f'seed_{seed}'/'checkpoint.pt'
        target=study/'previous'/f'seed_{seed}'/'checkpoint.pt'
        target.parent.mkdir(parents=True)
        shutil.copy2(source,target)
        previous_hashes[str(seed)]=digest(target)
    write_json(study/'protocol.json', {
        'study':'SiLU+b=3 targeted training improvements, 2026-10-18',
        'status':'development; final selection and training source hashes pending',
        'scenario':asdict(scenario), 'training_seeds':list(SEEDS),
        'configuration_by_seed':{str(seed):asdict(config_for(seed)) for seed in SEEDS},
        'updates_per_model':500,'training_episodes_per_model':8000,
        'training_physical_periods_per_model':96000,
        'previous_checkpoint_sha256':previous_hashes,
        'previous_checkpoint_source':str(previous.resolve()),
        'diagnostic_seed_base':4100001,'diagnostic_episodes':64,
        'training_validation_seed_base':4300001,'training_validation_episodes':64,
        'rule_validation_seed_base':4500001,'rule_validation_episodes':128,
        'test_seed_base':4700001,'test_episodes':512,
        'high_demand_seed_base':4900001,'low_inventory_seed_base':5100001,
        'stress_episodes_per_case':256,'benchmark_action_seed':459991,
        'pilot_plan':{
            'bundles':['pure','guided'],'seed':7,'updates_each':150,
            'score':'Mean stochastic validation profit at updates100,125,150',
            'selection':'Choose guided only if its score exceeds pure by more than1.00; otherwise prefer pure.',
            'warmstart':WARMSTART,
        },
        'primary':'All three final stochastic models; paired against previous same-seed models and the strongest rule selected on new rule validation.',
        'secondary':'Final deterministic on all cases; validation-best stochastic and optional behavior-cloned warmstart on base only.',
        'success_criteria':{
            'consistent_improvement':'Positive new-minus-previous point differences for every seed and positive lower endpoint of aggregate paired95%interval.',
            'consistent_rule_advantage':'Positive new-minus-rule point differences for every seed and positive lower endpoint of aggregate paired95%interval.',
        },
        'selection':'Development diagnostics and fixed pilots precede frozen final tests. No test-driven extra training, seed, mode, checkpoint, or hyperparameter selection.',
    })
    print('Initialized '+str(study),flush=True)


def training_sources():
    paths=sorted((ROOT/'src/charging_env').rglob('*.py'))+[Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)):digest(path) for path in paths}


def verify_training_inputs(scenario, settings, protocol, seed, *, pilot=False):
    if json.loads(json.dumps(asdict(scenario)))!=protocol['scenario']:
        raise ValueError('Scenario differs from the frozen study protocol')
    expected=asdict(config_for(seed,pilot=True)) if pilot else protocol['configuration_by_seed'][str(seed)]
    if json.loads(json.dumps(asdict(settings)))!=expected:
        raise ValueError('Training configuration differs from the frozen protocol')


def verify_pilot_summary(summary, horizon):
    if (summary['total_updates']!=PILOT_UPDATES
            or summary['training_episodes']!=PILOT_UPDATES*16
            or summary['training_physical_periods']!=PILOT_UPDATES*16*horizon
            or not summary['checkpoint_reload_matches']
            or not summary['all_metrics_finite']
            or summary['invalid_evaluation_actions']):
        raise ValueError('Pilot incomplete, nonfinite, or invalid')


def choose(study):
    protocol=json.loads((study/'protocol.json').read_text())
    if 'selected_bundle' in protocol:
        raise FileExistsError('The training bundle is already frozen')
    scores={}
    for bundle in ('pure','guided'):
        run=study/'pilots'/bundle
        summary=json.loads((run/'summary.json').read_text())
        verify_pilot_summary(summary, protocol['scenario']['horizon'])
        with (run/'validation_metrics.csv').open(newline='') as stream:
            rows=[row for row in csv.DictReader(stream)
                  if row['policy']=='stochastic' and int(row['update']) in (100,125,150)]
        if len(rows)!=3:
            raise ValueError('Expected three fixed late-pilot validation scores')
        values=[float(row['mean_discounted_profit']) for row in rows]
        scores[bundle]={'validation_scores':values,'selection_score':sum(values)/3}
    selected='guided' if scores['guided']['selection_score']>scores['pure']['selection_score']+1. else 'pure'
    for seed in SEEDS:
        previous=study/'previous'/f'seed_{seed}'/'checkpoint.pt'
        if digest(previous)!=protocol['previous_checkpoint_sha256'][str(seed)]:
            raise ValueError('Previous checkpoint changed before freezing the study')
    selection={'selected_bundle':selected,'pilot_scores':scores,'test_scenarios_accessed':False,
               'criterion':protocol['pilot_plan']['selection']}
    write_json(study/'pilot_selection.json',selection)
    protocol.update(status='frozen for final three-seed training and independent testing',
                    secondary='Final deterministic on all cases; validation-best stochastic and optional behavior-cloned warmstart on base only.',
                    selected_bundle=selected,selected_training_config=asdict(config_for(7)),
                    warmstart=WARMSTART if selected=='guided' else None,
                    training_source_sha256=training_sources())
    write_json(study/'protocol.json',protocol)
    print(json.dumps(selection,indent=2),flush=True)


def train(study, seed, *, pilot=False, bundle=None):
    protocol=json.loads((study/'protocol.json').read_text())
    if not pilot:
        bundle=protocol['selected_bundle']
        for name, expected in protocol['training_source_sha256'].items():
            if digest(ROOT/name)!=expected:
                raise ValueError('Frozen training source changed: '+name)
    elif seed!=7 or bundle not in ('pure','guided'):
        raise ValueError('The predeclared pilots use seed7 and pure/guided bundles')
    run=study/'pilots'/bundle if pilot else study/'training'/f'seed_{seed}'
    if run.exists():
        raise FileExistsError('Existing evidence is never overwritten: '+str(run))
    scenario=Scenario.from_json(ROOT/'configs/utility_b3.json')
    settings=config_for(seed,pilot=pilot)
    verify_training_inputs(scenario,settings,protocol,seed,pilot=pilot)
    trainer=Trainer(scenario,settings)
    run.mkdir(parents=True)
    trainer.save_checkpoint(run/'initial_checkpoint.pt')
    if bundle=='guided':
        from charging_env.rl.warmstart import warm_start_from_inventory
        demo_seed_base=62000000 if pilot else 60000000+2000*SEEDS.index(seed)
        metrics=warm_start_from_inventory(trainer,seed_base=demo_seed_base,**WARMSTART)
        write_json(run/'warmstart.json',metrics)
        trainer.save_checkpoint(run/'warmstart_checkpoint.pt')
        print('Warmstart '+json.dumps(metrics),flush=True)
    result=trainer.train(run, updates=PILOT_UPDATES if pilot else UPDATES)
    print(json.dumps(result,indent=2),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('init','pilot','choose','train'),required=True)
    parser.add_argument('--study-dir',type=Path,default=ROOT/'outputs/b3_improved_study')
    parser.add_argument('--previous-root',type=Path,default=ROOT/'outputs/silu_b3_study/training')
    parser.add_argument('--seed',type=int,choices=SEEDS,default=7)
    parser.add_argument('--bundle',choices=('pure','guided'))
    args=parser.parse_args(argv)
    study=args.study_dir.resolve()
    if args.phase=='init':
        initialize(study,args.previous_root)
    elif args.phase=='choose':
        choose(study)
    else:
        train(study,args.seed,pilot=args.phase=='pilot',bundle=args.bundle)


if __name__=='__main__':
    main()
