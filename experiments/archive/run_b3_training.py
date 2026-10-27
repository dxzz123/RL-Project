"""Fixed 500-update SiLU+b=3 study, retaining initial models for paired checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from charging_env.config import Scenario
from charging_env.rl.trainer import TrainingConfig
from run_convergence import BudgetTrainer

SEEDS = (7, 19, 37)


def config_for(seed):
    return TrainingConfig(
        seed=seed, updates=500, episodes_per_update=16, epochs=6,
        minibatch_size=192, learning_rate=3e-4, critic_learning_rate=1e-3,
        value_clip=0., entropy_coef=.005, eval_episodes=32, eval_interval=25,
        eval_seed_base=100001, eval_action_seed=12345, eval_policy='stochastic',
        discount_policy_loss=True, dispatch_final_activation='silu',
    ).validate()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    scenario = Scenario.from_json(ROOT/'configs/utility_b3.json')
    sources = sorted((ROOT/'src/charging_env').rglob('*.py')) + [
        Path(__file__).resolve(), ROOT/'experiments/archive/run_convergence.py']
    return {
        'study': 'SiLU+b=3, three training seeds, fixed 500-update budget',
        'training_seeds': list(SEEDS), 'scenario': asdict(scenario),
        'configuration_by_seed': {str(seed):asdict(config_for(seed)) for seed in SEEDS},
        'updates_per_model':500, 'training_episodes_per_model':8000,
        'training_physical_periods_per_model':96000,
        'rule_validation_seed_base':3300001, 'rule_validation_episodes':128,
        'test_seed_base':3500001, 'test_episodes':512,
        'stress_seed_base':3700001, 'stress_episodes_per_case':256,
        'benchmark_action_seed':271991,
        'primary': 'All three final stochastic models, average and individual results against the strongest validation-selected b=3 rule.',
        'secondary': 'Final deterministic and validation-best models; initial-to-final stochastic changes within b=3.',
        'selection': 'Rule parameters selected only on b=3 validation. No test-driven model, training-seed, execution-mode or hyperparameter selection.',
        'uncertainty': 'Paired scenario intervals condition on fixed models. Report optimization-seed variation separately; training scenario seed schedules largely overlap.',
        'training_source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--study-dir',type=Path,default=ROOT/'outputs/silu_b3_study')
    args=parser.parse_args(argv)
    study=args.study_dir.resolve()
    run=study/'training'/f'seed_{args.seed}'
    if run.exists():
        parser.error('Use a new training directory; existing runs are never overwritten')
    study.mkdir(parents=True,exist_ok=True)
    study_protocol=study/'protocol.json'
    if not study_protocol.exists():
        with study_protocol.open('x') as stream:
            json.dump(protocol(),stream,indent=2,allow_nan=False);stream.write('\n')
    frozen=json.loads(study_protocol.read_text())
    config=config_for(args.seed)
    scenario=Scenario.from_json(ROOT/'configs/utility_b3.json')
    if (frozen['configuration_by_seed'][str(args.seed)]!=asdict(config)
            or frozen['scenario']!=json.loads(json.dumps(asdict(scenario)))):
        parser.error('Training must match the precommitted study scenario and configuration')
    for name,expected in frozen['training_source_sha256'].items():
        if digest(ROOT/name)!=expected:
            parser.error('Frozen training source changed: '+name)
    trainer=BudgetTrainer(scenario,config)
    run.mkdir(parents=True)
    trainer.save_checkpoint(run/'initial_checkpoint.pt')
    trainer.train(run)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
