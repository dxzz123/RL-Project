"""Diagnose and compare inventory-warmstarted Joint PPO on fresh single days.

Initialize once, train each seed, then evaluate the frozen final checkpoints.
The copied original PPO and ADP controls require outputs/paper_ppo_study.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from charging_env.config import Scenario
from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer, _scenario_data
from charging_env.rl.fitted_value import FittedValuePolicy
from charging_env.benchmarks.compact import evaluate_rule, candidates
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from run_fitted_value_study import average_models, table_row

SEEDS = (7, 19, 37)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def config_for(seed):
    if seed not in SEEDS:
        raise ValueError('Choose training seed 7, 19, or 37')
    offset = 20000 * SEEDS.index(seed)
    return PaperPPOConfig(seed=seed, updates=500, episodes_per_update=16,
        actor_learning_rate=1e-4, critic_learning_rate=1e-3,
        gae_lambda=1., normalize_advantages=True, reward_scale=20.,
        dispatch_entropy_coef=.005, target_kl=.03,
        warmstart_episodes=512, warmstart_epochs=100,
        warmstart_seed_base=422000000 + offset,
        warmstart_learning_rate=.001, training_seed_base=420000000 + offset,
        eval_episodes=64, eval_interval=100, eval_seed_base=421000001,
        eval_action_seed=429991).validate()


def init(study, controls):
    if study.exists():
        raise FileExistsError('Choose an unused study directory')
    # Validate every source control before creating the study.
    sources = {}
    for seed in SEEDS:
        for family, name in (('training', 'checkpoint.pt'), ('fitted', 'checkpoint.npz')):
            path = controls / family / f'seed_{seed}' / name
            sources[str(path.relative_to(controls))] = digest(path)
    rule = controls / 'rule_selection/selection.json'
    sources['rule_selection/selection.json'] = digest(rule)
    scenario = Scenario.from_json(ROOT / 'configs/utility_b3.json')
    configs = {str(seed): asdict(config_for(seed)) for seed in SEEDS}
    files = sorted((ROOT / 'src/charging_env').rglob('*.py')) + [Path(__file__), ROOT / 'experiments/run_fitted_value_study.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    protocol = dict(schema='charging_env.recharge_study.v1', scenario=asdict(scenario),
        training_seeds=list(SEEDS), configurations=configs,
        test_seeds=[431000001 + 2*i for i in range(512)], action_seed=429991,
        source_sha256=hashes, control_sha256=sources,
        checkpoint_rule='Final update 500; cloning-only snapshot before PPO is reported separately.',
        execution_mode='stochastic for both PPO variants and cloning-only controls',
        primary_comparison='New PPO minus same-seed frozen original Joint PPO.',
        success_rule='Positive paired 95% lower bound for all three training seeds; otherwise no consistent profit improvement claim.',
        budget='512 teacher episodes plus 8000 on-policy episodes per new model; teacher enumerates current joint outcomes. Original PPO controls used 8000 episodes without demonstrations.',
        limitations='Development selected this combined repair. This is benchmark-guided initialization, not PPO from scratch, and is not a single-factor ablation. Intervals condition on the three trained models.',
        rule_selection='Previously selected time-inventory rule; no new tuning or selection on test seeds.')
    study.mkdir(parents=True)
    write(study / 'protocol.json', protocol)
    for name in hashes:
        target = study / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in sources:
        target = study / 'controls' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(controls / name, target)
    print('Initialized ' + str(study), flush=True)


def verify(study):
    p = json.loads((study / 'protocol.json').read_text())
    for name, expected in p['source_sha256'].items():
        if digest(ROOT / name) != expected or digest(study / 'source_snapshot' / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for name, expected in p['control_sha256'].items():
        if digest(study / 'controls' / name) != expected:
            raise ValueError('Frozen control changed: ' + name)
    return p


def train(study, seed):
    p = verify(study)
    if seed not in SEEDS:
        raise ValueError('Training requires --seed 7, 19, or 37')
    trainer = PaperPPOTrainer(_scenario_data(p['scenario']), PaperPPOConfig(**p['configurations'][str(seed)]))
    trainer.train(study / 'training' / f'seed_{seed}')
    verify(study)


def evaluate(study):
    p = verify(study)
    dest = study / 'test'
    if dest.exists():
        raise FileExistsError('Independent test already exists; use --phase show to read it')
    scenario = _scenario_data(p['scenario'])
    selection = json.loads((study / 'controls/rule_selection/selection.json').read_text())
    rule_spec = selection['family_winners']['time_inventory']
    if {k: v for k, v in rule_spec.items() if k != 'mean_discounted_profit'} not in candidates(scenario):
        raise ValueError('Frozen inventory rule is not valid for this scenario')
    paths = {}
    for seed in SEEDS:
        paths[f'new_ppo/{seed}'] = study / f'training/seed_{seed}/checkpoint.pt'
        paths[f'cloning_only/{seed}'] = study / f'training/seed_{seed}/warmstart_checkpoint.pt'
        paths[f'original_ppo/{seed}'] = study / f'controls/training/seed_{seed}/checkpoint.pt'
        paths[f'approximate_dp/{seed}'] = study / f'controls/fitted/seed_{seed}/checkpoint.npz'
    # Validate every model before opening any test scene.
    test_seeds = set(p['test_seeds'])
    for key, path in paths.items():
        family, seed = key.split('/')
        if family == 'approximate_dp':
            model = FittedValuePolicy.load_checkpoint(path)
        else:
            model = PaperPPOTrainer.load_checkpoint(path)
            c = model.config
            if family in ('new_ppo', 'cloning_only'):
                if asdict(c) != p['configurations'][seed]:
                    raise ValueError('New model configuration differs from the frozen protocol')
                if not model.warmstart_metadata['completed']:
                    raise ValueError('Model has not completed inventory initialization: ' + key)
                warm = model.warmstart_metadata['summary']
                expected_demos = [2*(c.warmstart_seed_base+c.seed+i) for i in range(512)]
                if (warm['seeds'] != expected_demos or warm['demonstration_episodes'] != 512
                        or warm['demonstration_periods'] != 512*scenario.horizon
                        or warm['reward_scale'] != c.reward_scale
                        or any(p['source_sha256'].get(k) != v for k,v in warm['source_sha256'].items())):
                    raise ValueError('Warm-start provenance differs from frozen study: ' + key)
                expected_updates = 500 if family == 'new_ppo' else 0
                if model.update_count != expected_updates:
                    raise ValueError('Incorrect PPO/cloning checkpoint stage: ' + key)
                expected_episodes = 8000 if family == 'new_ppo' else 0
                if model.episode_count != expected_episodes or model.environment_periods != expected_episodes*scenario.horizon:
                    raise ValueError('Incorrect training budget: ' + key)
            elif model.update_count != 500 or model.episode_count != 8000:
                raise ValueError('Original PPO control has an unexpected training budget')
            used = {2*(c.training_seed_base+c.seed+i) for i in range(model.episode_count)}
            used |= {c.eval_seed_base+2*i for i in range(c.eval_episodes)}
            used |= {2*(c.warmstart_seed_base+c.seed+i) for i in range(c.warmstart_episodes)}
            if used & test_seeds:
                raise ValueError('Training or monitoring overlaps independent tests')
        if model.config.seed != int(seed):
            raise ValueError('Training seed differs from checkpoint label: ' + key)
        if model.scenario != scenario:
            raise ValueError('Checkpoint scenario mismatch: ' + key)
    # Hash all fixed models before opening any test scene.
    model_hashes = {key: digest(path) for key, path in paths.items()}
    dest.mkdir()
    write(dest / 'frozen_models.json', model_hashes)
    results = {key: [] for key in ('new_ppo', 'cloning_only', 'original_ppo', 'approximate_dp')}
    for key, path in paths.items():
        family, seed = key.split('/')
        if family == 'approximate_dp':
            policy = FittedValuePolicy.load_checkpoint(path)
            if policy.scenario != scenario:
                raise ValueError('ADP scenario mismatch')
            result = evaluate_policy(scenario, p['test_seeds'], policy.prices, policy.dispatch,
                action_seed=p['action_seed'], label=key)
        else:
            policy = PaperPPOTrainer.load_checkpoint(path)
            if policy.scenario != scenario:
                raise ValueError('PPO scenario mismatch')
            if family == 'new_ppo' and policy.update_count != 500:
                raise ValueError('New PPO must complete the prescribed 500 updates')
            result = policy.evaluate(p['test_seeds'], False, p['action_seed'])
            result['label'] = key
        result.update(training_seed=int(seed), checkpoint_sha256=model_hashes[key])
        if result['invalid_actions']:
            raise RuntimeError('Evaluation emitted invalid actions: ' + key)
        results[family].append(result)
        write(dest / (key.replace('/', '_') + '.json'), result)
        print(f"{key}: profit {result['mean_discounted_profit']:.4f}, recharge {result['mean_charge_grid_kwh']:.2f}", flush=True)
    rule = evaluate_rule(scenario, p['test_seeds'], rule_spec,
        label='inventory_rule', action_seed=p['action_seed'])
    write(dest / 'inventory_rule.json', rule)
    aggregate = {family: average_models(items, family) for family, items in results.items()}
    aggregate['inventory_rule'] = rule
    comparisons = {name: paired_comparison(aggregate['new_ppo'], aggregate[name])
                   for name in ('original_ppo', 'cloning_only', 'inventory_rule', 'approximate_dp')}
    per_seed = {}
    for i, seed in enumerate(SEEDS):
        per_seed[str(seed)] = {name: paired_comparison(results['new_ppo'][i],
            rule if name == 'inventory_rule' else results[name][i])
            for name in ('original_ppo', 'cloning_only', 'inventory_rule', 'approximate_dp')}
    if model_hashes != {key: digest(path) for key, path in paths.items()}:
        raise RuntimeError('A model changed during independent evaluation')
    verify(study)
    analysis = dict(protocol_sha256=digest(study / 'protocol.json'), model_sha256=model_hashes,
        scorecard=[table_row(k, v) for k, v in aggregate.items()], comparisons=comparisons,
        per_seed=per_seed, by_model={k: {str(r['training_seed']): r['mean_discounted_profit'] for r in v} for k, v in results.items()},
        consistent_improvement=all(per_seed[str(s)]['original_ppo']['paired_bootstrap_confidence_interval_95'][0] > 0 for s in SEEDS),
        invalid_actions=sum(r['invalid_actions'] for v in results.values() for r in v)+rule['invalid_actions'],
        policy_episodes=len(p['test_seeds'])*(len(paths)+1),
        uncertainty='Paired scene bootstrap, 5000 resamples; conditional on fixed models, not retraining uncertainty.')
    write(study / 'analysis.json', analysis)
    show(study)


def show(study):
    data = json.loads((study / 'analysis.json').read_text())
    print(f"{'Method':20} {'Profit/day':>12} {'Recharge kWh':>14} {'Mobile kWh':>12}")
    for row in data['scorecard']:
        print(f"{row['policy']:20} {row['mean_discounted_profit']:12.2f} {row['mean_charge_grid_kwh']:14.2f} {row['mean_mobile_delivered_kwh']:12.2f}")
    print('Consistent improvement over original PPO:', data['consistent_improvement'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'train', 'evaluate', 'show'))
    parser.add_argument('--study-dir', type=Path, default=Path('outputs/recharge_study'))
    parser.add_argument('--control-root', type=Path, default=Path('outputs/paper_ppo_study'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == 'init': init(args.study_dir, args.control_root)
    elif args.phase == 'train': train(args.study_dir, args.seed)
    elif args.phase == 'evaluate': evaluate(args.study_dir)
    else: show(args.study_dir)


if __name__ == '__main__':
    main()
