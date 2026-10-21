"""Development-only SiLU+b=3 pricing/dispatch execution-swap diagnostics.

Uses fresh paired development scenes, never final-test scenes. Frozen learned
actors and the previously selected rule are not retrained or retuned. Execution
swaps change state visitation and are not retrained ablations or causal effects.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from charging_env.config import Scenario
from charging_env.simulator import Mode
from charging_env.rl.trainer import Trainer
from charging_env.benchmarks.learned import localize_observation
from charging_env.benchmarks.evaluation import evaluate_policy,paired_comparison
from charging_env.benchmarks import policies as rules
from run_policy_benchmarks import average_trained_models

SEEDS = [4100001+2*i for i in range(64)]
ACTION_SEED = 419991
TRAINING_SEEDS = (7,19,37)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    def convert(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=convert)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_exchange(obs,config,agent,action):
    distance=config.distances[int(obs['locations'][agent]),action.target_zone]
    hours=max(0.,config.period_hours-distance/config.speed_kmph)
    energy=obs['energy_kwh'][agent]-distance*config.travel_kwh_per_km
    if action.mode==Mode.RECHARGE:
        return max(0.,min(hours*config.mobile_charge_kw,(config.battery_capacity_kwh-energy)/config.charge_efficiency))
    if action.mode==Mode.DISCHARGE:
        return max(0.,min(hours*config.mobile_discharge_kw,(energy-config.reserve_kwh)*config.discharge_efficiency))
    return 0.


@torch.no_grad()
def evaluate(config,checkpoint,pricing_source,dispatch_source,label):
    records=[]
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        trainer=Trainer.load_checkpoint(checkpoint,device='cpu') if checkpoint else None
        if trainer:
            if trainer.scenario!=config:raise ValueError('Frozen checkpoint scenario must match b=3 exactly')
            for module in trainer._modules().values():module.eval()
        context={}
        def start(seed,action_seed):
            torch.manual_seed(seed+action_seed)
            context.clear();context['seed']=seed
        def pricing(obs,rng):
            if 'predicted_energy' in context:
                np.testing.assert_allclose(obs['energy_kwh'],context['predicted_energy'],rtol=0,atol=1e-8)
            # Consume each trained actor's draws in the original stage order,
            # including when a rule replaces its executed decision in a swap.
            sampled=trainer.pricing_actor.sample([obs]) if trainer else None
            levels=sampled['actions'][0].cpu().numpy() if pricing_source=='rl' else rules.fixed_prices(obs,config,4)
            active=np.asarray(obs['mobile_available'],bool)
            context['quotes']=np.asarray(config.price_levels)[levels[active]].tolist()
            context['quote_indices']=levels[active].tolist()
            context['potential_energy']=np.asarray(obs['potential_energy_kwh'])[active].tolist()
            return levels
        def dispatch(obs,rng):
            local=[localize_observation(obs,m,config.n_mcv) for m in range(config.n_mcv)]
            sampled=trainer.dispatch_actor.sample(local) if trainer else None
            commands=(trainer._dispatch_commands(sampled['actions'].cpu().numpy(),sampled['energy_fraction'].cpu().numpy())
                      if dispatch_source=='rl' else rules.coordinated_dispatch(obs,config,shadow_price=.3))
            outcome=rules.dispatch_outcome(obs,config,commands)
            comparator=rules.coordinated_dispatch(obs,config,shadow_price=.3)
            alternative=rules.dispatch_outcome(obs,config,comparator)
            period=int(obs['period'][0]);shadow=0. if period+1==config.horizon else .3
            score=lambda result:result['reward']+shadow*(np.sum(result['next_energy_kwh'])-np.sum(obs['energy_kwh']))
            distances=np.asarray([config.distances[int(obs['locations'][m]),a.target_zone] if a.mode!=Mode.WAIT else 0. for m,a in enumerate(commands)])
            mobile_by_agent=outcome['delivered_mobile_kwh'].sum(axis=1)
            serve=[m for m,a in enumerate(commands) if a.mode==Mode.SERVE]
            empty=[m for m in serve if mobile_by_agent[m]<=1e-9]
            energy_actions=[m for m,a in enumerate(commands) if a.mode in (Mode.RECHARGE,Mode.DISCHARGE)]
            maxima=[maximum_exchange(obs,config,m,a) for m,a in enumerate(commands)]
            records.append({'seed':context['seed'],'period':period,'reward':outcome['reward'],
                'reward_terms':outcome['reward_terms'],'inventory_before_kwh':float(obs['energy_kwh'].sum()),
                'inventory_after_kwh':float(outcome['next_energy_kwh'].sum()),
                'charge_kwh':float(outcome['charge_grid_kwh'].sum()),'export_kwh':float(outcome['export_grid_kwh'].sum()),
                'mobile_delivered_kwh':float(mobile_by_agent.sum()),'travel_kwh':float(distances.sum()*config.travel_kwh_per_km),
                'mobile_requested_kwh':float(obs['mobile_requested_kwh'].sum()),
                'unmet_mobile_kwh':float(outcome['unmet_mobile_kwh']),
                'same_serve_target':len(set(commands[m].target_zone for m in serve))<len(serve),
                'serve_actions':len(serve),'empty_serve_actions':len(empty),
                'empty_serve_travel_kwh':float(distances[empty].sum()*config.travel_kwh_per_km),
                'energy_actions':len(energy_actions),'clipped_energy_actions':int(sum(commands[m].energy_kwh>maxima[m]+1e-8 for m in energy_actions)),
                'recharge_requests_kwh':float(sum(a.energy_kwh for a in commands if a.mode==Mode.RECHARGE)),
                'inventory_objective_gap':float(score(alternative)-score(outcome)),
                'active_quotes':context['quotes'],'active_quote_indices':context['quote_indices'],
                'active_potential_energy':context['potential_energy'],
                'commands':[{'mode':a.mode.name,'target':a.target_zone,'requested_kwh':a.energy_kwh,
                             'energy_before_kwh':float(obs['energy_kwh'][m]),'physical_exchange_cap_kwh':maxima[m]} for m,a in enumerate(commands)]})
            context['predicted_energy']=outcome['next_energy_kwh']
            return commands
        result=evaluate_policy(config,SEEDS,pricing,dispatch,action_seed=ACTION_SEED,label=label,on_episode_start=start)
        for episode in result['episodes']:
            predicted=sum(row['reward'] for row in records if row['seed']==episode['seed'])
            if not np.isclose(predicted,episode['profit'],rtol=0,atol=1e-8):
                raise AssertionError('Independent public-state diagnostics disagree with actual simulator reward')
        if result['invalid_actions']:raise AssertionError('Diagnostic execution emitted invalid actions')
        result['training_seed']=trainer.config.seed if trainer else None
        result['checkpoint']=str(checkpoint) if checkpoint else None
        result['decisions']=records
        result['period_averages']=[summarize_records([row for row in records if row['period']==period]) for period in range(config.horizon)]
        result['behavior']=summarize_records(records)
        return result


def summarize_records(records):
    fields=['inventory_before_kwh','inventory_after_kwh','charge_kwh','export_kwh','mobile_delivered_kwh',
            'travel_kwh','mobile_requested_kwh','unmet_mobile_kwh','same_serve_target','serve_actions','empty_serve_actions',
            'empty_serve_travel_kwh','inventory_objective_gap','recharge_requests_kwh']
    result={key:float(np.mean([row[key] for row in records])) for key in fields}
    result['reward_terms']={key:float(np.mean([row['reward_terms'][key] for row in records])) for key in records[0]['reward_terms']}
    quotes=[quote for row in records for quote in row['active_quotes']]
    weights=[energy for row in records for energy in row['active_potential_energy']]
    result.update(active_quote_mean=float(np.mean(quotes)) if quotes else None,
        active_quote_potential_energy_weighted_mean=float(np.average(quotes,weights=weights)) if quotes and sum(weights)>0 else None,
        active_quote_histogram=np.bincount([i for row in records for i in row['active_quote_indices']],minlength=7).tolist(),
        empty_serve_fraction=sum(row['empty_serve_actions'] for row in records)/max(1,sum(row['serve_actions'] for row in records)),
        clipped_energy_fraction=sum(row['clipped_energy_actions'] for row in records)/max(1,sum(row['energy_actions'] for row in records)),
        realized_mobile_price=sum(row['reward_terms']['mobile_service_revenue'] for row in records)/max(1e-12,sum(row['mobile_delivered_kwh'] for row in records)),
        realized_export_price=sum(row['reward_terms']['discharge_revenue'] for row in records)/max(1e-12,sum(row['export_kwh'] for row in records)),
        mode_counts={mode.name:sum(action['mode']==mode.name for row in records for action in row['commands']) for mode in Mode})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/b3_gap_diagnostics')
    args=parser.parse_args(argv)
    if args.output_dir.exists():parser.error('Choose a new output directory; diagnostics are never overwritten')
    torch.set_num_threads(1)
    scenario=Scenario.from_json(ROOT/'configs/utility_b3.json')
    paths={seed:args.checkpoint_root/f'seed_{seed}/checkpoint.pt' for seed in TRAINING_SEEDS}
    hashes={str(seed):digest(path) for seed,path in paths.items()}
    protocol={'scenario':asdict(scenario),'development_seeds':SEEDS,'action_seed':ACTION_SEED,
        'checkpoint_paths':{str(seed):str(path) for seed,path in paths.items()},'checkpoint_sha256':hashes,
        'rule':{'price_index':4,'price':.6,'shadow_price':.3},
        'purpose':'Development diagnostics only; not final-test evidence, retraining ablations, or tuning of the frozen rule.',
        'rng':'Common evaluate_policy harness; episode-local Torch seed action_seed+environment_seed; both trained actors sampled in stage order even for execution swaps.',
        'inventory_objective_gap':'Rule minus executed current reward plus0.30*stored-energy change, zero artificial shadow at terminal; heuristic objective, not full-horizon regret.'}
    save(args.output_dir/'protocol.json',protocol)
    results={}
    label='rule_pricing_rule_dispatch'
    results[label]=evaluate(scenario,None,'rule','rule',label)
    save(args.output_dir/f'{label}.json',results[label]);print(label,results[label]['mean_discounted_profit'],flush=True)
    for seed,path in paths.items():
        for pricing,dispatch in [('rl','rl'),('rl','rule'),('rule','rl')]:
            label=f'{pricing}_pricing_{dispatch}_dispatch_seed_{seed}'
            results[label]=evaluate(scenario,path,pricing,dispatch,label)
            save(args.output_dir/f'{label}.json',results[label])
            print(label,results[label]['mean_discounted_profit'],flush=True)
    averages={}
    for pricing,dispatch in [('rl','rl'),('rl','rule'),('rule','rl')]:
        key=f'{pricing}_pricing_{dispatch}_dispatch'
        models=[results[f'{key}_seed_{seed}'] for seed in TRAINING_SEEDS]
        average=average_trained_models(models,key)
        average['behavior_pooled']=summarize_records([row for model in models for row in model['decisions']])
        average['period_averages_pooled']=[summarize_records([row for model in models for row in model['decisions'] if row['period']==period]) for period in range(scenario.horizon)]
        average['comparison_to_rule']=paired_comparison(average,results['rule_pricing_rule_dispatch'])
        averages[key]=average
    if any(digest(paths[int(seed)])!=value for seed,value in hashes.items()):raise RuntimeError('A frozen model changed')
    rule={key:value for key,value in results['rule_pricing_rule_dispatch'].items() if key not in ('decisions','episodes')}
    raw=averages['rl_pricing_rl_dispatch']['mean_discounted_profit']
    summary={'rule':rule,'model_averages':averages,'execution_swap_changes_from_original_rl':
        {key:value['mean_discounted_profit']-raw for key,value in averages.items() if key!='rl_pricing_rl_dispatch'},
        'diagnostic_policy_episodes':len(SEEDS)*len(results),'invalid_actions':sum(result['invalid_actions'] for result in results.values()),
        'limitations':'Execution swaps alter the visited state distribution. They do not identify standalone retrained module contributions or full-horizon optimality.'}
    save(args.output_dir/'summary.json',summary)
    print('Complete:',args.output_dir/'summary.json',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
