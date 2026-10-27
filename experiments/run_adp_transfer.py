"""Portability checks for rollout ADP on two different synthetic operating models.

These are known-model configuration changes, not a test against model error.
All policies use the same fixed price and predeclared parameters. There is no
training or tuning on these test days. Run with --workers 2; use --show to read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from charging_env.config import Scenario
from charging_env.benchmarks.evaluation import evaluate_policy
from charging_env.benchmarks.policies import fixed_prices, service_first_dispatch
from charging_env.benchmarks.scalable import coordinate_inventory_dispatch
from charging_env.rl.fitted_train import _new_run_directory
from charging_env.rl.rollout_adp import RolloutADPConfig, RolloutADPPolicy, evaluate_rollout_policy


def cases():
    """Vary dimensions, horizon, batteries, capacities, prices and demand shape."""
    result = {}
    for name, side, fleet, horizon, hours, stations in (
        ('long_constrained', 3, 5, 18, 1., (0, 4, 8)),
        ('wide_short', 6, 12, 8, .5, (0, 14, 35)),
    ):
        xy = tuple((2.5*x, 2.5*y) for y in range(side) for x in range(side))
        rates = tuple(tuple(float((1.8+.25*(z%3)) *
            (1.6 if (z//side < side//2) == (t < horizon//2) else .65) * hours)
            for z in range(side*side)) for t in range(horizon))
        c = replace(Scenario(), horizon=horizon, period_hours=hours, zone_xy_km=xy,
            fcs_zones=stations, initial_locations=tuple(stations[(i//4)%len(stations)]
                if i%4 == 0 else (2+3*i)%(side*side) for i in range(fleet)),
            initial_energy_kwh=tuple((5., 25., 55., 80.)[i%4] for i in range(fleet)),
            arrival_rates=rates, fcs_retail_per_kwh=(.38, .48, .43),
            grid_buy_per_kwh=tuple(.18 if t < horizon//2 else .28 for t in range(horizon)),
            grid_sell_per_kwh=tuple(.10 if t < horizon//2 else .17 for t in range(horizon)),
            fcs_ev_capacity_kwh=(30., 45., 25.),
            fcs_fleet_exchange_capacity_kwh=(15., 25., 20.),
            mobile_charge_kw=20., charging_utility_intercept=3.,
            price_noise_distribution='normal', price_noise_scale=.15,
            terminal_value_per_stored_kwh=.08 if name == 'wide_short' else 0.)
        result[name] = c.validate()
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(result, seeds):
    if result['evaluation_seeds'] != seeds or result['invalid_actions']:
        raise ValueError('Invalid evaluation scenarios or actions')
    for row in result['episodes']:
        terms = row['discounted_reward_terms']
        value = sum(terms[k] for k in ('mobile_service_revenue','fixed_service_revenue',
            'discharge_revenue','terminal_inventory_value'))-sum(terms[k] for k in
            ('procurement_cost','travel_cost','operating_cost','unmet_demand_cost'))
        if not np.isclose(value, row['discounted_profit'], rtol=0, atol=1e-7):
            raise ValueError('Profit ledger does not reconcile')


def job(arguments):
    case, family, seed, destination, seeds = arguments
    scenario = cases()[case]
    directory = Path(destination)/case
    name = f'rollout_{seed}' if family == 'rollout' else family
    if family == 'rollout':
        policy = RolloutADPPolicy(scenario, RolloutADPConfig(seed=seed, price_index=5))
        result = evaluate_rollout_policy(policy, seeds, action_seed=719991, label=name)
        write(directory/(name+'_trace.json'), policy.decisions)
        write(directory/(name+'_provenance.json'), policy.provenance())
    else:
        def dispatch(obs, rng):
            if family == 'service_first':
                return service_first_dispatch(obs, scenario, .2)
            shadow = 0. if family == 'current_profit' else (.3 if int(obs['period'][0]) < scenario.horizon//2 else .15)
            return coordinate_inventory_dispatch(obs, scenario, shadow, sweeps=2, energy_fractions=(1.,))
        result = evaluate_policy(scenario, seeds, lambda obs,rng: fixed_prices(obs,scenario,5),
            dispatch, action_seed=719991, label=name)
    verify(result, seeds)
    write(directory/(name+'.json'), result)
    return case, name, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/adp_transfer'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    if args.show:
        print((args.output_dir/'scorecard.txt').read_text())
        return 0
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    directory = _new_run_directory(args.output_dir)
    scenarios = cases()
    sources = RolloutADPPolicy(scenarios['long_constrained']).provenance()['source_sha256']
    sources['experiments/run_adp_transfer.py'] = digest(Path(__file__))
    banks = {name: [base+2*i for i in range(32)] for name,base in
             [('long_constrained',705000001), ('wide_short',707000001)]}
    protocol = dict(algorithm='online_policy_rollout_adp', offline_training=False,
        interpretation='Known-model portability; not robustness to model error or real-city validation',
        planning_seeds=[7,19,37], rollout_samples=8, fixed_price_index=5,
        controls='service_first threshold .2; inventory .3/.15 and current_profit shadow 0 use two coordinate sweeps',
        action_seed=719991, test_seeds=banks, scenarios={k:asdict(v) for k,v in scenarios.items()},
        source_sha256=sources, test_episodes_per_policy=32, tuning=False)
    write(directory/'protocol.json', protocol)
    for name, scenario in scenarios.items():
        (directory/name).mkdir()
        scenario.to_json(directory/name/'scenario.json')
    tasks = [(case,family,seed,str(directory),banks[case]) for case in scenarios
             for family,seed in [('rollout',7),('rollout',19),('rollout',37),
                 ('inventory',0),('service_first',0),('current_profit',0)]]
    results = {case:{} for case in scenarios}
    print('Frozen portability protocol:', directory, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for case, name, result in executor.map(job, tasks):
            results[case][name] = result
            print(case, name, round(result['mean_discounted_profit'],2), flush=True)
    if any(digest(ROOT/name) != value for name,value in sources.items()):
        raise ValueError('Source changed during portability evaluation')
    analysis = dict(policy_episodes=sum(r['n_episodes'] for rows in results.values() for r in rows.values()),
        invalid_actions=sum(r['invalid_actions'] for rows in results.values() for r in rows.values()), cases={})
    lines = ['Known-model portability checks; fixed price 0.80, 32 independent scenarios per case.']
    for case, rows in results.items():
        average = np.mean([[e['discounted_profit'] for e in rows[f'rollout_{s}']['episodes']]
                          for s in (7,19,37)], axis=0)
        data = dict(profits={name:r['mean_discounted_profit'] for name,r in rows.items()},
            rollout_mean=float(average.mean()), paired_differences={},
            mean_dispatch_ms={name:r['mean_policy_decision_ms']['dispatch'] for name,r in rows.items()})
        for control in ('inventory','service_first','current_profit'):
            delta = average-np.array([e['discounted_profit'] for e in rows[control]['episodes']])
            sem = float(delta.std(ddof=1)/np.sqrt(len(delta)))
            data['paired_differences'][control] = dict(mean=float(delta.mean()),
                confidence_interval_95=[float(delta.mean()-1.96*sem),float(delta.mean()+1.96*sem)])
        analysis['cases'][case] = data
        lines.append(f"{case}: rollout mean {data['rollout_mean']:.2f}; "+'; '.join(f'{k} {v:.2f}' for k,v in data['profits'].items()))
    analysis['protocol_sha256'] = digest(directory/'protocol.json')
    analysis['evidence_sha256'] = {str(p.relative_to(directory)):digest(p) for case in scenarios
        for p in (directory/case).glob('*.json')}
    write(directory/'analysis.json', analysis)
    (directory/'scorecard.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
