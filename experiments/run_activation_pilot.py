"""Bounded paired pilot: dispatch final hidden activation, tanh versus SiLU.

Only the dispatch encoder's final hidden activation changes. All other settings
match the existing convergence configuration, with a fixed shorter budget and
new validation/test seeds. Both final models are frozen before testing. This is
one training seed and does not establish convergence or retraining robustness.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator
from charging_env.rl.trainer import Trainer, TrainingConfig
from charging_env.benchmarks import policies as rules
from charging_env.benchmarks.learned import evaluate_checkpoint, localize_observation
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison

ACTIVATIONS = ('tanh', 'silu')
VALIDATION_SEEDS = [2300001+2*i for i in range(32)]
TEST_SEEDS = [2500001+2*i for i in range(256)]
DIAGNOSTIC_SEEDS = [2700001+2*i for i in range(32)]
VALIDATION_ACTION_SEED = 173971
TEST_ACTION_SEED = 193939


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def write_json(path, value, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if immutable else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=json_default)
        stream.write('\n')


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def config_for(seed, updates, activation):
    return TrainingConfig(
        seed=seed, updates=updates, episodes_per_update=16, epochs=6,
        minibatch_size=192, hidden_size=64, learning_rate=3e-4,
        critic_learning_rate=1e-3, value_clip=0., entropy_coef=.005,
        eval_episodes=len(VALIDATION_SEEDS), eval_interval=25,
        eval_seed_base=VALIDATION_SEEDS[0], eval_action_seed=VALIDATION_ACTION_SEED,
        eval_policy='stochastic', discount_policy_loss=True,
        dispatch_final_activation=activation,
    ).validate()


def initial_tensor_hashes(trainer):
    """Hash tensor contents, names, shapes, and dtypes without pickle metadata."""
    hashes = {}
    for name, model in trainer._modules().items():
        hasher = hashlib.sha256()
        for key, value in model.state_dict().items():
            value = value.detach().cpu().contiguous()
            hasher.update(f'{key}|{value.dtype}|{tuple(value.shape)}'.encode())
            hasher.update(value.numpy().tobytes())
        hashes[name] = hasher.hexdigest()
    return hashes


def rule_callbacks(scenario, specification):
    pricing = lambda obs, rng: rules.fixed_prices(obs, scenario, specification['price_index'])
    dispatch = lambda obs, rng: rules.coordinated_dispatch(
        obs, scenario, shadow_price=specification['shadow_price'])
    return pricing, dispatch


def diagnostic_state_bank(scenario, specification):
    """Generate identical held-out local inputs, independently of either actor."""
    pricing, dispatch = rule_callbacks(scenario, specification)
    records = []
    for seed in DIAGNOSTIC_SEEDS:
        sim = ChargingSimulator(scenario)
        obs, _ = sim.reset(seed=seed)
        while sim.stage != 2:
            period = int(obs['period'][0])
            post, _, _, _, _ = sim.quote_price_levels(pricing(obs, None))
            for agent in range(scenario.n_mcv):
                records.append({'environment_seed':seed, 'period':period, 'agent':agent,
                                'observation':localize_observation(post, agent, scenario.n_mcv)})
            obs, _, _, _, info = sim.dispatch(dispatch(post, None))
            if info['invalid_actions']:
                raise AssertionError('Frozen diagnostic rule emitted invalid actions')
    return records


@torch.no_grad()
def responsiveness(actor, state_bank, bank_hash):
    """Separate hidden saturation and raw-score response from mask-only changes.

    SiLU is unbounded, so |h| > .99 is not used as a saturation criterion.
    State-response metrics center logits within each state to remove common
    offsets, which do not change categorical probabilities. These diagnostics
    describe the frozen rule's state distribution, not a trained actor's own
    visitation distribution and not a causal test of the activation mechanism.
    """
    final = actor.encoder.body[-1]
    if not isinstance(final, (nn.Tanh, nn.SiLU)):
        raise ValueError('Expected final dispatch hidden activation tanh or SiLU')
    capture, preactivations, features, raw_scores, energy_params, masks = {}, [], [], [], [], []
    hook = final.register_forward_pre_hook(lambda module, inputs: capture.update(pre=inputs[0].detach()))
    was_training = actor.training
    actor.eval()
    try:
        for start in range(0, len(state_bank), 128):
            observations = [row['observation'] for row in state_bank[start:start+128]]
            hidden = actor.encoder(observations)
            preactivations.append(capture['pre'].cpu())
            features.append(hidden.cpu())
            raw_scores.append(actor.action_head(hidden).cpu())
            raw_energy = actor.energy_head(hidden).reshape(-1, actor.n_actions, 2)
            energy_params.append((F.softplus(raw_energy)+1.).cpu())
            masks.append(torch.as_tensor(np.stack([o['action_mask'] for o in observations]),
                                         dtype=torch.bool).reshape(len(observations), -1))
    finally:
        hook.remove()
        actor.train(was_training)
    z, h, raw, energy, mask = [torch.cat(parts,dim=0).double() for parts in
                              (preactivations,features,raw_scores,energy_params,masks)]
    mask = mask.bool()
    centered = raw-raw.mean(dim=1,keepdim=True)
    probabilities = raw.masked_fill(~mask,-torch.inf).softmax(dim=1)
    static_scores = raw.mean(dim=0,keepdim=True).expand_as(raw)
    mask_only = static_scores.masked_fill(~mask,-torch.inf).softmax(dim=1)
    if isinstance(final, nn.Tanh):
        derivative = 1.-z.tanh().square()
        saturation = float((z.tanh().abs()>.99).double().mean())
        name = 'tanh'
    else:
        sigmoid = z.sigmoid()
        derivative = sigmoid+z*sigmoid*(1.-sigmoid)
        saturation, name = None, 'silu'
    energy_mean = energy[...,0]/energy.sum(dim=-1)
    energy_variance = energy[...,0]*energy[...,1]/(energy.sum(dim=-1).square()*(energy.sum(dim=-1)+1.))
    energy_mask = mask & actor.energy_actions.cpu().bool()[None,:]
    branch_sd = [energy_mean[energy_mask[:,a],a].std(unbiased=False).item()
                 for a in range(actor.n_actions) if int(energy_mask[:,a].sum())>=2]
    valid_energy = energy_mean[energy_mask]
    def rms(value):
        return float(value.square().mean().sqrt())
    def state_rms(value):
        return rms(value-value.mean(dim=0,keepdim=True))
    return {
        'activation':name, 'state_bank_sha256':bank_hash, 'n_local_states':len(state_bank),
        'preactivation_rms':rms(z), 'feature_rms':rms(h),
        'tanh_saturation_share_abs_activation_above_0_99':saturation,
        'absolute_activation_derivative_mean':float(derivative.abs().mean()),
        'small_gradient_share_abs_derivative_below_0_01':float((derivative.abs()<.01).double().mean()),
        'feature_across_state_sd_mean':float(h.std(dim=0,unbiased=False).mean()),
        'centered_raw_logit_variance_within_state_mean':float(centered.square().mean()),
        'centered_raw_score_across_state_rms':state_rms(centered),
        'masked_probability_across_state_rms':state_rms(probabilities),
        'mask_only_probability_across_state_rms':state_rms(mask_only),
        'actual_minus_mask_only_probability_rms':rms(probabilities-mask_only),
        'mean_valid_discrete_actions':float(mask.sum(dim=1).double().mean()),
        'masked_categorical_entropy_mean':float(-(probabilities*probabilities.clamp_min(1e-300).log()).sum(dim=1).mean()),
        'conditional_energy_mean_fraction_on_valid_branches':float(valid_energy.mean()) if valid_energy.numel() else None,
        'conditional_energy_mean_fraction_min':float(valid_energy.min()) if valid_energy.numel() else None,
        'conditional_energy_mean_fraction_max':float(valid_energy.max()) if valid_energy.numel() else None,
        'conditional_energy_mean_across_state_sd_mean_by_valid_branch':float(np.mean(branch_sd)) if branch_sd else None,
        'conditional_energy_distribution_variance_mean':float(energy_variance[energy_mask].mean()) if energy_mask.any() else None,
        'scope':'Identical frozen-rule local states; descriptive response, not causal evidence or hyperparameter selection.',
        'metric_definitions':{
            'centered_raw_scores':'Unmasked logits minus their action mean within each state.',
            'across_state_rms':'RMS deviation from each action/feature column mean across states.',
            'mask_only':'Current feasible mask applied to one constant vector of mean raw logits across all bank states.',
            'energy_branch_sd':'Per-action predicted Beta-mean standard deviation over states where that energy branch is feasible, then averaged across branches.',
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/activation_pilot')
    parser.add_argument('--selection-path',type=Path,required=True,
                        help='Existing frozen outputs/policy_benchmarks/selection.json')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--updates',type=int,default=100,help='Fixed budget, between 1 and 100; no adaptive extension')
    parser.add_argument('--preflight-only',action='store_true',help='Validate inputs without training or evaluating scenes')
    args = parser.parse_args(argv)
    if not 1<=args.updates<=100:
        parser.error('This bounded pilot permits 1 through 100 updates, fixed before execution')
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Choose a new output directory; previous evidence is never overwritten')
    scenario = Scenario().validate()
    configs = {activation:config_for(args.seed,args.updates,activation) for activation in ACTIVATIONS}
    settings = {key:asdict(config) for key,config in configs.items()}
    differences = [key for key in settings['tanh'] if settings['tanh'][key]!=settings['silu'][key]]
    if differences != ['dispatch_final_activation']:
        raise AssertionError('The arms must differ only in dispatch_final_activation')
    selection = json.loads(args.selection_path.read_text())
    specification = selection['family_winners']['fixed_inventory']
    if specification['family']!='fixed_inventory':
        parser.error('Expected a previously selected fixed_inventory benchmark')
    source_files = sorted((ROOT/'src/charging_env').rglob('*.py'))+[Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(ROOT)):digest(path) for path in source_files}
    protocol = {
        'experiment':'Paired final-dispatch-hidden activation pilot',
        'training_seed':args.seed, 'updates_per_arm':args.updates,
        'training_physical_periods_per_arm':args.updates*16*scenario.horizon,
        'only_configuration_difference':differences, 'configuration':settings, 'scenario':asdict(scenario),
        'validation_seeds':VALIDATION_SEEDS,'validation_action_seed':VALIDATION_ACTION_SEED,
        'test_seeds':TEST_SEEDS,'test_action_seed':TEST_ACTION_SEED,
        'diagnostic_seeds':DIAGNOSTIC_SEEDS,
        'primary_comparison':'Final stochastic SiLU minus final stochastic tanh, paired by scenario.',
        'secondary_comparison':'Final deterministic execution; both arms also compared with the frozen inventory rule.',
        'selection':'Fixed budget and configurations. Neither checkpoint selection nor hyperparameter tuning uses tests. Validation-best files are training artifacts and are not evaluated on the test split.',
        'rule_selection_path':str(args.selection_path.resolve()),'rule_selection_sha256':digest(args.selection_path),
        'frozen_rule_specification':specification,'source_sha256':source_hashes,
        'pairing':'Identical initial tensor contents, training environment seed schedule, optimizer settings, validation and test scenarios. Actions and visited states may diverge because the policies differ.',
        'uncertainty_scope':'Paired scene bootstrap conditional on these two fixed models from one training seed; no estimate of retraining variability.',
        'diagnostic_scope':'Both activations before/after use the same separate local-state bank generated by the frozen rule. Diagnostics do not select configurations or checkpoints.',
    }
    if args.preflight_only:
        print(json.dumps(protocol,indent=2,allow_nan=False))
        return 0
    started = time.monotonic()
    output.mkdir(parents=True)
    write_json(output/'protocol.json',protocol,immutable=True)
    protocol_hash = digest(output/'protocol.json')
    bank = diagnostic_state_bank(scenario,specification)
    write_json(output/'diagnostic_states.json',bank,immutable=True)
    bank_hash = digest(output/'diagnostic_states.json')
    expected_initial, diagnostics, summaries = None, {}, {}
    for activation in ACTIVATIONS:
        trainer = Trainer(scenario,configs[activation])
        initial = initial_tensor_hashes(trainer)
        if expected_initial is None:
            expected_initial = initial
        elif initial!=expected_initial:
            raise AssertionError('Paired arms did not start from identical tensor contents')
        diagnostics[activation] = {'initial_tensor_sha256':initial,
                                  'before':responsiveness(trainer.dispatch_actor,bank,bank_hash)}
        run = output/activation
        run.mkdir()
        write_json(run/'responsiveness_before.json',diagnostics[activation]['before'],immutable=True)
        trainer.save_checkpoint(run/'initial_checkpoint.pt')
        with (run/'train.log').open('x',encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout,log)):
            summaries[activation] = trainer.train(run)
        diagnostics[activation]['after'] = responsiveness(trainer.dispatch_actor,bank,bank_hash)
        write_json(run/'responsiveness_after.json',diagnostics[activation]['after'],immutable=True)
        expected_periods = protocol['training_physical_periods_per_arm']
        if trainer.environment_periods!=expected_periods or trainer.update_count!=args.updates:
            raise AssertionError('Training exceeded or failed to meet the precommitted fixed budget')
    checkpoints = {activation:output/activation/'checkpoint.pt' for activation in ACTIVATIONS}
    frozen = {'checkpoint_sha256':{key:digest(path) for key,path in checkpoints.items()},
              'checkpoint_paths':{key:str(path) for key,path in checkpoints.items()},
              'protocol_sha256':protocol_hash,'diagnostic_state_bank_sha256':bank_hash,
              'initial_tensor_contents_identical':True, 'training_summaries':summaries}
    write_json(output/'frozen_models.json',frozen,immutable=True)  # Before the first test scene.
    if any(digest(ROOT/path)!=value for path,value in source_hashes.items()):
        raise RuntimeError('Frozen source changed during training; do not run final tests')
    pricing,dispatch = rule_callbacks(scenario,specification)
    benchmark = evaluate_policy(scenario,TEST_SEEDS,pricing,dispatch,
                                action_seed=TEST_ACTION_SEED,label='fixed_inventory')
    write_json(output/'test/fixed_inventory.json',benchmark,immutable=True)
    tests = {}
    for deterministic in (False,True):
        mode = 'deterministic' if deterministic else 'stochastic'
        tests[mode] = {}
        for activation in ACTIVATIONS:
            label = f'{activation}_final_{mode}'
            result = evaluate_checkpoint(scenario,TEST_SEEDS,checkpoints[activation],
                deterministic=deterministic,action_seed=TEST_ACTION_SEED,label=label)
            if result['invalid_actions']:
                raise AssertionError('A trained policy emitted invalid test actions')
            tests[mode][activation] = result
            write_json(output/f'test/{label}.json',result,immutable=True)
            print(f'{label}: {result["mean_discounted_profit"]:.4f}',flush=True)
    comparisons = {mode:{
        'silu_minus_tanh':paired_comparison(results['silu'],results['tanh']),
        'versus_inventory_rule':{key:paired_comparison(result,benchmark) for key,result in results.items()},
    } for mode,results in tests.items()}
    if benchmark['invalid_actions']:
        raise AssertionError('The frozen rule emitted invalid test actions')
    if any(digest(checkpoints[key])!=value for key,value in frozen['checkpoint_sha256'].items()):
        raise RuntimeError('A frozen final checkpoint changed during testing')
    if (digest(output/'protocol.json')!=protocol_hash or digest(output/'diagnostic_states.json')!=bank_hash
            or digest(args.selection_path)!=protocol['rule_selection_sha256']
            or any(digest(ROOT/path)!=value for path,value in source_hashes.items())):
        raise RuntimeError('Frozen protocol, state bank, selection, or source changed during execution')
    def compact(result):
        return {key:value for key,value in result.items() if key not in ('episodes','evaluation_seeds')}
    summary = {'protocol_sha256':protocol_hash,'fixed_budget_completed':True,
               'initial_tensor_contents_identical':True,'responsiveness':diagnostics,
               'test':{mode:{key:compact(value) for key,value in values.items()} for mode,values in tests.items()},
               'fixed_inventory_test':compact(benchmark),'comparisons':comparisons,
               'primary_comparison':comparisons['stochastic']['silu_minus_tanh'],
               'invalid_test_actions':0,'test_tuning_performed':False,
               'test_policy_episodes':len(TEST_SEEDS)*(1+len(ACTIVATIONS)*2),
               'elapsed_seconds':time.monotonic()-started,
               'limitation':'One training seed and a bounded short pilot; no claim of stable convergence or general superiority.'}
    write_json(output/'summary.json',summary,immutable=True)
    print('Complete: '+str(output/'summary.json'),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
