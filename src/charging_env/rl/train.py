"""Train, resume, or evaluate SA-PPO–MAPPO: python -m charging_env.rl.train."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from ..config import Scenario
from .trainer import Trainer, TrainingConfig, _write_json


def _new_run_directory(requested):
    """Claim an unused directory so repeated CLI runs keep separate artifacts."""
    requested = Path(os.path.abspath(requested))
    directory = requested
    suffix = 2
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            directory = requested.with_name(f'{requested.name}_{suffix}')
            suffix += 1
        else:
            return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', type=Path, help='Resume saved networks, optimizers, counters and RNG in a new output directory')
    mode.add_argument('--evaluate-only', type=Path, help='Evaluate a saved checkpoint without training')
    parser.add_argument('--scenario', type=Path, help='Scenario JSON; omitted uses the default toy city')
    parser.add_argument('--output-dir', type=Path,
                        help='Training directory (default: outputs/sa_ppo_mappo); adds _2, _3, ... if occupied. '
                             'Evaluation-only saves evaluation.json here without replacing an existing file')
    for flag in ('updates', 'episodes-per-update', 'epochs', 'minibatch-size', 'hidden-size',
                 'seed', 'eval-episodes', 'eval-interval', 'eval-seed-base',
                 'eval-action-seed', 'torch-threads', 'training-seed-base', 'dispatch-entropy-anneal-updates'):
        parser.add_argument('--'+flag, type=int, default=None)
    for flag in ('learning-rate', 'critic-learning-rate', 'clip-ratio', 'value-clip', 'gae-lambda', 'value-coef',
                 'entropy-coef', 'max-grad-norm', 'target-kl', 'inventory-potential',
                 'dispatch-entropy-coef', 'dispatch-entropy-final-coef'):
        parser.add_argument('--'+flag, type=float, default=None)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    parser.add_argument('--eval-policy', choices=('deterministic', 'stochastic'), default=None,
                        help='Evaluation action selection and periodic best-checkpoint criterion')
    parser.add_argument('--dispatch-final-activation', choices=('tanh', 'silu'), default=None,
                        help='Dispatch actor final hidden activation; new architecture trials must start from scratch')
    parser.add_argument('--discount-policy-loss', action=argparse.BooleanOptionalAction, default=None,
                        help='Weight both stage policy losses by the physical-period discount')
    parser.add_argument('--dispatch-agent-id', action=argparse.BooleanOptionalAction, default=None,
                        help='Include vehicle role identity in the shared local dispatch policy')
    parser.add_argument('--encoder-layer-norm', action=argparse.BooleanOptionalAction, default=None,
                        help='Normalize dense encoder hidden features before their nonlinearities')
    parser.add_argument('--separate-network-grad-clip', action=argparse.BooleanOptionalAction, default=None,
                        help='Clip each actor and critic independently')
    args = parser.parse_args(argv)
    names = set(TrainingConfig.__dataclass_fields__)
    changes = {key: value for key, value in vars(args).items() if key in names and value is not None}
    checkpoint = args.resume or args.evaluate_only
    if checkpoint:
        if args.scenario is not None:
            parser.error('The checkpoint includes its scenario; --scenario cannot replace it')
        trainer = Trainer.load_checkpoint(checkpoint, device=args.device or 'cpu')
        for name in ('hidden_size', 'seed', 'dispatch_final_activation', 'dispatch_agent_id', 'encoder_layer_norm',
                     'inventory_potential', 'training_seed_base'):
            if name in changes and changes[name] != getattr(trainer.config, name):
                parser.error(f'--{name.replace("_", "-")} cannot change for a saved checkpoint')
        trainer.config = replace(trainer.config, **changes).validate()
        if args.learning_rate is not None:
            for group in trainer.actor_optimizer.param_groups:
                group['lr'] = args.learning_rate
        if (args.critic_learning_rate is not None
                or (args.learning_rate is not None and trainer.config.critic_learning_rate is None)):
            critic_rate = (trainer.config.critic_learning_rate
                           if trainer.config.critic_learning_rate is not None else trainer.config.learning_rate)
            for group in trainer.critic_optimizer.param_groups:
                group['lr'] = critic_rate
        if args.torch_threads is not None:
            import torch
            torch.set_num_threads(args.torch_threads)
    else:
        scenario = Scenario.from_json(args.scenario) if args.scenario else Scenario()
        trainer = Trainer(scenario, TrainingConfig(**changes))
    if args.evaluate_only:
        seeds = [trainer.config.eval_seed_base+2*i for i in range(trainer.config.eval_episodes)]
        result = trainer.evaluate(seeds, deterministic=trainer.config.eval_policy == 'deterministic',
                                  action_seed=trainer.config.eval_action_seed)
        if args.output_dir:
            target = args.output_dir/'evaluation.json'
            if target.exists():
                parser.error(f'{target} already exists; choose a new output directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_json(target, result)
    else:
        directory = _new_run_directory(args.output_dir or Path('outputs/sa_ppo_mappo'))
        print(f'Training outputs: {directory}', flush=True)
        result = trainer.train(directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
