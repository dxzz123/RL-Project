"""Train the Joint PPO controller."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from ..config import Scenario
from .fitted_train import _new_run_directory
from .paper_ppo import PaperPPOConfig, PaperPPOTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', type=Path)
    mode.add_argument('--evaluate-only', type=Path)
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='Default outputs/paper_ppo; occupied training paths gain _2, _3, ...')
    for name in ('seed', 'updates', 'episodes_per_update', 'epochs', 'minibatch_size', 'hidden_size',
                 'eval_episodes', 'eval_interval', 'eval_seed_base', 'eval_action_seed',
                 'training_seed_base', 'torch_threads'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=None)
    for name in ('actor_learning_rate', 'critic_learning_rate', 'clip_ratio', 'gae_lambda',
                 'pricing_entropy_coef', 'dispatch_entropy_coef', 'max_grad_norm', 'target_kl'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=None)
    parser.add_argument('--dispatch-critic-target', choices=('dispatch', 'pricing_advantage'), default=None)
    parser.add_argument('--normalize-advantages', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval-policy', choices=('stochastic', 'deterministic'), default=None)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    args = parser.parse_args(argv)
    changes = {name: value for name, value in vars(args).items()
               if name in PaperPPOConfig.__dataclass_fields__ and value is not None}
    try:
        if args.resume or args.evaluate_only:
            if args.scenario:
                parser.error('The saved checkpoint supplies its scenario; --scenario cannot replace it')
            trainer = PaperPPOTrainer.load_checkpoint(args.resume or args.evaluate_only, device=args.device or 'cpu')
            for name in ('seed', 'hidden_size', 'dispatch_critic_target', 'training_seed_base', 'normalize_advantages'):
                if name in changes and changes[name] != getattr(trainer.config, name):
                    parser.error('--'+name.replace('_', '-')+' cannot change for an existing checkpoint')
            trainer.config = replace(trainer.config, **changes).validate()
            torch.set_num_threads(trainer.config.torch_threads)
            for name, optimizer in trainer.optimizers.items():
                for group in optimizer.param_groups:
                    group['lr'] = (trainer.config.actor_learning_rate if name.endswith('actor')
                                   else trainer.config.critic_learning_rate)
        else:
            scenario = Scenario.from_json(args.scenario) if args.scenario else Scenario()
            trainer = PaperPPOTrainer(scenario, PaperPPOConfig(**changes))
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if args.evaluate_only:
        c = trainer.config
        result = trainer.evaluate([c.eval_seed_base+2*i for i in range(c.eval_episodes)],
                                  deterministic=c.eval_policy == 'deterministic')
        if args.output_dir:
            directory = _new_run_directory(args.output_dir)
            (directory/'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    else:
        directory = _new_run_directory(args.output_dir or Path('outputs/paper_ppo'))
        print('Joint PPO outputs: '+str(directory), flush=True)
        result = trainer.train(directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
