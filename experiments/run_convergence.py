"""Reproduce one seed of the predeclared longer-training experiment.

Run from repository root: python experiments/run_convergence.py --seed 7
Test episodes are evaluated separately after all training finishes.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from charging_env.config import Scenario
from charging_env.rl.trainer import Trainer, TrainingConfig


class BudgetTrainer(Trainer):
    def save_checkpoint(self, path):
        super().save_checkpoint(path)
        # Original control uses 1000*4*12=48000 periods. Preserve the same
        # data-budget checkpoint here (250*16*12), without selecting on tests.
        if Path(path).name == 'checkpoint.pt' and self.update_count == 250:
            super().save_checkpoint(Path(path).with_name('checkpoint_250.pt'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', type=Path, help='Scenario JSON; omitted means the unchanged default scenario')
    parser.add_argument('--dispatch-final-activation', choices=('tanh', 'silu'), default='tanh',
                        help='Only the final dispatch hidden activation; start each variant from scratch')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/convergence_study'))
    args = parser.parse_args(argv)
    try:
        scenario = Scenario.from_json(args.scenario) if args.scenario else Scenario().validate()
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    config = TrainingConfig(
        seed=args.seed, updates=500, episodes_per_update=16, epochs=6,
        minibatch_size=192, learning_rate=3e-4, critic_learning_rate=1e-3,
        value_clip=0., entropy_coef=.005, eval_episodes=32, eval_interval=25,
        eval_seed_base=100001, eval_action_seed=12345, eval_policy='stochastic',
        discount_policy_loss=True,
        dispatch_final_activation=args.dispatch_final_activation,
    )
    trainer = BudgetTrainer(scenario, config)
    trainer.train(args.output_root/f'seed_{args.seed}')


if __name__ == '__main__':
    main()
