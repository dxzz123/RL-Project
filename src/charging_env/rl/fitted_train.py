"""Train a fitted continuation-value policy without PyTorch.

Example: python -m charging_env.rl.fitted_train --scenario configs/utility_b3.json
"""
import argparse
import json
import os
from pathlib import Path

from ..config import Scenario
from .fitted_value import FittedValueConfig, FittedValueTrainer


def _new_run_directory(requested):
    """Atomically claim a new directory, automatically suffixing occupied paths."""
    requested = Path(os.path.abspath(requested))
    directory, suffix = requested, 2
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            directory = requested.with_name(f"{requested.name}_{suffix}")
            suffix += 1
        else:
            return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, help="Scenario JSON; omitted uses the original toy scenario")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fitted_value"),
                        help="Adds _2, _3, ... automatically when the requested directory exists")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=10.)
    parser.add_argument("--samples", "--samples-per-period", dest="samples_per_period", type=int, default=8)
    parser.add_argument("--price-index", type=int, default=4)
    parser.add_argument("--energy-fractions", type=float, nargs="+", default=(.5, 1.))
    parser.add_argument("--max-joint-actions", type=int, default=200_000)
    parser.add_argument("--training-seed-base", type=int, default=130_000_000)
    args = parser.parse_args(argv)
    try:
        scenario = Scenario.from_json(args.scenario) if args.scenario else Scenario().validate()
        config = FittedValueConfig(seed=args.seed, grid_step=args.grid_step,
            samples_per_period=args.samples_per_period, price_index=args.price_index,
            energy_fractions=tuple(args.energy_fractions), max_joint_actions=args.max_joint_actions,
            training_seed_base=args.training_seed_base).validate(scenario)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    trainer = FittedValueTrainer(scenario, config)
    directory = _new_run_directory(args.output_dir)
    print(f"Fitted-value training outputs: {directory}", flush=True)
    result = trainer.train(directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
