"""Optional PyTorch adapter: evaluate frozen actors through the common harness."""
from pathlib import Path

import numpy as np
import torch

from ..rl.trainer import Trainer
from .evaluation import evaluate_policy


def localize_observation(observation, agent, n_mcv):
    """Match ChargingSimulator.local_observation using public global fields only."""
    local = {key: value.copy() if isinstance(value, np.ndarray) else value
             for key, value in observation.items()}
    peers = np.arange(n_mcv) != agent
    local['agent_id'] = np.array([agent], dtype=np.int64)
    local['other_locations'] = observation['locations'][peers].copy()
    local['other_energy_kwh'] = observation['energy_kwh'][peers].copy()
    for key in ('locations', 'energy_kwh', 'action_mask'):
        local[key] = observation[key][agent:agent+1].copy()
    return local


@torch.no_grad()
def evaluate_checkpoint(config, seeds, checkpoint, *, deterministic=False,
                        action_seed=54123, label=''):
    """Keep training-time feature scales even when evaluating a shifted scenario.

    This adapter supports demand, price-noise, and initial-inventory shifts with
    unchanged physical topology/action shapes. It never recalibrates encoders or
    refits weights on a test scenario, and restores the caller's Torch RNG.
    """
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        trainer = Trainer.load_checkpoint(checkpoint, device='cpu')
        original = trainer.scenario
        permitted = {'mean_requests_by_zone', 'arrival_profile', 'arrival_rates',
                     'initial_energy_kwh', 'price_noise_distribution', 'price_noise_scale'}
        for field in original.__dataclass_fields__:
            if field not in permitted and getattr(original, field) != getattr(config, field):
                raise ValueError(f'Unsupported checkpoint scenario shift: {field}')
        for model in trainer._modules().values():
            model.eval()

        def prices(obs, rng):
            result = trainer.pricing_actor.sample([obs], deterministic=deterministic)
            return result['actions'][0].cpu().numpy()

        def dispatch(obs, rng):
            local = [localize_observation(obs, m, config.n_mcv) for m in range(config.n_mcv)]
            result = trainer.dispatch_actor.sample(local, deterministic=deterministic)
            return trainer._dispatch_commands(result['actions'].cpu().numpy(),
                                               result['energy_fraction'].cpu().numpy())

        result = evaluate_policy(config, seeds, prices, dispatch, action_seed=action_seed,
                                 label=label, on_episode_start=lambda seed, base: torch.manual_seed(base+seed))
        result.update(checkpoint=str(Path(checkpoint)), deterministic=bool(deterministic),
                      training_seed=trainer.config.seed, training_updates=trainer.update_count,
                      training_physical_periods=trainer.environment_periods,
                      feature_scaling='frozen training-scenario scales')
        return result
