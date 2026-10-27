"""Bounded coordinate search for a large fleet's inventory-rule dispatch.

The objective and current-period physics match the exhaustive inventory rule.
Only the search changes: feasible actions are considered for one vehicle at a
time, with every candidate evaluated as a complete fleet action. The result is
a local-search approximation, not a global optimum or an approximate-DP value.
"""
from numbers import Real

import numpy as np

from ..simulator import DispatchAction, Mode
from .policies import _batch_dispatch_outcomes, _maximum_exchange, service_first_dispatch


def _individual_candidates(obs, config, masks, fractions):
    candidates = []
    for m in range(config.n_mcv):
        alternatives = []
        location = int(obs['locations'][m])
        for mode, target in np.argwhere(masks[m]):
            mode, target = Mode(int(mode)), int(target)
            if mode == Mode.REPOSITION and target == location and masks[m, Mode.WAIT, location]:
                continue  # Preserve WAIT on an exact physical tie.
            if mode in (Mode.RECHARGE, Mode.DISCHARGE):
                maximum = _maximum_exchange(obs, config, m, mode, target)
                alternatives.extend(DispatchAction(mode, target, float(maximum*f)) for f in fractions)
            else:
                alternatives.append(DispatchAction(mode, target))
        if not alternatives:
            raise ValueError(f'Coordinate inventory has no admissible action for vehicle {m}')
        candidates.append(alternatives)
    return candidates


def coordinate_inventory_dispatch(obs, config, shadow_price=.3, *, sweeps=2,
                                  energy_fractions=(1.,), action_mask=None,
                                  diagnostics=None):
    """Improve WAIT and service-first starts with bounded coordinate sweeps.

    ``action_mask`` optionally restricts the simulator's native [vehicle, mode,
    zone] mask; it cannot add physically infeasible actions. Exchange requests
    always lie on ``energy_fractions`` of the individually feasible maximum.
    Shared station allocation and competing service requests are recomputed for
    the full fleet at every comparison. Ties preserve the incumbent action.

    At most ``2 + 2*sweeps*sum(individual_candidate_counts)`` complete current
    outcomes are evaluated. No Cartesian joint grid is allocated. Diagnostics
    accumulate query counts and retain the latest call's bounded search trace.
    Neither global randomness nor future simulator draws are accessed.
    """
    if int(obs['stage']) != 1:
        raise ValueError('Coordinate inventory requires the dispatch stage')
    if (isinstance(shadow_price, (bool, np.bool_)) or not isinstance(shadow_price, Real)
            or not np.isfinite(shadow_price) or shadow_price < 0):
        raise ValueError('shadow_price must be finite and nonnegative')
    if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, (int, np.integer)) or sweeps < 1:
        raise ValueError('sweeps must be a positive integer')
    fractions = np.asarray(energy_fractions)
    if (fractions.ndim != 1 or not len(fractions) or fractions.dtype.kind not in 'iuf'
            or not np.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any()):
        raise ValueError('energy_fractions must be a nonempty finite sequence in [0, 1]')
    fractions = np.unique(fractions.astype(float))
    physical = np.asarray(obs['action_mask'])
    expected = (config.n_mcv, len(Mode), config.n_zones)
    if physical.shape != expected or not np.isin(physical, [0, 1]).all():
        raise ValueError('Physical action mask has incompatible shape or values')
    physical = physical.astype(bool)
    if action_mask is None:
        masks = physical
    else:
        supplied = np.asarray(action_mask)
        if supplied.shape != expected or not np.isin(supplied, [0, 1]).all():
            raise ValueError('action_mask must be binary [vehicles, modes, zones]')
        masks = supplied.astype(bool)
        if np.any(masks & ~physical):
            raise ValueError('action_mask may restrict, but not add, physical actions')
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError('diagnostics must be a dictionary or None')

    candidates = _individual_candidates(obs, config, masks, fractions)
    base = []
    for m, alternatives in enumerate(candidates):
        wait = DispatchAction(Mode.WAIT, int(obs['locations'][m]))
        base.append(wait if wait in alternatives else alternatives[0])
    # Project the second start onto exactly the same action grid. In particular,
    # a service-first residual-capacity recharge must not slip a fractional
    # amount into a maximum-only teacher's labels.
    service = service_first_dispatch(obs, config)
    projected = []
    for m, action in enumerate(service):
        matching = [a for a in candidates[m]
                    if a.mode == action.mode and a.target_zone == action.target_zone]
        projected.append(min(matching, key=lambda a: abs(a.energy_kwh-action.energy_kwh))
                         if matching else base[m])
    starts = [base]
    if projected != base:
        starts.append(projected)

    effective_shadow = 0. if int(obs['period'][0])+1 >= config.horizon else float(shadow_price)
    initial_inventory = float(np.sum(obs['energy_kwh']))
    counts = [len(a) for a in candidates]
    evaluated = completed_sweeps = accepted = 0

    def evaluate(modes, targets, amounts):
        nonlocal evaluated
        outcomes = _batch_dispatch_outcomes(obs, config, modes, targets, amounts)
        evaluated += len(modes)
        scores = outcomes['reward'] + effective_shadow * (
            outcomes['next_energy_kwh'].sum(axis=1)-initial_inventory)
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite coordinate inventory score')
        return scores

    best_actions, best_score = None, -np.inf
    initial_scores, final_scores, traces = [], [], []
    for start in starts:
        incumbent = list(start)
        modes = np.array([[int(a.mode) for a in incumbent]], dtype=np.int64)
        targets = np.array([[a.target_zone for a in incumbent]], dtype=np.int64)
        amounts = np.array([[a.energy_kwh for a in incumbent]], dtype=float)
        score = float(evaluate(modes, targets, amounts)[0])
        initial_scores.append(score)
        trace = [score]
        for _ in range(int(sweeps)):
            changed = False
            for m, alternatives in enumerate(candidates):
                replacements = [a for a in alternatives if a != incumbent[m]]
                if not replacements:
                    continue
                trial_modes = np.repeat(modes, len(replacements), axis=0)
                trial_targets = np.repeat(targets, len(replacements), axis=0)
                trial_amounts = np.repeat(amounts, len(replacements), axis=0)
                trial_modes[:, m] = [int(a.mode) for a in replacements]
                trial_targets[:, m] = [a.target_zone for a in replacements]
                trial_amounts[:, m] = [a.energy_kwh for a in replacements]
                scores = evaluate(trial_modes, trial_targets, trial_amounts)
                winner = int(np.argmax(scores))
                if scores[winner] > score + 1e-10:
                    incumbent[m] = replacements[winner]
                    modes[0] = trial_modes[winner]
                    targets[0] = trial_targets[winner]
                    amounts[0] = trial_amounts[winner]
                    score = float(scores[winner])
                    trace.append(score)
                    changed = True
                    accepted += 1
            completed_sweeps += 1
            if not changed:
                break
        final_scores.append(score)
        traces.append(trace)
        if score > best_score + 1e-10:
            best_actions, best_score = incumbent, score

    if diagnostics is not None:
        for name, amount in (('calls', 1), ('candidate_outcomes', evaluated),
                             ('starts', len(starts)), ('coordinate_updates', accepted),
                             ('sweeps_completed', completed_sweeps)):
            diagnostics[name] = diagnostics.get(name, 0) + amount
        diagnostics['max_candidates_per_call'] = max(diagnostics.get('max_candidates_per_call', 0), evaluated)
        exhaustive_size = 1
        for count in counts:
            exhaustive_size *= count
        diagnostics['last_call'] = dict(
            candidate_budget=2+2*int(sweeps)*sum(counts), candidate_outcomes=evaluated,
            individual_candidate_counts=counts, exhaustive_joint_grid_size=exhaustive_size,
            starts=len(starts), sweeps_completed=completed_sweeps,
            initial_scores=initial_scores, final_scores=final_scores,
            accepted_score_traces=traces, selected_score=float(best_score),
            effective_shadow_price=effective_shadow)
    return list(best_actions)
