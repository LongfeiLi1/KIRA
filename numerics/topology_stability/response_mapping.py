"""Prove a complete MNA permutation using independent named AC excitations.

No OP ordering, device-number assumption, or passive value matching is used.
Backward-Euler final accepted intervals bootstrap C; physical charge anchors and
all-state held-out AC samples validate the recovered permutation independently.
Ambiguous/unobservable nodes fail closed rather than being removed.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .core import ContractError

def map_response_states(g, c, expected_names, source_branches, frequencies, named_responses):
    names = tuple(expected_names)
    n = len(names)
    f = np.asarray(frequencies, float)
    g = np.asarray(g, float)
    c = np.asarray(c, float)
    obs = np.asarray(named_responses, complex)
    if len(set(names)) != n or g.shape != (n, n) or c.shape != (n, n) or (obs.shape != (len(source_branches), len(f), n)):
        raise ContractError('mapping shape/name coverage mismatch')
    if len(f) < 12 or np.any(np.diff(f) <= 0) or np.any(f <= 0) or (not all((np.all(np.isfinite(a)) for a in (g, c, f, obs)))):
        raise ContractError('invalid independent mapping evidence')
    branches = [i for i in range(n) if np.count_nonzero(g[i]) == 1 and np.count_nonzero(c[i]) == 0 and (np.max(abs(g[i])) == 1) and (g[i, i] == 0)]
    if len(branches) != len(source_branches) or len(branches) > 4:
        raise ContractError('unsupported source incidence count')
    for b in branches:
        node = int(np.flatnonzero(g[b])[0])
        if np.count_nonzero(g[:, b]) != 1 or g[node, b] != g[b, node] or np.any(c[:, b]):
            raise ContractError('impure source incidence')
    train = np.arange(0, len(f), 10)
    held = np.setdiff1d(np.arange(len(f)), train)
    if len(train) < 2 or len(held) < 8:
        raise ContractError('independent held-out frequencies missing')
    hypotheses = []
    for order in itertools.permutations(branches):
        rhs = np.eye(n)[:, order]
        pred = np.array([np.linalg.solve(g + 2j * np.pi * ff * c, rhs).T for ff in f[train]]).transpose(1, 0, 2)
        aa = pred.reshape(-1, n).T
        bb = obs[:, train, :].reshape(-1, n).T
        scale = np.maximum(np.maximum(np.linalg.norm(aa, axis=1)[:, None], np.linalg.norm(bb, axis=1)[None, :]), 1e-30)
        cost = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2) / scale
        rows, cols = linear_sum_assignment(cost)
        hypotheses.append((float(np.max(cost[rows, cols])), float(np.sum(cost[rows, cols])), order, cols, cost))
    hypotheses.sort(key=lambda h: h[:2])
    score, total, order, cols, cost = hypotheses[0]
    if score > 0.0001 or (len(hypotheses) > 1 and hypotheses[1][0] <= max(10 * score, 0.001)):
        raise ContractError('no unique source/state mapping hypothesis')
    assigned = cost[np.arange(n), cols]
    alternatives = np.array([np.min(np.delete(cost[i], cols[i])) for i in range(n)])
    if np.any(alternatives <= 10 * assigned + 1e-06):
        raise ContractError('ambiguous dynamic states; no mode may be discarded')
    mapped = tuple((names[j] for j in cols))
    if any((mapped[b] != source for b, source in zip(order, source_branches))):
        raise ContractError('mapped source identity disagrees with excitation')
    rhs = np.eye(n)[:, order]
    predicted = np.array([np.linalg.solve(g + 2j * np.pi * ff * c, rhs).T for ff in f[held]]).transpose(1, 0, 2)
    observed = obs[:, held, :][:, :, cols]
    error = float(np.max(abs(predicted - observed) / np.maximum(np.max(abs(observed), axis=(0, 1), keepdims=True), 1e-30)))
    if error > 0.0001:
        raise ContractError('held-out complete-state AC mapping validation failed')
    return (mapped, {'method': 'independent_named_ac_response_mapping', 'mapped_variables': n, 'training_frequencies': len(train), 'held_out_frequencies': len(held), 'independent_excitations': list(source_branches), 'maximum_training_relative_error': score, 'maximum_held_out_scaled_error': error, 'minimum_alternative_state_distance': float(min(alternatives)), 'source_hypothesis_maximum_errors': [h[0] for h in hypotheses], 'matrix_names_in_order': list(mapped), 'discarded_states': 0, 'provenance': 'unchanged circuit, named AC responses matched to numerical G+sC solves; every permutation of grounded source identities checked'})
