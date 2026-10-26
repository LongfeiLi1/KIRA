"""Local realization feedback changes target penalties, never domains or gates."""
import math
from types import MethodType
import numpy as np
from rc_agent.allocation import bounded_update as original_update, Proposal
ETA = ('a0', 'wp_rad_s', 'ro_ohm')
TOLERANCES = np.array([0.25, 0.25, 0.35])

def feedback_weights(event):
    if event.get('target_realized_within_tolerances') is True:
        return [1.0, 1.0, 1.0]
    best, target = (event.get('best'), event.get('target'))
    if not isinstance(best, dict) or not isinstance(target, dict):
        return [1.0, 1.0, 1.0]
    try:
        achieved = np.array([best['eta'][k] for k in ETA], float)
        wanted = np.array([target[k] for k in ETA], float)
    except (KeyError, TypeError, ValueError):
        return [1.0, 1.0, 1.0]
    if not np.all(np.isfinite(achieved)) or not np.all(np.isfinite(wanted)) or np.any(achieved <= 0) or np.any(wanted <= 0):
        return [1.0, 1.0, 1.0]
    excess = np.maximum(np.abs(np.log(achieved / wanted)) / TOLERANCES - 1, 0)
    return list(map(float, 1 + np.minimum(excess, 3)))

def weighted_update(q, domain, c, jacobian, *, weights, trusted, scale, delta, margin, regularization, iterations=80):
    w = np.asarray(weights, float)
    if w.shape != np.asarray(q).shape or not np.all(np.isfinite(w)) or np.any(w < 1) or np.any(w > 4):
        raise ValueError('target_penalty_weights_invalid')
    if np.all(w == 1):
        return original_update(q, domain, c, jacobian, trusted=trusted, scale=scale, delta=delta, margin=margin, regularization=regularization, iterations=iterations)
    if not trusted:
        raise ValueError('Refuse update without measured model trust')
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ValueError('Bounded iteration count required')
    if not np.isfinite(delta) or delta <= 0 or (not np.isfinite(regularization)) or (regularization <= 0):
        raise ValueError('Positive trust radius and regularization required')
    z = domain.encode(q)
    ref, lo, hi = domain.arrays()
    c, j, d, m = (np.asarray(x, float) for x in (c, jacobian, scale, margin))
    if c.ndim != 1 or len(c) == 0 or j.shape != (len(c), len(z)) or (d.shape != z.shape) or (m.shape != c.shape):
        raise ValueError('Constraint/Jacobian/scale/margin shapes disagree')
    if not all((np.all(np.isfinite(x)) for x in (c, j, d, m))) or np.any(d <= 0) or np.any(m < 0):
        raise ValueError('Invalid update inputs')
    lower = np.maximum(np.log(lo / ref) - z, -delta / d)
    upper = np.minimum(np.log(hi / ref) - z, delta / d)
    step = np.zeros_like(z)
    penalty = regularization * w * d * d

    def loss(x):
        return float(np.sum(np.maximum(c + j @ x + m, 0) ** 2) + np.sum(penalty * x * x))
    before = loss(step)
    rate = 1 / (2 * (np.linalg.norm(j, 2) ** 2 + np.max(penalty)))
    for _ in range(iterations):
        gradient = 2 * j.T @ np.maximum(c + j @ step + m, 0) + 2 * penalty * step
        step = np.clip(step - rate * gradient, lower, upper)
    return Proposal(tuple(domain.decode(z + step)), tuple(step), before, loss(step), iterations)

def install(backend, *, enabled):
    from provider.stream import injected
    native_allocate = backend.allocation.__func__
    propose = native_allocate.__globals__['propose']
    memory = {'enabled': enabled, 'eta_penalty_weights': [1.0, 1.0, 1.0], 'revision': 0, 'paid_realization_events': 0, 'feedback_events': 0, 'weighted_allocations': 0, 'scope': 'this cold case only; no historical measurements or case-ID rules'}
    backend.adaptive_target_feedback = memory
    native_event = backend.event

    def event(self, kind, data):
        result = native_event(kind, data)
        if kind == 'device_target_realization':
            memory['paid_realization_events'] += 1
            new = feedback_weights(data) if enabled else [1.0, 1.0, 1.0]
            memory['eta_penalty_weights'] = new
            memory['revision'] += 1
            memory['feedback_events'] += int(any((x > 1 for x in new)))
            native_event('target_realization_feedback', dict(memory))
        return result
    backend.event = MethodType(event, backend)

    def update(q, domain, c, jacobian, **kwargs):
        weights = np.ones(len(q))
        passive_count = len(backend.names) - 3 if backend.case['type_id'] != 'T1' else 4
        if enabled and len(q) == passive_count + 4:
            weights[passive_count:passive_count + 3] = memory['eta_penalty_weights']
            memory['weighted_allocations'] += int(np.any(weights > 1))
        return weighted_update(q, domain, c, jacobian, weights=weights, **kwargs)
    adaptive_propose = injected(propose, bounded_update=update)
    allocate = injected(native_allocate, propose=adaptive_propose).__get__(backend, type(backend))

    def allocation(self, coupled, radius='normal'):
        plan = allocate(coupled, radius)
        plan['target_penalty_feedback'] = dict(memory)
        return plan
    backend.allocation = MethodType(allocation, backend)
