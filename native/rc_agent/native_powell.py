"""Shared native Powell with real-measurement caps enforced by its caller."""
import numpy as np
from scipy.optimize import minimize

def native_powell(objective, z0, bounds, settings, state, on_iteration):
    z0 = np.asarray(z0, float)
    bounds = np.asarray(bounds, float)
    if bounds.shape != (len(z0), 2) or np.any(z0 < bounds[:, 0]) or np.any(z0 > bounds[:, 1]):
        raise ValueError('Invalid shared domain')
    state.update(native_objective_calls=0, completed_iterations=0, coordinate_design_calls=0, scipy_objective_calls=0, scipy_started=False)

    def invoke(z, phase):
        state['native_objective_calls'] += 1
        state[phase] += 1
        return objective(z)
    invoke(z0, 'coordinate_design_calls')
    for k in range(len(z0)):
        z = z0.copy()
        direction = -1 if z0[k] >= np.mean(bounds[k]) else 1
        z[k] = np.clip(z[k] + direction * settings['log_coordinate_probe_step'], *bounds[k])
        invoke(z, 'coordinate_design_calls')
    state['scipy_started'] = True

    def callback(z):
        state['completed_iterations'] += 1
        on_iteration(dict(state), np.asarray(z, float))
    result = minimize(lambda z: invoke(z, 'scipy_objective_calls'), z0, method='Powell', bounds=bounds, callback=callback, options={'maxfev': settings['max_native_function_calls'], 'maxiter': settings['max_native_iterations'], 'xtol': settings['xtol'], 'ftol': settings['ftol']})
    return result
