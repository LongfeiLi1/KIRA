"""Pure previews from current paid observations. No characterization or executor."""
import math
DEVICE_NAMES = ('current_scale', 'output_scale', 'cc_pf')

def compare_allocation(plan):
    actual = list(plan['actual_constraints'])
    linear = list(plan['linearized_hard_constraints'])
    predicted = list(plan['predicted_constraints_with_margin'])
    if not len(actual) == len(linear) == len(predicted):
        raise ValueError('allocation_constraint_shapes')
    if not all((math.isfinite(x) for a in (actual, linear, predicted) for x in a)):
        raise ValueError('allocation_constraints_nonfinite')
    margin = [p - l for p, l in zip(predicted, linear)]
    if any((v < -1e-12 for v in margin)):
        raise ValueError('route_margin_negative')
    current_margin = [a + m for a, m in zip(actual, margin)]
    norm = lambda a: sum((max(v, 0.0) ** 2 for v in a))
    before, after = (norm(current_margin), norm(predicted))
    hard_before, hard_after = (norm(actual), norm(linear))
    return {'current_margin_violation_norm2': before, 'candidate_margin_violation_norm2': after, 'predicted_reduction': before - after, 'current_hard_violation_norm2': hard_before, 'candidate_linearized_hard_violation_norm2': hard_after, 'predicted_hard_reduction': hard_before - hard_after, 'ranking_basis': 'same constraints and same route margin before/after; device output goal not substituted', 'linearized_hard_constraints': linear, 'route_margin': margin, 'allocation_goal_is_not_measured_implementation': True}

def paid_power_seed(backend, plan, raw_key, z0=None):
    """The same calculation drives the numerical tool and its controller view."""
    import numpy as np
    row = backend.current
    initial = np.log([row['parameters'][n] for n in DEVICE_NAMES]) if z0 is None else np.asarray(z0, float).copy()
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError('original_device_seed_shape')
    paid = backend.descriptors.get(raw_key)
    source = None if paid is None else paid[1]
    power = None if source is None else source.get('power_w')
    goal = plan['power_target_w']
    valid = isinstance(power, (int, float)) and (not isinstance(power, bool)) and math.isfinite(power) and (power > 0) and isinstance(goal, (int, float)) and (not isinstance(goal, bool)) and math.isfinite(goal) and (goal > 0)
    prior = float(math.exp(initial[0]))
    if valid:
        proposed = prior * min(1.0, goal / power)
        lo, hi = backend.bounds['current_scale']
        initial[0] = math.log(min(max(proposed, lo), hi))
    params = {**row['parameters'], **plan.get('passives', {})}
    params.update({n: float(math.exp(v)) for n, v in zip(DEVICE_NAMES, initial)})
    ratio = float(params['current_scale'] / prior)
    return {'seed_log_parameters': list(map(float, initial)), 'seed_parameters': params, 'seed_current_scale_before': prior, 'seed_current_scale_after': params['current_scale'], 'power_ratio_rule_available': bool(valid), 'power_ratio_rule_applied': bool(valid and params['current_scale'] != prior), 'source_paid_OTA_power_w': power, 'source_paid_OTA_evidence': None if source is None else source['evidence'], 'allocated_device_power_goal_w': goal, 'approximate_seed_OTA_power_w': None if not valid else power * ratio, 'forecast_model': 'proportional bias/current-width knob from already paid operating point; approximation only', 'achieved_seed_OTA_power_w': None, 'achieved_seed_eta': None, 'predicted_full_filter_power_w': None, 'predicted_full_filter_phase_margin_deg': None, 'predicted_full_filter_response': None, 'predicted_complete_circuit_PASS': None, 'prediction_only': True, 'extra_SPICE_for_seed': 0, 'extra_SPICE_for_preview': 0, 'requires_OTA_full_and_fresh_endpoint_validation': True, 'remaining_native_device_solver_objective_cap': 12, 'first_seed_may_fail_target_and_native_solver_may_use_more_physical_candidates': True}

def measured_probe_preview(backend):
    """Shared raw-measurement guidance, not residual/Jacobian reconstruction."""
    row = backend.current or {}
    metrics = row.get('metrics', {})
    power = metrics.get('power_w')
    limit = backend.case['power_max_uW'] * 1e-06
    if not row.get('evidence') or row.get('measurement_status', 'valid') != 'valid' or isinstance(power, bool) or (not isinstance(power, (int, float))) or (not math.isfinite(power)) or (power <= 0):
        return {'available': False, 'reason': 'valid_measured_power_missing'}
    down = power * math.exp(-0.04)
    return {'available': True, 'action': 'probe', 'axis': 'current_scale', 'direction': -1, 'unchanged_log_step': -0.04, 'current_measured_full_power_w': power, 'approximate_power_after_single_probe_w': down, 'power_limit_w': limit, 'single_probe_estimate_reaches_power_limit': bool(down <= limit), 'forecast_model': 'generic proportional bias/width approximation; not a measurement', 'actual_candidate_power_w': None, 'actual_other_gates': None, 'knowledge_or_evidence_from_historical_answers': False}
