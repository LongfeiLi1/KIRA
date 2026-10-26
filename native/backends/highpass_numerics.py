"""Topology-correct response metrics + physical anchored log repair."""
import numpy as np
from rc_agent.allocation import LogDomain, log_jacobian, bounded_update
from rc_agent.models import OTA
from numerics.metrics.response import evaluate_response
from execution.highpass.adapter import physical, PASSIVE_NAMES

def rows(case, common):
    result = []

    def interval(name, lo, hi, scale):
        if lo is not None:
            result.append((name, 'lower', lo, scale))
        if hi is not None:
            result.append((name, 'upper', hi, scale))
    for name, target, tolerance in (('f0_hz', case['f0_hz'], case['f0_relative_tolerance']), ('q', case['q_target'], case['q_relative_tolerance'])):
        interval(name, target * (1 - tolerance), target * (1 + tolerance), target * tolerance)
    interval('reference_gain', case['gain_magnitude_min'], case['gain_magnitude_max'], 0.05)
    interval('power_w', 0, case['power_max_uW'] * 1e-06, case['power_max_uW'] * 1e-06)
    interval('phase_margin_deg', 60, None, 10)
    interval('dc_out', *common['dc_out_range_v'], 0.005)
    interval('minimum_saturation_margin_v', common['minimum_saturation_margin_v'], None, 0.05)
    interval('fit_max_component_error', 0, common['fit_max_component_error'], 0.03)
    interval('passband_max_loss_db', 0, case['passband_droop_max_db'], 1)
    interval('stopband_min_attenuation_db', case['stopband_attenuation_min_db'], None, 1)
    return result

def constraints(metrics, case, common):
    values = []
    for name, side, limit, scale in rows(case, common):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not np.isfinite(value)):
            raise ValueError('Missing actual metric: ' + name)
        values.append((value - limit) / scale if side == 'upper' else (limit - value) / scale)
    values += [-1.0 if metrics.get('stable') is True and metrics.get('full_local_stability') == 'stable' else 1.0, -1.0 if metrics.get('polarity_pass') is True else 1.0, -1.0 if metrics.get('step_status') == 'pass' else 1.0, abs(metrics['sampled_nyquist_turns']) / 0.1 - 1.0]
    return np.asarray(values)

def propose(row, eta, calibration, anchor, f, case, bound, common, *, coupled, radius, trusted):
    names = PASSIVE_NAMES[case['type_id']]
    n = len(names)
    q0 = [row['parameters'][k] for k in names]
    lo = [bound['physical_bounds'][k][0] for k in names]
    hi = [bound['physical_bounds'][k][1] for k in names]
    if coupled:
        a = np.asarray(anchor)
        if np.any(np.asarray(eta.values) < 0.5 * a) or np.any(np.asarray(eta.values) > 2 * a):
            raise ValueError('Measured ETA outside the fixed first-characterization domain')
        power = row['metrics']['power_w']
        q0 += list(eta.values) + [power]
        lo += list(0.5 * a) + [power * np.exp(-0.16)]
        hi += list(2 * a) + [power * np.exp(0.16)]
    domain = LogDomain(tuple(q0), tuple(lo), tuple(hi))
    actual = constraints(row['metrics'], case, common)
    profile = physical.data.response_profile(case)

    def model_constraints(x):
        p = dict(zip(names, x[:n]))
        a = OTA(*x[n:n + 3]) if coupled else eta
        h = calibration.response(p, a, f, case['load_cap_pF'] * 1e-12)
        m = dict(row['metrics'])
        m.update(evaluate_response(f, h, profile, fit_max_component_error=common['fit_max_component_error']))
        m['polarity_pass'] = row['metrics']['polarity_pass']
        if coupled:
            m['power_w'] = float(x[n + 3])
        return constraints(m, case, common)
    jacobian = log_jacobian(model_constraints, np.asarray(q0), domain)
    predictable = {'f0_hz', 'q', 'reference_gain', 'fit_max_component_error', 'passband_max_loss_db', 'stopband_min_attenuation_db', 'power_w'}
    margin = np.array([0.02 if name in predictable and (not (side == 'lower' and limit == 0)) else 0.0 for name, side, limit, _ in rows(case, common)] + [0.0] * 4)
    result = bounded_update(q0, domain, actual, jacobian, trusted=trusted, scale=np.ones(len(q0)), delta={'small': 0.04, 'normal': 0.08}[radius], margin=margin, regularization=0.05, iterations=160)
    from dataclasses import asdict
    linearized = actual + jacobian @ np.asarray(result.delta_z)
    values = result.values
    target = OTA(*values[n:n + 3]) if coupled else eta
    return {'passives': dict(zip(names, map(float, values[:n]))), 'ota_target': asdict(target), 'power_target_w': min(float(values[n + 3]) if coupled else row['metrics']['power_w'], 0.95 * case['power_max_uW'] * 1e-06), 'saturation_target_v': common['minimum_saturation_margin_v'] + 0.005, 'descriptor_domain_anchor': list(anchor), 'descriptor_domain_scale': [0.5, 2.0], 'actual_constraints': actual.tolist(), 'model_jacobian': jacobian.tolist(), 'linearized_hard_constraints': linearized.tolist(), 'predicted_constraints_with_margin': (linearized + margin).tolist(), 'required_route_margin': (0.5 * margin).tolist(), 'rc_only_feasible': bool(not coupled and np.all(linearized <= -0.5 * margin + 1e-12)), 'proposal': asdict(result), 'predicted_only': True, 'unsupported_gate_sensitivities': 'DC, margin, step, polarity, full MNA and port winding remain actual gates'}
