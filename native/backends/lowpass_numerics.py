"""Measured-constraint anchored log updates; all formulas are predictions."""
from dataclasses import asdict
import math
import numpy as np
from rc_agent.allocation import LogDomain, log_jacobian, bounded_update
from rc_agent.baseline import task_metrics
from rc_agent.physical import fit_lowpass, Devices
from rc_agent.models import OTA, sk_response
from rc_agent.relations import Passives
PASSIVE_NAMES = ('r1', 'r2', 'cg', 'cf')
DEVICE_NAMES = ('current_scale', 'output_scale', 'cc_pf')
NAMES = PASSIVE_NAMES + DEVICE_NAMES
RADII = {'small': 0.04, 'normal': 0.08}

def analytic_initial(task, bounds):
    """Unified specification-only SK synthesis; physical acceptance is separate."""
    from initialization import synthesize
    receipt = synthesize(dict(task_id=task['id'], topology='t1_sk_lp', frequency_hz=task['f0_hz'], q=task['q'], gain=1.0), bounds)
    return receipt['parameters']

def constraint_rows(task, common):
    """Signed violations <=0; gates and units inherited from the public evaluator."""
    rows = []

    def interval(name, lo, hi, scale):
        if lo is not None:
            rows.append((name, 'lower', lo, scale))
        if hi is not None:
            rows.append((name, 'upper', hi, scale))
    interval('f0_hz', task['f0_hz'] * (1 - task['f0_relative_tolerance']), task['f0_hz'] * (1 + task['f0_relative_tolerance']), task['f0_hz'] * task['f0_relative_tolerance'])
    interval('q', task['q'] * (1 - task['q_relative_tolerance']), task['q'] * (1 + task['q_relative_tolerance']), task['q'] * task['q_relative_tolerance'])
    interval('dc_gain', *common['dc_gain_range'], 0.05)
    interval('power_w', 0, task['power_limit_w'], task['power_limit_w'])
    interval('phase_margin_deg', task['phase_margin_min_deg'], 180, 10)
    interval('dc_out', *common['dc_out_range_v'], 0.005)
    interval('minimum_saturation_margin_v', common['minimum_saturation_margin_v'], None, 0.05)
    interval('fit_max_component_error', 0, common['fit_max_component_error'], 0.03)
    interval('step_final_error_v', 0, common['step_final_error_max_v'], 1e-05)
    interval('step_tail_peak_to_peak_v', 0, common['step_tail_peak_to_peak_max_v'], 1e-05)
    interval('passband_max_loss_db', 0, task['passband_max_loss_db'], 1)
    interval('stopband_min_attenuation_db', task['stopband_min_attenuation_db'], None, 1)
    return rows

def constraints(metrics, task, common):
    values = []
    for name, side, limit, scale in constraint_rows(task, common):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not np.isfinite(value)):
            raise ValueError('Missing actual constraint: ' + name)
        values.append((value - limit) / scale if side == 'upper' else (limit - value) / scale)
    values.append(-1.0 if metrics.get('stable') is True and metrics.get('full_stability_status') == 'pass_numerical_local' else 1.0)
    return np.asarray(values)

def model_response(p, eta, f, load_f, calibration=None):
    return sk_response(p, eta, f, load_f) if calibration is None else calibration.response(p, eta, f, load_f)

def propose(observation, eta, f, task, common, bounds, *, calibration=None, coupled=False, radius='normal', trusted=False, descriptor_anchor=None):
    """c comes from complete physical measurements; J comes only from the model.
    Unsupported DC/stability/transient rows have zero sensitivity, remain in c,
    and must be validated physically. Power is an explicit allocation coordinate.
    """
    x = observation['parameters']
    p0 = [x[n] for n in PASSIVE_NAMES]
    q0 = p0 + (list(eta.values) + [observation['metrics']['power_w']] if coupled else [])
    lower = [bounds[n][0] for n in PASSIVE_NAMES]
    upper = [bounds[n][1] for n in PASSIVE_NAMES]
    if coupled:
        if descriptor_anchor is None:
            raise ValueError('Fixed first-characterization descriptor anchor required')
        anchor = np.asarray(descriptor_anchor, float)
        if anchor.shape != (3,) or np.any(anchor <= 0) or (not np.all(np.isfinite(anchor))):
            raise ValueError('Invalid fixed descriptor anchor')
        lower += list(anchor * 0.5) + [q0[-1] * math.exp(-0.16)]
        upper += list(anchor * 2.0) + [q0[-1] * math.exp(0.16)]
        if np.any(np.asarray(q0[4:7]) < anchor * 0.5) or np.any(np.asarray(q0[4:7]) > anchor * 2.0):
            raise ValueError('Actual OTA outside fixed shared descriptor domain; probe or finish required')
    domain = LogDomain(tuple(q0), tuple(lower), tuple(upper))
    actual = constraints(observation['metrics'], task, common)

    def predicted(values):
        p = Passives(*values[:4])
        a = OTA(*values[4:7]) if coupled else eta
        h = model_response(p, a, f, task['load_f'], calibration)
        metrics = dict(observation['metrics'])
        metrics.update(fit_lowpass(f, h, task['f0_hz']), dc_gain=float(abs(h[0])))
        if coupled:
            metrics['power_w'] = float(values[7])
        metrics, _, _, _ = task_metrics(f, h, metrics, task, common)
        return constraints(metrics, task, common)
    jacobian = log_jacobian(predicted, np.asarray(q0), domain)
    predictable = {'f0_hz', 'q', 'dc_gain', 'fit_max_component_error', 'passband_max_loss_db', 'stopband_min_attenuation_db', 'power_w'}
    margin = np.array([0.02 if name in predictable and (not (side == 'lower' and limit == 0)) else 0.0 for name, side, limit, _ in constraint_rows(task, common)] + [0.0])
    result = bounded_update(np.asarray(q0), domain, actual, jacobian, trusted=trusted, scale=np.ones(len(q0)), delta=RADII[radius], margin=margin, regularization=0.05, iterations=160)
    next_values = np.asarray(result.values)
    linearized = actual + jacobian @ np.asarray(result.delta_z)
    prediction = linearized + margin
    required_route_margin = 0.5 * margin
    target = OTA(*next_values[4:7]) if coupled else eta
    power_target = min(float(next_values[7]) if coupled else observation['metrics']['power_w'], 0.95 * task['power_limit_w'])
    return {'descriptor_domain_anchor': None if descriptor_anchor is None else list(map(float, descriptor_anchor)), 'descriptor_domain_scale': [0.5, 2.0], 'passives': dict(zip(PASSIVE_NAMES, map(float, next_values[:4]))), 'ota_target': asdict(target), 'power_target_w': power_target, 'saturation_target_v': common['minimum_saturation_margin_v'] + 0.005, 'actual_constraints': actual.tolist(), 'model_jacobian': jacobian.tolist(), 'predicted_constraints_with_margin': prediction.tolist(), 'linearized_hard_constraints': linearized.tolist(), 'required_route_margin': required_route_margin.tolist(), 'rc_only_feasible': bool(not coupled and np.all(linearized <= -required_route_margin + 1e-12)), 'predicted_only': True, 'proposal': asdict(result), 'unsupported_gate_sensitivities': 'zero; DC, headroom, full stability and transient gates remain measured'}
