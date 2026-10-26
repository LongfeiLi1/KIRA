"""Shared complete-circuit acceptance metrics and coordinate names."""
import math
import numpy as np
NAMES = ('r1', 'r2', 'cg', 'cf', 'current_scale', 'output_scale', 'cc_pf')

class FirstFeasible(Exception):
    pass

def task_metrics(f, h, m, task, common):
    m = dict(m)
    f = np.asarray(f)
    h = np.asarray(h)
    f0 = task['f0_hz']
    q = task['q']
    if not f[0] < task['passband_edge_hz'] < task['stopband_edge_hz'] < f[-1]:
        raise ValueError('Frequency grid misses task bands')

    def mag_at(ff):
        return float(np.exp(np.interp(np.log(ff), np.log(f), np.log(abs(h)))))
    dc = abs(h[0])
    gain = abs(h)
    pass_values = np.r_[gain[f <= task['passband_edge_hz']], mag_at(task['passband_edge_hz'])]
    stop_values = np.r_[gain[f >= task['stopband_edge_hz']], mag_at(task['stopband_edge_hz'])]
    m['passband_max_loss_db'] = float(max(0, np.max(-20 * np.log10(pass_values / dc))))
    m['stopband_min_attenuation_db'] = float(np.min(-20 * np.log10(stop_values / dc)))
    band = (f >= f0 * 0.1) & (f <= f0 * 10)
    s = 1j * f[band] / f0
    target = 1 / (1 + s / q + s * s)
    m['band_complex_relative_error'] = float(np.max(abs(h[band] - target) / np.maximum(abs(target), 0.01)))
    m['band_error_is_extra_gate'] = False
    m['band_definition_hz'] = [f0 * 0.1, f0 * 10]
    m['attenuation_reference'] = 'relative to measured 1Hz DC-gain proxy'
    bounds = {'f0_hz': (f0 * (1 - task['f0_relative_tolerance']), f0 * (1 + task['f0_relative_tolerance'])), 'q': (q * (1 - task['q_relative_tolerance']), q * (1 + task['q_relative_tolerance'])), 'power_w': (0, task['power_limit_w']), 'phase_margin_deg': (task['phase_margin_min_deg'], 180), 'dc_gain': common['dc_gain_range'], 'dc_out': common['dc_out_range_v'], 'minimum_saturation_margin_v': (common['minimum_saturation_margin_v'], float('inf')), 'fit_max_component_error': (0, common['fit_max_component_error']), 'step_final_error_v': (0, common['step_final_error_max_v']), 'step_tail_peak_to_peak_v': (0, common['step_tail_peak_to_peak_max_v']), 'passband_max_loss_db': (0, task['passband_max_loss_db']), 'stopband_min_attenuation_db': (task['stopband_min_attenuation_db'], float('inf'))}
    failed = []
    for k, (lo, hi) in bounds.items():
        v = m.get(k)
        if v is None or isinstance(v, bool) or (not np.isfinite(v)) or (not lo <= v <= hi):
            failed.append(k)
    if m.get('stable') is not True or m.get('full_stability_status') != 'pass_numerical_local':
        failed.append('stability_unknown' if m.get('stable') is None else 'stability')
    status = 'unknown' if 'stability_unknown' in failed else 'fail' if failed else 'success'
    loss = (np.log(m['f0_hz'] / f0) / task['f0_relative_tolerance']) ** 2 + (np.log(m['q'] / q) / task['q_relative_tolerance']) ** 2
    loss += max(m['power_w'] / task['power_limit_w'] - 1, 0) ** 2 * 100
    loss += max(task['phase_margin_min_deg'] - m['phase_margin_deg'], 0) ** 2
    loss += max(m['passband_max_loss_db'] - task['passband_max_loss_db'], 0) ** 2 + max(task['stopband_min_attenuation_db'] - m['stopband_min_attenuation_db'], 0) ** 2
    loss += 1000 * sum((k in failed for k in ['dc_gain', 'dc_out', 'minimum_saturation_margin_v', 'fit_max_component_error', 'step_final_error_v', 'step_tail_peak_to_peak_v']))
    if m.get('stable') is not True:
        loss += 10000
    return (m, status, failed, float(loss))
