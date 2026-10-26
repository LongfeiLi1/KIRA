"""Reviewed analytic rules with explicit domains, freedoms and model provenance."""
import hashlib
import json
import math
from pathlib import Path
VERSION = 'active_filter_initialization_v2_20261020'

class InitializationError(ValueError):
    """Invalid or infeasible specification; never silently replace the topology."""

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise InitializationError('positive finite scalar required: ' + name)
    return float(value)

def domain(value):
    if type(value) is not dict or not value:
        raise InitializationError('explicit fixed parameter domain required')
    result = {}
    for name, pair in value.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InitializationError('bound pair required: ' + name)
        lo, hi = (positive(x, name) for x in pair)
        if lo > hi:
            raise InitializationError('reversed domain: ' + name)
        result[name] = [lo, hi]
    return result

def intersection(lower, upper, preferred=None):
    lo, hi = (max(lower), min(upper))
    if not 0 < lo <= hi or not all((math.isfinite(v) for v in (lo, hi))):
        raise InitializationError('analytic topology outside fixed joint domain')
    value = math.sqrt(lo * hi) if preferred is None else min(hi, max(lo, preferred))
    return (value, dict(lower=lo, upper=hi, preferred=preferred, choice='geometric_center' if preferred is None else 'project_reference_scale_to_feasible_interval', selected=value))

def template_policy():
    return dict(id='external_ota_reference_defaults_v1', kind='fixed_template', current_scale=1.0, output_scale=1.0, cc_pf=2.0, rz_ohm=4604.0, bias_reference_a=1e-05, length_um=1.0, widths_um={'M1': 3.469, 'M2': 3.469, 'M3': 25.676, 'M4': 25.676, 'M5': 4.117, 'M6': 46.478, 'M7': 11.178, 'M8': 2.058}, multiplicity={'M6': 3}, task_adaptive=False, reason='No validated specification-to-device sizing map; retain explicit common template.', characterization='after_first_full_measurement_if_needed_for_repair', initial_feedback=False, source='rc_agent.physical.Devices/ota_netlist')

def sk_lp(f0, q, bounds):
    w, ratio = (2 * math.pi * f0, 4 * q * q)
    constant = 1 / (w * 2 * q)
    rlo, rhi = (max((bounds[n][0] for n in ('r1', 'r2'))), min((bounds[n][1] for n in ('r1', 'r2'))))
    cg, scale = intersection([bounds['cg'][0], bounds['cf'][0] / ratio, constant / rhi], [bounds['cg'][1], bounds['cf'][1] / ratio, constant / rlo])
    r = constant / cg
    return (dict(r1=r, r2=r, cg=cg, cf=ratio * cg), [dict(rule='SK_LP_UNITY_EQUAL_R', omega_rad_s=w, q=q, cf_over_cg=ratio, r_times_cg=constant, scale=scale, equations=['Cf=4*Q^2*Cg', 'R=1/(omega*2*Q*Cg)'])])

def sk_hp(f0, q, bounds):
    w = 2 * math.pi * f0
    c, scale = intersection([bounds['c1'][0], bounds['c2'][0], 1 / (w * 2 * q * bounds['rf'][1]), 2 * q / (w * bounds['rg'][1])], [bounds['c1'][1], bounds['c2'][1], 1 / (w * 2 * q * bounds['rf'][0]), 2 * q / (w * bounds['rg'][0])], 4e-11)
    return (dict(c1=c, c2=c, rf=1 / (w * 2 * q * c), rg=2 * q / (w * c)), [dict(rule='SK_HP_UNITY_EQUAL_C', omega_rad_s=w, q=q, scale=scale, equations=['Rf=1/(omega*2*Q*C)', 'Rg=2*Q/(omega*C)'], reference_policy='40pF generic historical design scale, not a per-case parameter lookup')])

def synthesize(spec, bounds):
    if type(spec) is not dict or set(spec) - {'task_id', 'topology', 'frequency_hz', 'q', 'gain'}:
        raise InitializationError('exact specification schema required')
    kind = spec.get('topology')
    if kind not in {'t1_sk_lp', 't2_sk_hp'}:
        raise InitializationError('Only Sallen-Key lowpass and highpass are available')
    f0 = positive(spec.get('frequency_hz'), 'frequency_hz')
    q = positive(spec.get('q'), 'q')
    if not isinstance(spec.get('task_id'), str) or not spec['task_id']:
        raise InitializationError('task label required')
    if type(spec.get('gain')) not in (int, float) or spec['gain'] != 1:
        raise InitializationError('Only unity-gain Sallen-Key topologies are available')
    b = domain(bounds)
    required = {'r1', 'r2', 'cg', 'cf'} if kind == 't1_sk_lp' else {'c1', 'c2', 'rf', 'rg'}
    if set(b) != required | {'current_scale', 'output_scale', 'cc_pf'}:
        raise InitializationError('exact topology and OTA bounds required')
    values, steps = (sk_lp if kind == 't1_sk_lp' else sk_hp)(f0, q, b)
    values.update(current_scale=1.0, output_scale=1.0, cc_pf=2.0)
    for name, value in values.items():
        positive(value, name)
        if not b[name][0] <= value <= b[name][1]:
            raise InitializationError('solved value outside domain: ' + name)
    receipt = dict(schema='kira.initialization.receipt.v2', version=VERSION, specification=json.loads(canonical(spec)), task_label_used_for_numeric_lookup=False, topology=kind, domain=b, parameters=values, steps=steps, ota_policy=template_policy(), rule_ids=[s['rule'] for s in steps], online_llm_calls=0, online_spice_processes=0, first_full_measurement_occurred=False, physical_acceptance=False, predictions_are_measurements=False, parameter_units={n: 'pF' if n.endswith('_pf') else 'dimensionless' if n.endswith('scale') else 'F' if n.startswith('c') else 'ohm' for n in values}, implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt['receipt_sha256'] = digest(receipt)
    return receipt
