"""Audited equations and topology mapping. Outputs are predictions, not acceptance."""
from dataclasses import dataclass
import math
from rc_agent.relations import Passives
_UNITS = {'Ohm': ('resistance', 1.0), 'kOhm': ('resistance', 1000.0), 'MOhm': ('resistance', 1000000.0), 'F': ('capacitance', 1.0), 'uF': ('capacitance', 1e-06), 'nF': ('capacitance', 1e-09), 'pF': ('capacitance', 1e-12), 'Hz': ('frequency', 1.0), 'kHz': ('frequency', 1000.0), 'MHz': ('frequency', 1000000.0), 'rad/s': ('angular_frequency', 1.0)}

def to_si(value, unit, quantity):
    if unit not in _UNITS or _UNITS[unit][0] != quantity:
        raise ValueError('Explicit dimension-matched units required; ambiguous SPICE prefixes are not accepted')
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite value required')
    return value * _UNITS[unit][1]

@dataclass(frozen=True)
class Element:
    name: str
    kind: str
    nodes: tuple
    value_si: float

def map_sk_nodes(elements, vin, positive, output, negative, reference='0', reference_is_ac_ground=True):
    """Map by connections, never by C1/C2 names. Supply filter R/C only.

    A load capacitor is external and must be supplied separately, not included
    in this four-element filter list. Internal OTA components are not filter RC.
    """
    if negative != output or not reference_is_ac_ground:
        raise ValueError('Require direct unity feedback and verified AC reference')
    if len({vin, positive, output, reference}) != 4 or len(elements) != 4:
        raise ValueError('Ambiguous fixed SK terminals/elements')
    for e in elements:
        if e.kind not in {'R', 'C'} or len(e.nodes) != 2 or len(set(e.nodes)) != 2 or (not math.isfinite(e.value_si)) or (e.value_si <= 0):
            raise ValueError('Invalid passive element')
    rs = [e for e in elements if e.kind == 'R']
    cs = [e for e in elements if e.kind == 'C']
    if len(rs) != 2 or len(cs) != 2:
        raise ValueError('Require two filter resistors and two filter capacitors')
    first = [e for e in rs if vin in e.nodes]
    if len(first) != 1:
        raise ValueError('Cannot identify input resistor')
    r1 = first[0]
    x = next((n for n in r1.nodes if n != vin))
    if x in {vin, positive, output, reference}:
        raise ValueError('Invalid intermediate node')
    r2 = next((e for e in rs if e is not r1))
    if set(r2.nodes) != {x, positive}:
        raise ValueError('R2 must connect intermediate node to positive input')
    cg = [e for e in cs if set(e.nodes) == {positive, reference}]
    cf = [e for e in cs if set(e.nodes) == {x, output}]
    if len(cg) != 1 or len(cf) != 1:
        raise ValueError('Capacitor connectivity does not match SK Cg/Cf mapping')
    return {'node_map_id': 'sk_lp_cg_cf_v1', 'nodes': {'input': vin, 'x': x, 'y': positive, 'z': output, 'reference': reference}, 'component_names': {'r1': r1.name, 'r2': r2.name, 'cg': cg[0].name, 'cf': cf[0].name}, 'passives': Passives(r1.value_si, r2.value_si, cg[0].value_si, cf[0].value_si)}

def sk_constant_gain(p, k=1.0):
    if not math.isfinite(k) or k <= 0:
        raise ValueError('Positive constant real gain required')
    a = p.r1 * p.r2 * p.cg * p.cf
    b = (p.r1 + p.r2) * p.cg + p.r1 * p.cf * (1 - k)
    if b <= 0 or not math.isfinite(a + b) or a <= 0:
        raise ValueError('Stable positive second-order model required')
    return {'f0_hz': 1 / (2 * math.pi * math.sqrt(a)), 'q': math.sqrt(a) / b, 'gain': k, 'a_s2': a, 'b_s': b, 'provenance': 'derived_ideal_constant_gain_prediction', 'measured': False}

def f3db_relative_dc(f0_hz, q):
    if not all((math.isfinite(v) and v > 0 for v in (f0_hz, q))):
        raise ValueError('Positive finite f0 and Q required')
    if q < 0.5:
        a = 1 - 2 * q * q
        r = q * math.sqrt(2 / (math.hypot(a, 2 * q * q) + a))
    else:
        a = (1 / q) ** 2 - 2
        r = math.sqrt((math.hypot(a, 2) - a) / 2)
    return {'f_3db_hz': f0_hz * r, 'reference': 'relative_to_dc_gain', 'provenance': 'derived_second_order_prediction', 'measured': False}

def bounded_rc_coordinate(p, coordinate, log_step, bounds, max_log_step):
    if coordinate not in {'common_r', 'reciprocal_c', 'impedance_scale'}:
        raise ValueError('Unknown reviewed coordinate')
    if not all((math.isfinite(v) for v in (log_step, max_log_step))) or max_log_step <= 0 or abs(log_step) > max_log_step:
        raise ValueError('Step exceeds frozen log budget')
    if set(bounds) != {'r1', 'r2', 'cg', 'cf'}:
        raise ValueError('Complete explicit physical domain required')
    direction = {'common_r': (1, 1, 0, 0), 'reciprocal_c': (0, 0, -1, 1), 'impedance_scale': (1, 1, -1, -1)}[coordinate]
    values = []
    for name, v, s in zip(('r1', 'r2', 'cg', 'cf'), p.values, direction):
        lo, hi = bounds[name]
        if not (0 < lo < hi and math.isfinite(lo + hi) and (lo <= v <= hi)):
            raise ValueError('Invalid domain or current passive point')
        new = v * math.exp(s * log_step)
        if not lo <= new <= hi:
            raise ValueError('Proposed step leaves physical domain; no silent clipping')
        values.append(new)
    return Passives(*values)
