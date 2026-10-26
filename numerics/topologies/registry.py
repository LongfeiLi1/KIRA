"""Explicit outer topologies. No simulations, optimization, or gate promotion.

Transfer coefficients are independently checked from nodal equations by tests.
The transistor subcircuit and PDK header are imported unchanged, on demand.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
import numpy as np
VERSION = 'kira_type_level_text_v1'
FREQUENCIES = (12648.820123726657, 25297.640247453313, 37946.46037117997)
Q = 0.7071067811865476
OLD_BOUNDS = {'r1': (10000.0, 1000000.0), 'r2': (10000.0, 1000000.0), 'cg': (5e-12, 1e-10), 'cf': (5e-12, 2e-10), 'current_scale': (0.7, 1.4), 'output_scale': (0.7, 1.4), 'cc_pf': (1.0, 4.0)}

@dataclass(frozen=True)
class Topology:
    topology_id: str
    type_id: str
    family: str
    elements: tuple
    ota_plus: str
    ota_minus: str
    bounds_mapping: tuple
    execution_policy: str = 'requires_per_case_gates'

    @property
    def passive_names(self):
        return tuple((row[3] for row in self.elements))

    @property
    def physical_bounds(self):
        return {name: list(OLD_BOUNDS[source]) for name, source in self.bounds_mapping} | {name: list(OLD_BOUNDS[name]) for name in ('current_scale', 'output_scale', 'cc_pf')}

    @property
    def structure(self):
        return {'order': 2, 'sections': 1, 'ota_count': 1, 'elements': [list(e[:3]) for e in self.elements], 'ota_ports': [self.ota_plus, self.ota_minus, 'ampout', 'vdd', '0', 'vbias'], 'fixed_edges': [['Vloop', 'ampout', 'out'], ['Iport', '0', 'out'], ['CL', 'out', '0'], ['Vref', 'vref', '0'], ['Vin', 'in', '0'], ['Vdd', 'vdd', '0'], ['XB', 'vdd', '0', 'vbias']]}

    @property
    def structure_hash(self):
        return hashlib.sha256(json.dumps(self.structure, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
TOPOLOGIES = MappingProxyType({t.topology_id: t for t in (Topology('sallen_key_lowpass', 'T1', 'lowpass', (('R1', 'in', 'x', 'r1'), ('R2', 'x', 'y', 'r2'), ('Cg', 'y', 'vref', 'cg'), ('Cf', 'x', 'out', 'cf')), 'y', 'out', (('r1', 'r1'), ('r2', 'r2'), ('cg', 'cg'), ('cf', 'cf'))), Topology('sallen_key_highpass', 'T2', 'highpass', (('C1', 'in', 'x', 'c1'), ('C2', 'x', 'y', 'c2'), ('Rf', 'x', 'out', 'rf'), ('Rg', 'y', 'vref', 'rg')), 'y', 'out', (('c1', 'cg'), ('c2', 'cf'), ('rf', 'r1'), ('rg', 'r2'))))})

def get_topology(topology_id):
    try:
        return TOPOLOGIES[topology_id]
    except KeyError as exc:
        raise ValueError('Unknown topology_id: ' + str(topology_id)) from exc

def validate_passives(topology_id, passives, enforce_bounds=False):
    t = get_topology(topology_id)
    if set(passives) != set(t.passive_names):
        raise ValueError('Explicit passive keys mismatch: expected ' + repr(t.passive_names))
    p = {k: float(v) for k, v in passives.items()}
    if not all((math.isfinite(v) and v > 0 for v in p.values())):
        raise ValueError('Passives must be finite positive SI values')
    if enforce_bounds:
        for k, v in p.items():
            lo, hi = t.physical_bounds[k]
            if not lo <= v <= hi:
                raise ValueError('Passive outside frozen domain: ' + k)
    return p

def ideal_coefficients(topology_id, passives):
    """Return descending numerator and denominator coefficients in physical s."""
    p = validate_passives(topology_id, passives)
    family = get_topology(topology_id).family
    if family == 'lowpass':
        a = p['r1'] * p['r2'] * p['cg'] * p['cf']
        b = (p['r1'] + p['r2']) * p['cg']
        numerator = [0.0, 0.0, 1.0]
    elif family == 'highpass':
        a = p['c1'] * p['c2'] * p['rf'] * p['rg']
        b = p['rf'] * (p['c1'] + p['c2'])
        numerator = [a, 0.0, 0.0]
    else:
        raise ValueError('Unsupported SK response')
    return (numerator, [a, b, 1.0])

def ideal_parameters(topology_id, passives):
    _, (a, b, _) = ideal_coefficients(topology_id, passives)
    return {'f0_hz': 1 / (2 * math.pi * math.sqrt(a)), 'q': math.sqrt(a) / b}

def ideal_response(topology_id, passives, frequency_hz):
    f = np.asarray(frequency_hz, dtype=float)
    if not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError('Frequency must be finite and nonnegative')
    num, den = ideal_coefficients(topology_id, passives)
    s = 2j * np.pi * f
    return np.polyval(num, s) / np.polyval(den, s)

def analytic_seed(topology_id, f0_hz):
    if not math.isfinite(f0_hz) or f0_hz <= 0:
        raise ValueError('Positive finite target frequency required')
    w = 2 * math.pi * f0_hz
    family = get_topology(topology_id).family
    if family == 'lowpass':
        r = 1 / (w * math.sqrt(2e-11 * 4e-11))
        p = dict(r1=r, r2=r, cg=2e-11, cf=4e-11)
    elif family == 'highpass':
        rf = 1 / (w * 2e-11 * math.sqrt(2))
        p = dict(c1=2e-11, c2=2e-11, rf=rf, rg=2 * rf)
    else:
        raise ValueError('Unsupported SK response')
    return validate_passives(topology_id, p, enforce_bounds=True)

def real_netlist_prefix(topology_id, passives, devices, load_f, input_spec=None, closed_dc_only=False):
    """Unterminated real-OTA prefix; caller must supply and validate analyses.

    Only the two supported Sallen-Key topologies are rendered.
    Never treats this deck construction as measurement or gate acceptance.
    """
    t = get_topology(topology_id)
    if t.execution_policy == 'proposal_only':
        raise ValueError('Unsupported topology')
    p = validate_passives(topology_id, passives, enforce_bounds=True)
    if not math.isfinite(load_f) or load_f <= 0:
        raise ValueError('Positive explicit load required')
    from rc_agent.physical import header
    if closed_dc_only and input_spec is not None:
        raise ValueError('DC-only construction does not accept waveform override')
    if closed_dc_only:
        input_spec = '1.2'
    elif input_spec is None:
        input_spec = 'DC 1.2 AC 1'
    if '\n' in input_spec or '\r' in input_spec:
        raise ValueError('Input source spec must occupy exactly one line')
    output_node = 'out' if closed_dc_only else 'ampout'
    reference_source = 'Vref vref 0 1.2' if closed_dc_only else 'Vref vref 0 DC 1.2 AC 0'
    lines = [header(devices), 'Vdd vdd 0 1.8', reference_source, 'Vin in 0 ' + input_spec, 'XB vdd 0 vbias bias', f'XU {t.ota_plus} {t.ota_minus} {output_node} vdd 0 vbias ota']
    if not closed_dc_only:
        lines += ['Vloop ampout out DC 0 AC 0', 'Iport 0 out DC 0 AC 0']
    lines += [f'{name} {a} {b} {p[key]:.16g}' for name, a, b, key in t.elements]
    lines.append(f'CL out 0 {load_f:.16g}')
    return '\n'.join(lines) + '\n'
