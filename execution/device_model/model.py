"""Named-node SK highpass model using independently characterized unloaded OTA A/Z.

The OTA Thevenin equation is (Vout-A*(Vplus-Vminus))/Z plus external
output currents = 0. Every R/C edge is stamped from the registered graph.
No full-filter response is used to fit or correct this model.
"""
from dataclasses import dataclass
import math
import numpy as np
from execution.highpass.adapter import physical
from numerics.topologies.registry import get_topology, validate_passives
from rc_agent.models import OTA

def network_response(type_id, passives, frequencies, gain, impedance, load_f=0.0):
    if type_id not in ('T2',):
        raise ValueError('T2 topology required')
    topology = get_topology(physical.data.TOPOLOGY_IDS[type_id])
    p = validate_passives(topology.topology_id, passives)
    f = np.asarray(frequencies, float)
    a = np.asarray(gain, complex)
    z = np.asarray(impedance, complex)
    if f.ndim != 1 or not len(f) or np.any(f <= 0) or np.any(np.diff(f) <= 0) or (not np.all(np.isfinite(f))):
        raise ValueError('Finite positive increasing frequency vector required')
    if a.shape != f.shape or z.shape != f.shape or (not np.all(np.isfinite(a))) or (not np.all(np.isfinite(z))) or np.any(abs(z) == 0):
        raise ValueError('Invalid unloaded A/Z samples')
    if not math.isfinite(load_f) or load_f < 0:
        raise ValueError('Invalid output load')
    nodes = ('x', 'y', 'out')
    index = {name: i for i, name in enumerate(nodes)}
    known = {'in': 1.0, 'vref': 0.0}
    s = 2j * np.pi * f
    matrix = np.zeros((len(f), 3, 3), complex)
    rhs = np.zeros((len(f), 3), complex)
    for component, n1, n2, key in topology.elements:
        admittance = s * p[key] if component.startswith('C') else np.full(len(f), 1 / p[key])
        for first, second in ((n1, n2), (n2, n1)):
            if first in index:
                i = index[first]
                matrix[:, i, i] += admittance
                if second in index:
                    matrix[:, i, index[second]] -= admittance
                else:
                    rhs[:, i] += admittance * known[second]
    out = index['out']
    matrix[:, out, out] += 1 / z + s * load_f
    for node, sign in ((topology.ota_plus, -1), (topology.ota_minus, 1)):
        if node in index:
            matrix[:, out, index[node]] += sign * a / z
        else:
            rhs[:, out] -= sign * a / z * known[node]
    result = np.linalg.solve(matrix, rhs[..., None])[:, out, 0]
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite topology model response')
    return result

@dataclass(frozen=True)
class TopologyCalibration:
    anchor: OTA
    frequencies: tuple
    gain: tuple
    impedance: tuple
    type_id: str

    def response(self, p, eta, f_hz, load_f=0.0):
        f = np.asarray(f_hz, float)
        source = np.asarray(self.frequencies, float)
        query = f * self.anchor.wp_rad_s / eta.wp_rad_s
        if query.min() < source.min() or query.max() > source.max():
            raise ValueError('Calibration extrapolation forbidden')

        def interp(values):
            v = np.asarray(values, complex)
            if v.shape != source.shape or not np.all(np.isfinite(v)):
                raise ValueError('Invalid calibration samples')
            return np.interp(np.log(query), np.log(source), v.real) + 1j * np.interp(np.log(query), np.log(source), v.imag)
        a = interp(self.gain) * eta.a0 / self.anchor.a0
        z = interp(self.impedance) * eta.ro_ohm / self.anchor.ro_ohm
        return network_response(self.type_id, p, f, a, z, load_f)

def calibration_from_file(type_id, folder):
    from execution.highpass.adapter import read
    result = read(folder / 'measurement.json')
    eta = OTA(**result['eta'])
    table = physical.table(folder, 'descriptor_ac.dat', 5)
    return (eta, TopologyCalibration(eta, tuple(table[:, 0]), tuple(table[:, 1] + 1j * table[:, 2]), tuple(table[:, 3] + 1j * table[:, 4]), type_id))

def prediction_error(calibration, eta, passives, f, h, case, *, floor=0.01):
    """Same M1 trust criterion/window as T1; signed topology-correct ideal reference."""
    from numerics.metrics.response import ideal_response
    keep = (f >= case['f0_hz'] * 0.1) & (f <= case['f0_hz'] * 10)
    if np.count_nonzero(keep) < 20:
        raise ValueError('Insufficient independent measured trust band')
    ff = f[keep]
    target = ideal_response(ff, physical.data.response_profile(case))
    predicted = calibration.response(passives, eta, ff, case['load_cap_pF'] * 1e-12)
    return float(np.max(abs(h[keep] - predicted) / np.maximum(abs(target), floor)))
