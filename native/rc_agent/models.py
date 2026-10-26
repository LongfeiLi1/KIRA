"""One-pole unloaded VCVS plus output resistance; not a transistor DUT."""
from dataclasses import dataclass
import math
import numpy as np
from rc_agent.relations import Passives
from rc_agent.relations import frequency_grid

@dataclass(frozen=True)
class OTA:
    a0: float
    wp_rad_s: float
    ro_ohm: float

    def __post_init__(self):
        if not all((math.isfinite(v) and v > 0 for v in self.values)):
            raise ValueError('Positive finite unloaded OTA descriptors required')

    @property
    def values(self):
        return (self.a0, self.wp_rad_s, self.ro_ohm)

def sk_response(p: Passives, eta: OTA, f_hz, load_f=0.0):
    f = frequency_grid(f_hz)
    return network_response(p, f, eta.a0 / (1 + 2j * np.pi * f / eta.wp_rad_s), np.full(len(f), eta.ro_ohm), load_f)

def network_response(p, f_hz, gain, impedance, load_f=0.0):
    """Same outer nodal equations with independent measured amplifier A and Z."""
    if not math.isfinite(load_f) or load_f < 0:
        raise ValueError('External output load must be finite and nonnegative')
    frequencies = frequency_grid(f_hz)
    gain = np.asarray(gain, complex)
    impedance = np.asarray(impedance, complex)
    if gain.shape != frequencies.shape or impedance.shape != frequencies.shape or (not np.all(np.isfinite(gain))) or (not np.all(np.isfinite(impedance))) or np.any(abs(impedance) == 0):
        raise ValueError('Invalid amplifier response')
    result = []
    for f, a, z in zip(frequencies, gain, impedance):
        s = 2j * np.pi * f
        y = np.array([[1 / p.r1 + 1 / p.r2 + s * p.cf, -1 / p.r2, -s * p.cf], [-1 / p.r2, 1 / p.r2 + s * p.cg, 0], [-s * p.cf, -a / z, (1 + a) / z + s * (p.cf + load_f)]], complex)
        result.append(np.linalg.solve(y, np.array([1 / p.r1, 0, 0]))[2])
    out = np.asarray(result)
    if not np.all(np.isfinite(out)):
        raise ValueError('Nonfinite model response')
    return out

@dataclass(frozen=True)
class OTACalibration:
    """Unloaded OTA-only AC shape, never fit using full-filter responses.

    Local planned model assumes the measured shape persists in normalized pole
    coordinates. Fresh achieved shape and a separate full-filter trust check
    are required after sizing; this is not a validated extrapolation model.
    """
    anchor: OTA
    frequencies: tuple
    gain: tuple
    impedance: tuple

    def response(self, p, eta, f_hz, load_f=0.0):
        f = frequency_grid(f_hz)
        source = frequency_grid(self.frequencies)
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
        return network_response(p, f, a, z, load_f)
