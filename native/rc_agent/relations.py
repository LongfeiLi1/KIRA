"""Fixed unity-gain SK wiring, SI units, ideal reference relations."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Passives:
    r1: float
    r2: float
    cg: float
    cf: float

    def __post_init__(self):
        if not all((math.isfinite(v) and v > 0 for v in self.values)):
            raise ValueError('R1,R2,Cg,Cf must be finite and positive in SI units')

    @property
    def values(self):
        return (self.r1, self.r2, self.cg, self.cf)

def frequency_grid(f_hz):
    f = np.asarray(f_hz, dtype=float)
    if f.ndim != 1 or len(f) == 0 or (not np.all(np.isfinite(f))) or np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError('Require finite strictly increasing nonnegative frequencies')
    return f

def ideal_parameters(p: Passives):
    t = math.sqrt(p.r1 * p.r2 * p.cg * p.cf)
    return (1 / t, t / (p.cg * (p.r1 + p.r2)))

def ideal_response(p: Passives, f_hz):
    s = 2j * np.pi * frequency_grid(f_hz)
    return 1 / (1 + s * p.cg * (p.r1 + p.r2) + s * s * p.r1 * p.r2 * p.cg * p.cf)

def equal_r_seed(omega0: float, q: float, cg: float):
    if not all((math.isfinite(v) and v > 0 for v in (omega0, q, cg))):
        raise ValueError('Positive finite ideal specifications required')
    r = 1 / (2 * q * omega0 * cg)
    return Passives(r, r, cg, 4 * q * q * cg)
