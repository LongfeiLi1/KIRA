"""Finite difference Jacobian and bounded projected steps in log(q/qref)."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LogDomain:
    reference: tuple
    lower: tuple
    upper: tuple

    def arrays(self):
        ref, lo, hi = (np.asarray(v, float) for v in (self.reference, self.lower, self.upper))
        if ref.ndim != 1 or len(ref) == 0 or lo.shape != ref.shape or (hi.shape != ref.shape):
            raise ValueError('Domain shapes must agree')
        if not all((np.all(np.isfinite(v)) and np.all(v > 0) for v in (ref, lo, hi))) or np.any(lo >= hi):
            raise ValueError('Finite positive nondegenerate development bounds required')
        return (ref, lo, hi)

    def encode(self, q):
        ref, lo, hi = self.arrays()
        q = np.asarray(q, float)
        if q.shape != ref.shape or not np.all(np.isfinite(q)) or np.any(q < lo) or np.any(q > hi):
            raise ValueError('Point outside development domain')
        return np.log(q / ref)

    def decode(self, z):
        ref, lo, hi = self.arrays()
        z = np.asarray(z, float)
        if z.shape != ref.shape or not np.all(np.isfinite(z)) or np.any(z < np.log(lo / ref) - 1e-12) or np.any(z > np.log(hi / ref) + 1e-12):
            raise ValueError('Log point outside development domain')
        return np.clip(ref * np.exp(z), lo, hi)

def log_jacobian(constraints, q, domain: LogDomain, epsilon=0.0001):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite difference increment required')
    z = domain.encode(q)
    ref, lo, hi = domain.arrays()
    base = np.asarray(constraints(np.asarray(q, float)), float)
    if base.ndim != 1 or not len(base) or (not np.all(np.isfinite(base))):
        raise ValueError('Missing constraints')
    j = np.empty((len(base), len(z)))
    for k in range(len(z)):
        zm = z.copy()
        zp = z.copy()
        zm[k] = max(z[k] - epsilon, np.log(lo[k] / ref[k]))
        zp[k] = min(z[k] + epsilon, np.log(hi[k] / ref[k]))
        fm = np.asarray(constraints(domain.decode(zm)), float)
        fp = np.asarray(constraints(domain.decode(zp)), float)
        if fm.shape != base.shape or fp.shape != base.shape or (not np.all(np.isfinite(fm))) or (not np.all(np.isfinite(fp))):
            raise ValueError('Invalid finite difference observations')
        j[:, k] = (fp - fm) / (zp[k] - zm[k])
    return j

@dataclass(frozen=True)
class Proposal:
    values: tuple
    delta_z: tuple
    objective_before: float
    objective_after: float
    iterations: int
    predicted_only: bool = True

def bounded_update(q, domain, c, jacobian, *, trusted, scale, delta, margin, regularization, iterations=80):
    if not trusted:
        raise ValueError('Refuse update without measured model trust')
    if not isinstance(iterations, int) or not 1 <= iterations <= 1000:
        raise ValueError('Bounded iteration count required')
    if not np.isfinite(delta) or delta <= 0 or (not np.isfinite(regularization)) or (regularization <= 0):
        raise ValueError('Positive trust radius and regularization required')
    z = domain.encode(q)
    ref, lo, hi = domain.arrays()
    c = np.asarray(c, float)
    j = np.asarray(jacobian, float)
    d = np.asarray(scale, float)
    m = np.asarray(margin, float)
    if c.ndim != 1 or len(c) == 0 or j.shape != (len(c), len(z)) or (d.shape != z.shape) or (m.shape != c.shape):
        raise ValueError('Constraint/Jacobian/scale/margin shapes disagree')
    if not all((np.all(np.isfinite(v)) for v in (c, j, d, m))) or np.any(d <= 0) or np.any(m < 0):
        raise ValueError('Invalid update inputs')
    lower = np.maximum(np.log(lo / ref) - z, -delta / d)
    upper = np.minimum(np.log(hi / ref) - z, delta / d)
    step = np.zeros_like(z)

    def loss(x):
        return float(np.sum(np.maximum(c + j @ x + m, 0) ** 2) + regularization * np.sum((d * x) ** 2))
    before = loss(step)
    rate = 1 / (2 * (np.linalg.norm(j, 2) ** 2 + regularization * np.max(d * d)))
    for _ in range(iterations):
        gradient = 2 * j.T @ np.maximum(c + j @ step + m, 0) + 2 * regularization * d * d * step
        step = np.clip(step - rate * gradient, lower, upper)
    return Proposal(tuple(domain.decode(z + step)), tuple(step), before, loss(step), iterations)
