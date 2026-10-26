"""Provisional topology-specific response metrics; never full-circuit acceptance.

No simulator calls. Profiles encode the supplied text for engineering validation,
not frozen benchmark registration. Invalid measurements raise ValueError.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

@dataclass(frozen=True)
class ResponseProfile:
    profile_id: str
    kind: str
    polarity: int
    f0_hz: float
    q: float
    reference_hz: float
    passband_hz: tuple
    stopband_hz: tuple
    f0_relative_tolerance: float = 0.02
    q_relative_tolerance: float = 0.05
    gain_range: tuple = (0.95, 1.05)
    passband_max_loss_db: float = 1.0
    stopband_min_attenuation_db: float = 20.0

    def __post_init__(self):
        if self.kind not in ('lowpass', 'highpass') or self.polarity != 1:
            raise ValueError('Explicit supported response kind and signed polarity required')
        for name in ('f0_hz', 'q', 'reference_hz', 'f0_relative_tolerance', 'q_relative_tolerance'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('passband_hz', 'stopband_hz', 'gain_range'):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or (not 0 < bounds[0] < bounds[1]):
                raise ValueError(name + ' must be positive ordered finite bounds')
        if self.kind == 'highpass' and (not self.passband_hz[0] <= self.reference_hz <= self.passband_hz[1]):
            raise ValueError('Highpass requires finite in-passband reference')
        if not np.all(np.isfinite([self.passband_max_loss_db, self.stopband_min_attenuation_db])):
            raise ValueError('Attenuation limits must be finite')

def profile_for(type_id, level_id):
    """Return provisional text-spec profile, not permission to run a Case.

    Only the two noninverting Sallen-Key responses are available.
    """
    if type_id not in ('T1', 'T2') or level_id not in ('L1', 'L2', 'L3'):
        raise ValueError('Unknown provisional type or level')
    level = int(level_id[-1])
    f0 = (12648.820123726657, 25297.640247453313, 37946.46037117997)[level - 1]
    if type_id == 'T2':
        return ResponseProfile('provisional_T2_' + level_id, 'highpass', 1, f0, 0.7071067811865476, 40000.0 * level, (20000.0 * level, 100000.0 * level), (1.0, 3500.0 * level))
    return ResponseProfile('provisional_T1_' + level_id, 'lowpass', 1, f0, 0.7071067811865476, 1.0, (1.0, 8000.0 * level), (45000.0 * level, 100000000.0))

def _arrays(f_hz, h_complex):
    try:
        f = np.asarray(f_hz, dtype=float)
        h = np.asarray(h_complex, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('Frequency and complex response must be numeric') from exc
    if f.ndim != 1 or h.ndim != 1 or f.shape != h.shape or (len(f) < 3):
        raise ValueError('Matching one-dimensional frequency/response arrays required')
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(h)):
        raise ValueError('Nonfinite measured frequency/response')
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError('Strictly increasing positive frequencies required')
    with np.errstate(over='ignore', invalid='ignore'):
        magnitude = np.abs(h)
    if not np.all(np.isfinite(magnitude)) or np.any(magnitude <= 0):
        raise ValueError('Zero or nonfinite magnitude prevents log interpolation')
    return (f, h)

def _covered(f, points):
    if min(points) < f[0] or max(points) > f[-1]:
        raise ValueError('Frequency grid does not cover required band/reference boundaries')

def _mag_at(f, h, points):
    _covered(f, np.atleast_1d(points))
    return np.exp(np.interp(np.log(points), np.log(f), np.log(np.abs(h))))

def _band_magnitudes(f, h, bounds):
    lo, hi = bounds
    _covered(f, bounds)
    keep = (f > lo) & (f < hi)
    if np.count_nonzero((f >= lo) & (f <= hi)) < 2:
        raise ValueError('Insufficient measured samples in required band')
    return np.r_[_mag_at(f, h, lo), np.abs(h[keep]), _mag_at(f, h, hi)]

def ideal_response(f_hz, profile, *, gain=1.0, f0_hz=None, q=None):
    """Signed complex second-order reference, including HP s-squared numerator."""
    s = 1j * np.asarray(f_hz, dtype=float) / (profile.f0_hz if f0_hz is None else f0_hz)
    denominator = 1 + s / (profile.q if q is None else q) + s * s
    numerator = s * s if profile.kind == 'highpass' else np.ones_like(s)
    return profile.polarity * gain * numerator / denominator

def fit_response(f_hz, h_complex, profile):
    """Fit signed complex H, preserving legacy LP window/error weighting.

    Fit parameters are positive magnitude gain, f0 and Q; expected polarity is
    fixed in the complex model, never erased with abs(H). Model mismatch remains
    visible in fit_max_component_error. Stability requires independent MNA gates.
    """
    f, h = _arrays(f_hz, h_complex)
    lo, hi = (profile.f0_hz * 0.05, profile.f0_hz * 5)
    _covered(f, (lo, hi))
    keep = (f >= lo) & (f <= hi)
    if np.count_nonzero(keep) < 20:
        raise ValueError('Insufficient measured fitting band: at least 20 points required')
    ff, hh = (f[keep], h[keep])

    def residual(x):
        gain, f0, q = np.exp(x)
        model = ideal_response(ff, profile, gain=gain, f0_hz=f0, q=q)
        r = (model - hh) / np.maximum(np.abs(hh), 0.05)
        return np.r_[r.real, r.imag]
    fit = least_squares(residual, np.log([1.0, profile.f0_hz, profile.q]), bounds=(np.log([0.01, profile.f0_hz / 10, 0.1]), np.log([10.0, profile.f0_hz * 10, 10.0])), max_nfev=80)
    if not fit.success or not np.all(np.isfinite(fit.x)) or (not np.all(np.isfinite(fit.fun))):
        raise ValueError('Signed complex second-order fit failed to converge')
    gain, f0, q = np.exp(fit.x)
    return {'fitted_gain': float(gain), 'fitted_signed_gain': float(profile.polarity * gain), 'f0_hz': float(f0), 'q': float(q), 'fit_max_component_error': float(np.max(np.abs(fit.fun))), 'fit_band_hz': [lo, hi], 'fit_points': int(np.count_nonzero(keep)), 'fit_model': profile.kind + '_signed_complex_second_order'}

def band_metrics(f_hz, h_complex, profile):
    f, h = _arrays(f_hz, h_complex)
    reference = float(_mag_at(f, h, profile.reference_hz))
    pass_values = _band_magnitudes(f, h, profile.passband_hz)
    stop_values = _band_magnitudes(f, h, profile.stopband_hz)
    phase = float(np.interp(np.log(profile.reference_hz), np.log(f), np.unwrap(np.angle(h))))
    signed_real = reference * np.cos(phase)
    return {'reference_gain': reference, 'reference_hz': profile.reference_hz, 'reference_real_component': float(signed_real), 'polarity_pass': bool(profile.polarity * signed_real > 0), 'expected_polarity': profile.polarity, 'passband_max_loss_db': float(max(0.0, np.max(-20 * np.log10(pass_values / reference)))), 'stopband_min_attenuation_db': float(np.min(-20 * np.log10(stop_values / reference))), 'passband_hz': list(profile.passband_hz), 'stopband_hz': list(profile.stopband_hz), 'attenuation_reference': 'measured magnitude at finite reference frequency', 'band_edge_interpolation': 'log_frequency_log_magnitude', 'droop_definition': 'one-sided normalized attenuation clipped at zero'}

def evaluate_response(f_hz, h_complex, profile, *, fit_max_component_error):
    """Return response-only gates. Caller must supply frozen fit-error limit.

    Invalid extraction raises ValueError (fail closed); a returned pass covers
    neither OP, power, PM, step, nor complete local closed-loop stability.
    """
    if not np.isfinite(fit_max_component_error) or fit_max_component_error <= 0:
        raise ValueError('Explicit positive finite fit-error gate required')
    result = band_metrics(f_hz, h_complex, profile)
    result.update(fit_response(f_hz, h_complex, profile))
    bounds = {'reference_gain': profile.gain_range, 'f0_hz': (profile.f0_hz * (1 - profile.f0_relative_tolerance), profile.f0_hz * (1 + profile.f0_relative_tolerance)), 'q': (profile.q * (1 - profile.q_relative_tolerance), profile.q * (1 + profile.q_relative_tolerance)), 'fit_max_component_error': (0.0, fit_max_component_error), 'passband_max_loss_db': (0.0, profile.passband_max_loss_db), 'stopband_min_attenuation_db': (profile.stopband_min_attenuation_db, float('inf'))}
    failed = [key for key, (lo, hi) in bounds.items() if not np.isfinite(result[key]) or not lo <= result[key] <= hi]
    if not result['polarity_pass']:
        failed.append('polarity')
    result.update(profile_id=profile.profile_id, failed_constraints=failed, response_status='fail' if failed else 'pass', full_circuit_acceptance=False, registration_status='provisional_not_frozen', runnable=False)
    return result
