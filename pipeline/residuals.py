"""Three residuals are observations with validity, never inferred success gates."""
import math
NAMES = ('model', 'realization', 'planning')

def components(diagnostic, *, family, disabled=False):
    if family!='sallen_key':raise ValueError('Unsupported physical family')
    d = diagnostic or {}
    out = {}
    for name in NAMES:
        value = d.get(name + '_error')
        valid = type(value) in (int, float) and math.isfinite(value) and (value >= 0)
        out[name] = dict(status='measured' if valid and (not disabled) else 'withheld_by_ablation' if disabled else 'unavailable', normalized_error=float(value) if valid and (not disabled) else None)
    return dict(schema_version='kira.three_residuals.v1', components=out, definitions=dict(model='H_SPICE-H_achieved', realization='H_achieved-H_planned', planning='H_planned-H_target'), normalization_floor=d.get('normalization_floor'), frequency_grid_sha256=d.get('frequency_grid_sha256'), trust_band_component_errors=None if disabled else d.get('trust_band_component_errors'), model_trusted=d.get('model_trusted'), acceptance='unchanged physical verifier; residuals never substitute for gates')
