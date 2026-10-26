"""Three separate diagnostic residuals; they are not causal attributions."""
from dataclasses import dataclass
import math
import numpy as np
from rc_agent.relations import Passives
from rc_agent.relations import frequency_grid

@dataclass(frozen=True)
class Response:
    candidate_id: str
    passives: Passives
    frequencies_hz: tuple
    values: tuple
    source: str

    def __post_init__(self):
        f = frequency_grid(self.frequencies_hz)
        v = np.asarray(self.values, complex)
        if not self.candidate_id or self.source not in {'full_spice', 'achieved_model', 'planned_model', 'target', 'synthetic_test'}:
            raise ValueError('Explicit candidate identity and response provenance required')
        if v.shape != f.shape or not np.all(np.isfinite(v)):
            raise ValueError('Missing/nonfinite response samples')

@dataclass(frozen=True)
class Residuals:
    model: tuple
    realization: tuple
    planning: tuple
    errors: tuple
    total_error: float
    measured: bool
    passives: Passives
    candidate_id: str

def decompose(full: Response, achieved: Response, planned: Response, target: Response, floor: float):
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('Positive response normalization floor required')
    for r in (achieved, planned, target):
        if r.frequencies_hz != full.frequencies_hz:
            raise ValueError('Frequency grids differ')
    for r in (achieved, planned):
        if r.passives != full.passives or r.candidate_id != full.candidate_id:
            raise ValueError('Models must use same actual passives and candidate as full circuit')
    if achieved.source != 'achieved_model' or planned.source != 'planned_model' or target.source != 'target':
        raise ValueError('Response roles are inconsistent')
    h, ha, hp, ht = (np.asarray(r.values, complex) for r in (full, achieved, planned, target))
    denom = np.maximum(np.abs(ht), floor)
    parts = (h - ha, ha - hp, hp - ht)
    errors = tuple((float(np.max(np.abs(r) / denom)) for r in parts))
    return Residuals(*(tuple(r) for r in parts), errors, float(np.max(np.abs(h - ht) / denom)), full.source == 'full_spice', full.passives, full.candidate_id)

def model_trusted(residuals: Residuals, max_model_error: float):
    if not math.isfinite(max_model_error) or max_model_error < 0:
        raise ValueError('Explicit nonnegative trust threshold required')
    return residuals.measured and residuals.errors[0] <= max_model_error
