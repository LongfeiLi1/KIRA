"""Explicit MNA contracts. Unknown states are errors, never discarded modes.

Matrices use G x + C dx/dt = b. Names MUST be in the exporter-proven matrix
row/column order: sorting OP names does not establish that order. This module
does not infer names from component values or claim transistor qualification.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

class ContractError(ValueError):
    """Missing/unsupported evidence, which cannot count as a stability pass."""

@dataclass(frozen=True)
class GroundedSource:
    branch: str
    node: str
    incidence: int = 1

@dataclass(frozen=True)
class Layout:
    names: tuple[str, ...]
    dynamic_nodes: tuple[str, ...]
    sources: tuple[GroundedSource, ...]

    def validate(self):
        names = self.names
        if not names or any((not isinstance(n, str) or not n for n in names)):
            raise ContractError('nonempty matrix variable names required')
        if len(set(names)) != len(names):
            raise ContractError('duplicate matrix variable names')
        dynamic = list(self.dynamic_nodes)
        branches = [s.branch for s in self.sources]
        nodes = [s.node for s in self.sources]
        declared = dynamic + branches + nodes
        if not dynamic or len(set(declared)) != len(declared):
            raise ContractError('duplicate/overlapping dynamic or source variables')
        if set(declared) != set(names):
            raise ContractError('state coverage mismatch: missing=' + str(sorted(set(declared) - set(names))) + '; undeclared=' + str(sorted(set(names) - set(declared))))
        if any((type(s.incidence) is not int or s.incidence not in (-1, 1) for s in self.sources)):
            raise ContractError('grounded source incidence must be +1 or -1')
        return {name: i for i, name in enumerate(names)}

@dataclass(frozen=True)
class Ports:
    input_rhs: Mapping[str, float]
    output_voltage: Mapping[str, float]

@dataclass(frozen=True)
class ChargeAnchor:
    row: str
    column: str
    capacitance_f: float

def real_matrix(value, n, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ContractError(name + ' must be real')
    a = np.asarray(raw, dtype=float)
    if a.shape != (n, n) or not np.all(np.isfinite(a)):
        raise ContractError(name + ' dimension/nonfinite mismatch')
    return a

def validate_matrices(g, c, layout, observed_names):
    index = layout.validate()
    if tuple(observed_names) != layout.names:
        raise ContractError('exported matrix name/order coverage mismatch')
    n = len(index)
    g, c = (real_matrix(g, n, 'G'), real_matrix(c, n, 'C'))
    for source in layout.sources:
        branch, node = (index[source.branch], index[source.node])
        row = np.zeros(n)
        row[node] = source.incidence
        if not np.array_equal(g[branch], row) or not np.array_equal(g[:, branch], row):
            raise ContractError('source is not an exact pure grounded incidence: ' + source.branch)
        if np.any(c[branch]) or np.any(c[:, branch]):
            raise ContractError('source branch has charge: ' + source.branch)
    return (g, c, index)

def port_vectors(layout, ports):
    index = layout.validate()
    branch_names = {s.branch for s in layout.sources}
    vectors = []
    for name, weights in [('input_rhs', ports.input_rhs), ('output_voltage', ports.output_voltage)]:
        if not weights:
            raise ContractError('empty ' + name)
        v = np.zeros(len(index))
        for node, weight in weights.items():
            if node not in index:
                raise ContractError('missing port variable: ' + node)
            if isinstance(weight, (bool, complex)) or not np.isfinite(weight) or weight == 0:
                raise ContractError('finite nonzero real port weight required')
            if name == 'output_voltage' and node in branch_names:
                raise ContractError('voltage output mapped to source branch')
            v[index[node]] = weight
        vectors.append(v)
    return tuple(vectors)

def ac_response(g, c, layout, observed_names, ports, frequencies_hz):
    g, c, _ = validate_matrices(g, c, layout, observed_names)
    rhs, output = port_vectors(layout, ports)
    f = np.asarray(frequencies_hz, dtype=float)
    if f.ndim != 1 or len(f) == 0 or (not np.all(np.isfinite(f))) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ContractError('positive strictly increasing AC frequencies required')
    return np.array([output @ np.linalg.solve(g + 2j * np.pi * ff * c, rhs) for ff in f])

def match_spectra(first, second):
    if len(first) != len(second):
        raise ContractError('mode count changed')
    cost = abs(first[:, None] - second[None, :]) / np.maximum(abs(first[:, None]), 1.0)
    rows, columns = linear_sum_assignment(cost)
    return float(np.max(cost[rows, columns]))

def local_modes(g, c, layout, observed_names):
    g, c, index = validate_matrices(g, c, layout, observed_names)
    retained = [index[n] for n in layout.dynamic_nodes]
    gr, cr = (g[np.ix_(retained, retained)], c[np.ix_(retained, retained)])
    sv = np.linalg.svd(cr, compute_uv=False)
    if sv[0] == 0 or sv[-1] <= sv[0] * 1e-12:
        raise ContractError('dynamic charge rank/conditioning unresolved; no state removed')
    spectra, residuals = ([], [])
    for scale in (10000.0, 100000000.0):
        row = np.maximum(np.max(abs(gr), axis=1), np.max(abs(cr) * scale, axis=1))
        w, vectors = eig(-gr / row[:, None], cr * scale / row[:, None], homogeneous_eigvals=True)
        if np.any(w[1] == 0):
            raise ContractError('unexplained infinite mode; no state removed')
        poles = w[0] / w[1] * scale
        if len(poles) != len(retained) or not np.all(np.isfinite(poles)):
            raise ContractError('incomplete/nonfinite dynamic spectrum')
        residuals.extend((float(np.linalg.norm(gr @ x + p * cr @ x) / ((np.linalg.norm(gr) + abs(p) * np.linalg.norm(cr)) * np.linalg.norm(x))) if np.linalg.norm(gr) + abs(p) * np.linalg.norm(cr) else 0.0 for p, x in zip(poles, vectors.T)))
        spectra.append(poles)
    disagreement = match_spectra(*spectra)
    if disagreement > 1e-05 or max(residuals) > 1e-09:
        raise ContractError('eigenpair residual/coordinate scale check failed')
    poles = spectra[-1]
    if np.any(poles.real > 0):
        status, reason = ('unstable', 'retained right-half-plane mode')
    elif np.any(poles.real == 0) or np.any(-poles.real / np.maximum(abs(poles), 1e-300) < 1e-05):
        status, reason = ('unknown', 'zero or numerically marginal mode')
    else:
        status, reason = ('stable', 'all declared independent dynamic modes are LHP')
    return {'status': status, 'reason': reason, 'scope': 'supplied local linear MNA matrices only', 'full_circuit_qualified': False, 'mna_dimension': len(g), 'dynamic_state_names': list(layout.dynamic_nodes), 'finite_mode_count': len(poles), 'eliminated_algebraic_variables': [n for s in layout.sources for n in (s.node, s.branch)], 'discarded_dynamic_modes': 0, 'poles_rad_s': [[float(p.real), float(p.imag)] for p in sorted(poles, key=abs)], 'charge_condition_number': float(sv[0] / sv[-1]), 'maximum_eigenpair_residual': max(residuals), 'scale_match': disagreement}

def recover_charge(g, jacobians, layout, observed_names, anchors):
    """Recover C from >=2 BE Jacobians using declared physical C entries.

    Each anchor must be the TOTAL C[row,column], including any parasitics.
    No Cf, node-a, resistance value or topology-specific identity is inferred.
    """
    index = layout.validate()
    if tuple(observed_names) != layout.names:
        raise ContractError('exported matrix name/order coverage mismatch')
    g = real_matrix(g, len(index), 'G')
    if len(jacobians) < 2 or not anchors:
        raise ContractError('two independent Jacobians and charge anchors required')
    coordinates = []
    for a in anchors:
        if a.row not in index or a.column not in index:
            raise ContractError('missing charge anchor node')
        if not np.isfinite(a.capacitance_f) or a.capacitance_f == 0:
            raise ContractError('nonzero finite charge anchor required')
        coordinates.append((index[a.row], index[a.column], a.capacitance_f))
    if len({(r, c) for r, c, _ in coordinates}) != len(coordinates):
        raise ContractError('duplicate charge anchor')
    charges, alphas = ([], [])
    for j in jacobians:
        delta = real_matrix(j, len(index), 'Jacobian') - g
        estimates = np.array([delta[r, c] / value for r, c, value in coordinates])
        alpha = float(estimates[0])
        if not np.isfinite(alpha) or alpha <= 0 or (not np.allclose(estimates, alpha, rtol=1e-06, atol=0)):
            raise ContractError('charge anchors disagree on positive integration coefficient')
        c = delta / alpha
        validate_matrices(g, c, layout, observed_names)
        charges.append(c)
        alphas.append(alpha)
    if len({round(float(np.log10(a)), 8) for a in alphas}) < 2:
        raise ContractError('Jacobian coefficients are not independent')
    norm = np.linalg.norm(charges[-1])
    error = max((float(np.linalg.norm(c - charges[-1]) / norm) for c in charges))
    if error > 1e-05:
        raise ContractError('two-step charge matrices disagree')
    spectra = [np.array([complex(*p) for p in local_modes(g, c, layout, observed_names)['poles_rad_s']]) for c in charges]
    mode_error = max((match_spectra(s, spectra[-1]) for s in spectra))
    if mode_error > 0.001:
        raise ContractError('two-step mode spectra disagree')
    return (charges[-1], {'integration_coefficients_per_s': alphas, 'charge_relative_difference': error, 'mode_relative_difference': mode_error})

def check_linear_evidence(g, jacobians, layout, observed_names, anchors, ports, op_voltages, dc_trajectories, frequencies_hz, measured_ac):
    """Check supplied DC/AC/extraction evidence; never a full Case gate.

    The caller still owns exporter-name provenance, pinned model qualification,
    simulator-warning rejection, transient recovery, and all performance gates.
    ContractError is intentionally not converted into a successful result.
    """
    index = layout.validate()
    voltages = set(index) - {s.branch for s in layout.sources}
    if set(op_voltages) != voltages:
        raise ContractError('OP voltage coverage mismatch')
    if len(dc_trajectories) != len(jacobians):
        raise ContractError('each Jacobian requires a constant-bias trajectory')
    drift = 0.0
    for trajectory in dc_trajectories:
        if set(trajectory) != voltages:
            raise ContractError('DC trajectory node coverage mismatch')
        lengths = set()
        for node in voltages:
            data = np.asarray(trajectory[node], dtype=float)
            if data.ndim != 1 or len(data) < 2 or (not np.all(np.isfinite(data))) or (not np.isfinite(op_voltages[node])):
                raise ContractError('invalid constant-bias trajectory')
            lengths.add(len(data))
            drift = max(drift, float(np.max(abs(data - op_voltages[node]))))
        if len(lengths) != 1:
            raise ContractError('DC trajectory sample counts differ')
    if drift > 1e-07:
        raise ContractError('extraction changed DC operating point')
    c, extraction = recover_charge(g, jacobians, layout, observed_names, anchors)
    result = local_modes(g, c, layout, observed_names)
    predicted = ac_response(g, c, layout, observed_names, ports, frequencies_hz)
    observed = np.asarray(measured_ac, dtype=complex)
    if observed.shape != predicted.shape or not np.all(np.isfinite(observed)) or len(observed) < 3:
        raise ContractError('independent AC response missing/incomplete')
    error = float(np.max(abs(predicted - observed) / np.maximum(abs(observed), 0.01)))
    if error > 0.0001:
        raise ContractError('explicit mapping fails independent AC reconstruction')
    result.update(extraction=extraction, maximum_dc_drift_v=drift, ac_reconstruction_error=error)
    return result
