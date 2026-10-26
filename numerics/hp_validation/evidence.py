"""Fail-closed numerical adapters, with explicit unresolved real-data blockers."""
import numpy as np
from numerics.topology_stability.core import ContractError, check_linear_evidence
from .contract import explicit_hp_contract

def inspect_hp_linear_evidence(*, mapping_provenance, matrix_names_in_exporter_order, observed_names, c1_f, c2_f, g, jacobians, op_voltages, dc_trajectories, frequencies_hz, measured_ac):
    """Check supplied matrices, never certify or release a real Case.

mapping_provenance is a caller-owned independently reviewed evidence record.
This adapter cannot prove an arbitrary name/number table. Until a native-index
export or independently controlled mapping probe is reviewed, omit it: the
result is unknown. OP order, OP sorting, and guessed numbering are forbidden.
"""
    result = {'status': 'unknown', 'runnable': False, 'full_circuit_qualified': False, 'real_mapping_verified_by_this_module': False, 'discarded_dynamic_modes': 0}
    try:
        if not isinstance(mapping_provenance, dict):
            raise ContractError('independent matrix index/name evidence missing')
        if mapping_provenance.get('method') not in ('native_index_export', 'independent_mapping_probe'):
            raise ContractError('OP vector ordering and inferred numbering are not mapping evidence')
        for key in ('evidence_path', 'netlist_sha256', 'simulator_sha256', 'review_record'):
            if not isinstance(mapping_provenance.get(key), str) or not mapping_provenance[key]:
                raise ContractError('mapping evidence missing ' + key)
        layout, ports, anchors = explicit_hp_contract(matrix_names_in_exporter_order, c1_f, c2_f)
        linear = check_linear_evidence(g, jacobians, layout, observed_names, anchors, ports, op_voltages, dc_trajectories, frequencies_hz, measured_ac)
        result.update(status=linear['status'], linear=linear, mapping_provenance=mapping_provenance, reason='supplied local linear evidence checked; physical qualification remains separate')
    except (ContractError, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    return result

def highpass_step_metrics(time_s, output_v, *, dc_out_v, input_edge_time_s, tail_start_s, final_error_limit_v, tail_p2p_limit_v):
    """Measure HP return to its pre-step DC point, never use H(1 Hz) as gain.

Tail interval and limits must be explicitly supplied by the future frozen
validation plan. This is a waveform check, not a complete stability proof.
"""
    t, v = (np.asarray(time_s, float), np.asarray(output_v, float))
    values = (dc_out_v, input_edge_time_s, tail_start_s, final_error_limit_v, tail_p2p_limit_v)
    if any((not np.isfinite(x) for x in values)) or min(final_error_limit_v, tail_p2p_limit_v) <= 0:
        raise ContractError('finite explicit waveform conditions and positive limits required')
    if t.ndim != 1 or len(t) < 4 or v.shape != t.shape or (not np.all(np.isfinite(t))) or (not np.all(np.isfinite(v))) or np.any(np.diff(t) <= 0):
        raise ContractError('finite increasing aligned transient waveform required')
    if not t[0] < input_edge_time_s < tail_start_s < t[-1]:
        raise ContractError('waveform must cover pre-edge, edge and explicit late tail')
    tail = v[t >= tail_start_s]
    if len(tail) < 3:
        raise ContractError('insufficient transient tail samples')
    error, p2p = (float(abs(np.mean(tail) - dc_out_v)), float(np.ptp(tail)))
    return {'expected_final_v': float(dc_out_v), 'step_final_error_v': error, 'step_tail_peak_to_peak_v': p2p, 'step_status': 'pass' if error < final_error_limit_v and p2p < tail_p2p_limit_v else 'fail', 'full_circuit_qualified': False, 'runnable': False, 'reason': 'HP small step returns to original DC bias; DC normalization not used'}
