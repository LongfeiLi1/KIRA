"""Reviewable decision guidance; no executor, provider call, or gate change."""
import hashlib
import json
import re
from provider.stream import stream_payload
FAILURE = re.compile('(?im)(^\\s*(warning:|error:|fatal|singular matrix|convergence failed)|timestep too small|simulation\\(s\\) aborted)')

def failure_context(observation, attempt):
    """Extract actual saved diagnostics without inventing a physical conclusion."""
    if observation.get('measurement_status') == 'valid':
        return None
    if not isinstance(attempt, dict) or not isinstance(attempt.get('log'), str):
        return None
    log = attempt['log']
    matches = [line for line in log.splitlines() if FAILURE.search(line)]
    if not matches:
        return None
    return {'candidate_id': observation['candidate_id'], 'measurement_status': observation.get('measurement_status'), 'saved_log_sha256': hashlib.sha256(log.encode()).hexdigest(), 'diagnostics': matches[:8], 'diagnostics_truncated': len(matches) > 8, 'acceptance': 'invalid measurement; no stability or circuit PASS inferred', 'numerical_vs_physical_cause': 'not established by these diagnostics alone'}

def build_payload(state, protocol, feedback=None, *, runtime_authority=None, context=None):
    """Leave the original state and strict validator binding byte-for-byte intact.

    A future sealed launcher must supply verified current release authority.
    This module itself provides no launch authorization and no live entry.
    """
    p = stream_payload(state, protocol, feedback)
    a = runtime_authority
    if not isinstance(a, dict) or a.get('source_fingerprinted') is not True or a.get('execution_authorized') is not True or (a.get('case_id') != state['case']['case_id']):
        raise ValueError('verified_current_release_authority_required')
    if context is not None and (context.get('candidate_id') != state['observation']['candidate_id'] or state['observation'].get('measurement_status') == 'valid'):
        raise ValueError('measurement_context_candidate_mismatch')
    note = {'decision_guidance_version': 'measurement_guidance_v1', 'state_id': state['state_id'], 'runtime_authority': 'The caller verified a current case-specific execution release. Historical manifest_runnable/development_only fields describe registration provenance; they do not cancel this release or authorize any additional action.', 'measurement_semantics': 'An invalid complete measurement is not circuit PASS. Simulator returncode 0 does not certify convergence, complete trajectories, stability, power, or endpoint acceptance. All original physical gates still apply.', 'finish_semantics': 'finish is permitted abandonment when no valid measured candidate exists. It cannot prove success or force an endpoint. Keep the option to abandon; do not claim an invalid measurement establishes infeasibility.', 'search_guidance': 'When measurement is invalid and probe is allowed, consider a bounded legal probe within the published domains and remaining budgets to seek valid complete evidence before abandoning. Do not repeat the same cached candidate as a new measurement, invent metrics, or use an unauthorized action. No axis, direction, or successful outcome is prescribed.', 'strict_protocol': 'Use the unchanged full state_id and existing evidence_ids, knowledge_ids, and allowed_actions. Historical step labels are not state IDs.', 'saved_measurement_diagnostics': context}
    p['messages'].insert(1, {'role': 'system', 'content': json.dumps(note, allow_nan=False)})
    return p
