"""Same decision data for both controllers; full evidence retained separately."""
import copy
import json
import time
from types import MethodType
from control.policy import digest
STABILITY_DETAILS = ('op_vector_names', 'poles_rad_s', 'matrix_details', 'model_contract')

def compact(value):
    if isinstance(value, list):
        return [compact(v) for v in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {}
    for k, v in value.items():
        if k == 'stability_check' and isinstance(v, dict):
            result[k] = {a: compact(b) for a, b in v.items() if a not in STABILITY_DETAILS}
            result[k]['details_retained_in_original_snapshot'] = list(STABILITY_DETAILS)
        elif k == 'tool_history':
            result['tool_history_summary'] = {'event_count': len(v) if isinstance(v, list) else None, 'full_history_retained': True}
        elif k == 'coupled_device_implementation_preview' and isinstance(v, dict):
            result[k] = {a: compact(b) for a, b in v.items() if a not in ('seed_log_parameters', 'seed_parameters')}
        else:
            result[k] = compact(v)
    return result
INSTRUCTIONS = 'Choose a legal action using actual gates, separate model/realization/planning errors, local action outcomes and remaining budget. Tools own all circuit numbers. Predicted reduction alone does not show physical target reachability or complete PASS. Use calibration for measured model mismatch, RC when certified, coupled for supported joint repair, and bounded probes for invalid/unsupported measured gates. Do not repeat a failed identical action at unchanged state unless new evidence justifies it. Compare expected physical cost and risk; a cheap probe estimate does not establish its safety. Keep finish available for insufficient evidence. Do not read historical results or infer endpoint answers. All omitted matrix details remain in the original snapshot and do not alter hard acceptance.'

def install(backend):
    from evidence import save
    original_snapshot, original_execute = (backend.snapshot, backend.execute)
    backend.scheduler_original_execute = original_execute
    actions = []
    backend.scheduler_memory = actions

    def execute(self, decision):
        before = self.costs()
        start = time.monotonic()
        old = self.current
        try:
            result = original_execute(decision)
        except BaseException as exc:
            actions.append({'action': copy.deepcopy(decision['action']), 'candidate_id_before': old['candidate_id'], 'candidate_id_after': self.current['candidate_id'], 'status': 'exception', 'error_class': type(exc).__name__, 'wall_seconds': time.monotonic() - start, 'cost_before': before, 'cost_after': self.costs()})
            raise
        actions.append({'action': copy.deepcopy(decision['action']), 'candidate_id_before': old['candidate_id'], 'candidate_id_after': self.current['candidate_id'], 'measured_loss_before': old.get('loss'), 'measured_loss_after': self.current.get('loss'), 'rollback': result.get('rollback'), 'status': result.get('status', self.current.get('status')), 'wall_seconds': time.monotonic() - start, 'cost_before': before, 'cost_after': self.costs()})
        save(self.out / 'scheduler_action_memory.json', actions)
        return result
    backend.execute = MethodType(execute, backend)

    def snapshot(self, step):
        state = original_snapshot(step)
        full = copy.deepcopy(state)
        save(self.out / ('original_snapshot_%03d.json' % step), full)
        state = compact(state)
        state['scheduler_instructions'] = INSTRUCTIONS
        state['local_action_outcomes'] = copy.deepcopy(actions)
        feedback = self.adaptive_target_feedback
        state['dynamic_target_feedback'] = {k: copy.deepcopy(feedback[k]) for k in ('enabled', 'eta_penalty_weights', 'paid_realization_events', 'feedback_events', 'scope')}
        if state.get('diagnostic_policy_scope'):
            state.pop('dynamic_target_feedback', None)
        state['source_snapshot_sha256'] = digest(full)
        state['compact_state_bytes'] = {'original': len(json.dumps(full).encode()), 'compact_before_identity_fields': len(json.dumps(state).encode())}
        state.pop('state_id', None)
        state['state_id'] = digest(state)
        save(self.out / ('scheduler_snapshot_%03d.json' % step), state)
        return state
    backend.snapshot = MethodType(snapshot, backend)
