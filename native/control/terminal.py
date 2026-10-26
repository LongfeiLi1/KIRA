"""Explicit terminal-only engineering control; original loop bytecode reused."""
import json
import types
from control.interface_loop import run_loop as original_loop
from control.recovery import Controller
from backends.acceptance import initial_pass
from control.decisions import validate_decision

def terminal_decision(state, backend):
    if state['allowed_actions'] != ['finish'] or not initial_pass(backend):
        return None
    row = backend.current
    if row.get('measurement_status') != 'valid' or not all((row['evidence'].get(k) for k in ('performance', 'matrix'))):
        return None
    if not state['evidence_ids'] or not state['knowledge_ids']:
        return None
    raw = json.dumps({'schema_version': 1, 'state_id': state['state_id'], 'diagnosis': {'category': 'success', 'summary': 'Terminal-only engineering control: current complete measurement passes; original independent endpoint still required.', 'evidence_ids': [state['evidence_ids'][0]]}, 'action': {'name': 'finish', 'args': {}}, 'knowledge_ids': [state['knowledge_ids'][0]]})
    return (validate_decision(raw, state), raw)

class TerminalController(Controller):

    def __init__(self, enabled):
        super().__init__(enabled)
        self.terminal_controls = 0

    def choose(self, state, backend, client, record, persist):
        selected = terminal_decision(state, backend)
        if selected is None:
            return super().choose(state, backend, client, record, persist)
        decision, raw = selected
        record.update(attempts=[], recovery_trigger=None, decision_source='deterministic_terminal_control', decision=decision, terminal_control_raw=raw, terminal_control_trigger='only_finish_and_complete_current_measurement_PASS', terminal_control_changes_stopping_policy=True, provider_answer_used=False, provider_request_dispatched=False)
        self.terminal_controls += 1
        persist()
        return decision

def run_loop(backend, client, *, identity=None):
    instances = []

    def controller_factory(enabled):
        instance = TerminalController(enabled)
        instances.append(instance)
        return instance
    namespace = dict(original_loop.__globals__, Controller=controller_factory)
    isolated = types.FunctionType(original_loop.__code__, namespace, original_loop.__name__, original_loop.__defaults__, original_loop.__closure__)
    isolated.__kwdefaults__ = dict(original_loop.__kwdefaults__)
    result = isolated(backend, client, fallback_enabled=True, current_pass_shortcut_enabled=False, identity=identity)
    count = sum((c.terminal_controls for c in instances))
    result.update(method_version='provider_repair', terminal_control_count=count, terminal_control_enabled=True, stopping_policy_change='terminal-only measurement PASS + allowed exactly finish', actual_transport_retries=client.actual_retries, max_case_transport_retries=1, terminal_control_success=bool(result.get('method_success') and count), result_selection_scope='repair_view_not_one_uniform_45_case_run')
    if count:
        result.update(completion_path='deterministic_terminal_control', early_stop_path='terminal_only_current_measurement_PASS', pure_llm_success=False, corrected_llm_success=False, rule_fallback_success=False)
    return result
