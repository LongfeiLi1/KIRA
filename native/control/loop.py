"""Unchanged correction, transport and endpoint loop; topology-neutral terminal proof."""
import json
from control.terminal import TerminalController as NativeTerminal
from backends.acceptance import initial_pass
from control.decisions import validate_decision
from provider.stream import injected

class TerminalController(NativeTerminal):

    def choose(self, state, backend, client, record, persist):
        if state['allowed_actions'] == ['finish'] and initial_pass(backend):
            raw = json.dumps({'schema_version': 1, 'state_id': state['state_id'], 'diagnosis': {'category': 'success', 'summary': 'Complete current physical gates passed; fresh independent endpoint still required.', 'evidence_ids': [state['evidence_ids'][0]]}, 'action': {'name': 'finish', 'args': {}}, 'knowledge_ids': [state['knowledge_ids'][0]]})
            decision = validate_decision(raw, state)
            self.terminal_controls += 1
            record.update(attempts=[], recovery_trigger=None, decision_source='deterministic_terminal_control', decision=decision, terminal_control_raw=raw, provider_answer_used=False, provider_request_dispatched=False)
            persist()
            return decision
        return super().choose(state, backend, client, record, persist)

def run_loop(backend, client, *, identity=None):
    from control.interface_loop import run_loop as original_loop
    instances = []

    def controller_factory(enabled):
        controller = TerminalController(enabled)
        instances.append(controller)
        return controller
    result = injected(original_loop, Controller=controller_factory)(backend, client, fallback_enabled=True, current_pass_shortcut_enabled=True, identity=identity)
    count = sum((c.terminal_controls for c in instances))
    result.update(terminal_control_count=count, terminal_control_enabled=True, terminal_control_success=bool(result.get('method_success') and count), actual_transport_retries=client.actual_retries, max_case_transport_retries=1, stopping_policy_change='shared complete current PASS shortcut; fresh endpoint still required', post_PASS_planning_diagnostics_deferral_enabled=True, post_PASS_planning_diagnostics_deferred=bool(backend.refinement_audit['terminal_PASS_diagnostic_deferrals']))
    return result
