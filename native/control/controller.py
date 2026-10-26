"""Predeclared full-action deterministic controller; no provider access."""
import json
import math
from control.decisions import validate_decision
from backends.acceptance import initial_pass
from control.recovery import diagnostic_view
from control.recovery import parameter_axes

class NoProvider:
    is_mock = False
    is_live_transport = False
    actual_retries = 0

    def decide(self, *a, **k):
        raise AssertionError('fixed_rule_must_not_dispatch_LLM')

class FixedRuleController:

    def __init__(self, enabled):
        self.enabled = enabled
        self.rule_actions = 0
        self.terminal_controls = 0
        self.last_loss = None
        self.stall = 0
        self.decision_attempts = 0
        self.used = set()

    def select(self, state, backend):
        allowed = state['allowed_actions']
        diagnostic = diagnostic_view(state)
        observation = state['observation']
        last = state.get('last_tool_result') or {}
        cid = observation['candidate_id']
        failed = last.get('status') == 'tool_failed' or last.get('rollback') is True
        prior = getattr(self, 'previous', None)
        if failed and prior is not None:
            self.used.add(prior)

        def unused(name, args):
            return (cid, name, json.dumps(args, sort_keys=True)) not in self.used
        if initial_pass(backend) and 'finish' in allowed:
            return ('finish', {}, 'complete_measurement_PASS_fresh_endpoint')
        loss = observation.get('loss')
        if isinstance(loss, (int, float)) and math.isfinite(loss):
            self.stall = 0 if self.last_loss is None or loss < self.last_loss - 1e-12 else self.stall + 1
            self.last_loss = loss if self.last_loss is None else min(loss, self.last_loss)
        else:
            self.stall += 1
        if self.stall >= 6 or self.decision_attempts >= 32:
            return ('finish', {}, 'six_decisions_without_measured_loss_improvement_or_policy_cap')
        trusted = diagnostic.get('model_trusted') is True
        if not trusted and 'calibrate' in allowed and unused('calibrate', {}):
            return ('calibrate', {}, 'independent_model_qualification')
        alternatives = (state.get('repair_guidance') or {}).get('alternatives', [])
        ranked = (state.get('repair_guidance') or {}).get('predicted_candidate_order', [])
        for selected in ranked:
            name, radius = (selected['action'], selected['radius'])
            args = {'radius': radius}
            detail = next((a for a in alternatives if a.get('action') == name and a.get('radius') == radius), {})
            readiness = (state.get('fixed_target_allocation_availability') or {}).get('by_radius', {}).get(radius)
            eligible = name in allowed and unused(name, args) and (readiness is None or readiness['available'])
            if trusted and eligible and (detail.get('predicted_reduction', 0) > 1e-12) and (name != 'rc_step' or detail.get('RC_certified_margin')):
                return (name, args, 'residual_sensitivity_ranked_bounded_repair')
        rc = (state.get('rc_only_assessment') or {}).get('feasible_with_margin') is True
        if trusted and rc and ('rc_step' in allowed):
            for radius in ('normal', 'small'):
                args = {'radius': radius}
                if unused('rc_step', args):
                    return ('rc_step', args, 'trusted_RC_feasible')
        if trusted and 'coupled_step' in allowed:
            for radius in ('normal', 'small'):
                args = {'radius': radius}
                readiness = (state.get('fixed_target_allocation_availability') or {}).get('by_radius', {}).get(radius)
                if unused('coupled_step', args) and (readiness is None or readiness['available']):
                    return ('coupled_step', args, 'trusted_coupled_allocation')
        if 'calibrate' in allowed and unused('calibrate', {}):
            return ('calibrate', {}, 'model_requalification_after_no_progress')
        if 'probe' in allowed:
            names = parameter_axes(state)
            metrics = observation.get('metrics', {})
            case = state['case']
            frequency = metrics.get('f0_hz')
            priorities = []
            limit = case.get('power_max_uW', math.inf) * 1e-06
            if metrics.get('power_w', 0) > limit:
                priorities.append(('current_scale', -1))
            if metrics.get('phase_margin_deg', 180) < 60:
                priorities.append(('cc_pf', 1))
            if isinstance(frequency, (int, float)) and frequency != case['f0_hz']:
                direction = 1 if frequency > case['f0_hz'] else -1
                priorities += [(n, direction) for n in names if n.startswith(('r', 'c')) and n not in ('current_scale', 'cc_pf')]
            priorities += [(n, sign) for n in names for sign in (1, -1)]
            for axis, direction in dict.fromkeys(priorities):
                if axis not in names:
                    continue
                value = observation['parameters'][axis] * math.exp(0.04 * direction)
                lo, hi = backend.bounds[axis]
                args = {'axis': axis, 'direction': direction}
                if lo <= value <= hi and unused('probe', args):
                    return ('probe', args, 'bounded_measurement_feedback_probe')
        if 'finish' not in allowed:
            raise ValueError('legal_finish_missing')
        return ('finish', {}, 'no_remaining_legal_untried_repair')

    def choose(self, state, backend, client, record, persist):
        name, args, rule = self.select(state, backend)
        if not state['evidence_ids'] or not state['knowledge_ids']:
            raise ValueError('same_required_rule_evidence_missing')
        raw = json.dumps({'schema_version': 1, 'state_id': state['state_id'], 'diagnosis': {'category': 'success' if initial_pass(backend) else 'unknown', 'summary': 'Predeclared deterministic controller: ' + rule, 'evidence_ids': [state['evidence_ids'][0]]}, 'action': {'name': name, 'args': args}, 'knowledge_ids': state['knowledge_ids']})
        decision = validate_decision(raw, state)
        terminal = name == 'finish' and initial_pass(backend)
        if terminal:
            self.terminal_controls += 1
        else:
            self.decision_attempts += 1
        self.previous = (state['observation']['candidate_id'], name, json.dumps(args, sort_keys=True))
        self.used.add(self.previous)
        self.rule_actions += int(name != 'finish')
        record.update(attempts=[], recovery_trigger=None, decision_source='fixed_terminal_control' if terminal else 'fixed_rule', policy_decision_attempt=not terminal, rule_id=rule, rule_raw=raw, decision=decision, provider_answer_used=False, provider_request_dispatched=False)
        persist()
        return decision

def run_loop(backend, client, *, identity=None):
    from provider.stream import injected
    from control.interface_loop import run_loop as native_loop
    instances = []

    def factory(enabled):
        controller = FixedRuleController(enabled)
        instances.append(controller)
        return controller
    native_initialize = backend.initialize
    from types import MethodType

    def initialize(self):
        backend.state.update(llm_enabled=False, mock_decisions=False, controller_kind='fixed_rule')
        return native_initialize()
    backend.initialize = MethodType(initialize, backend)
    try:
        result = injected(native_loop, Controller=factory)(backend, client, fallback_enabled=True, current_pass_shortcut_enabled=True, identity=identity)
    finally:
        backend.initialize = native_initialize
    result.update(llm_enabled=False, mock_decisions=False, real_provider_exercised=False, fixed_rule_controller=True, hidden_LLM_calls=0, fixed_rule_action_count=sum((c.rule_actions for c in instances)), policy_decision_attempt_count=sum((c.decision_attempts for c in instances)), terminal_control_count=sum((c.terminal_controls for c in instances)), pure_llm_success=False, corrected_llm_success=False, rule_fallback_success=False)
    if result['decision_count']:
        result.update(completion_path='fixed_rule', measurement_shortcut_success=False, fixed_rule_success=result.get('method_success', False))
    else:
        result['fixed_rule_success'] = False
    return result
