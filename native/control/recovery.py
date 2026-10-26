"""Finite decision correction; illegal raw is never edited or executed."""
import json
import time
from control.decisions import validate_decision
from control.decisions import DecisionError
from provider.recovery import ProviderFailure
from evidence import clean
from evidence import sha
from evidence import save
MAX_CORRECTIONS = 3
MAX_RULE_ACTIONS = 1

class RecoveryFailure(DecisionError):
    pass

def api_remaining(backend, client):
    ledger = getattr(client, 'ledger', None)
    prior = getattr(ledger, 'prior_calls', 0)
    return max(0, backend.engine.protocol['max_llm_calls_per_case'] - prior - len(backend.engine.api_calls))

def budget_view(backend, client):
    e = backend.engine
    limits = getattr(e, 'limits', {})
    cache, slots = (getattr(e, 'cache', {}), getattr(e, 'slots', []))
    search = [s for s in slots if not s.get('endpoint', False)]
    used = {'candidates': len(cache), 'search_process_slots': len(search), 'search_analyses': sum((s.get('analysis_count', 0) for s in search)), 'API_calls_new': len(e.api_calls), 'launch_attempts': len(getattr(e, 'attempts', []))}
    keys = {'candidates': 'max_new_candidates', 'search_process_slots': 'search_base_process_slots', 'search_analyses': 'search_base_analysis_allowance'}
    remain = {k: max(0, limits[v] - used[k]) if type(limits.get(v)) is int else None for k, v in keys.items()}
    remain['API_calls'] = api_remaining(backend, client)
    totals = {}
    for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        vals = [r.get(k) for r in e.api_calls]
        totals[k] = {'known_subtotal': sum((v for v in vals if type(v) is int and v >= 0)), 'complete': all((type(v) is int and v >= 0 for v in vals))}
    return {'used': used, 'remaining': remain, 'usage': totals, 'endpoint_reserve_preserved': True, 'endpoint_slots_already_used': sum((bool(s.get('endpoint')) for s in slots)), 'endpoint_already_started': getattr(e, 'endpoint', None) is not None}

def parameter_axes(state):
    """Read current family fields; never mutate state or broaden validator axes."""
    if state['case']['type_id'] == 'T1':
        from backends.lowpass_contracts import AXES
        parameters = state['observation']['parameters']
        if not isinstance(parameters, dict) or any((axis not in parameters for axis in AXES)):
            raise ValueError('feedback_parameter_schema_invalid')
        return list(AXES)
    names = state['parameter_names']
    if not isinstance(names, list) or not all((isinstance(axis, str) for axis in names)):
        raise ValueError('feedback_parameter_schema_invalid')
    return list(names)

def feedback_for(state, error, raw):
    args = {}
    for name in state['allowed_actions']:
        if name in ('rc_step', 'coupled_step'):
            args[name] = {'exact_keys': ['radius'], 'radius': ['small', 'normal']}
        elif name == 'probe':
            args[name] = {'exact_keys': ['axis', 'direction'], 'axis': parameter_axes(state), 'direction': [-1, 1], 'direction_type': 'integer, not boolean'}
        else:
            args[name] = {'exact_keys': []}
    return {'feedback_kind': 'invalid_decision_correction', 'error_code': error.code, 'nested_error_code': error.nested, 'error_explanation': 'Rejected without execution: ' + error.code.replace('_', ' ') + ('; nested: ' + error.nested.replace('_', ' ') if error.nested else ''), 'current_state_id': state['state_id'], 'state_binding_instruction': 'Copy this exact state_id. Never substitute step/stage/observation numbers.', 'allowed_actions': state['allowed_actions'], 'argument_contract': args, 'allowed_evidence_ids': state['evidence_ids'], 'allowed_knowledge_ids': state['knowledge_ids'], 'required_fields': ['schema_version', 'state_id', 'diagnosis', 'action', 'knowledge_ids'], 'schema_version': 1, 'diagnosis_contract': {'exact_keys': ['category', 'summary', 'evidence_ids'], 'category': ['model_mismatch', 'realization_gap', 'planning_gap', 'constraints', 'success', 'unknown'], 'summary_length': [1, 1600], 'evidence_ids': 'nonempty subset of allowed_evidence_ids'}, 'knowledge_ids_contract': 'nonempty unique subset of allowed_knowledge_ids', 'previous_raw_sha256': sha(raw.encode()), 'previous_raw_excerpt': clean(raw[:32768]), 'instruction': 'Return a fresh bounded JSON decision; no extra keys, Markdown, code or circuit values. The unchanged original system contract also applies.'}

def diagnostic_view(state):
    """Read the canonical diagnostic dict; do not alter state or infer trust."""
    fields = [key for key in ('diagnostics', 'diagnostic') if key in state]
    for key in fields:
        value = state[key]
        if value is not None and (not isinstance(value, dict)):
            raise RecoveryFailure('diagnostic_field_malformed')
        if isinstance(value, dict) and 'model_trusted' in value and (type(value['model_trusted']) is not bool):
            raise RecoveryFailure('diagnostic_model_trusted_malformed')
    if len(fields) == 2 and state['diagnostics'] != state['diagnostic']:
        raise RecoveryFailure('diagnostic_fields_conflict')
    value = state[fields[0]] if fields else None
    return {} if value is None else value

def select_rule(state, backend, trigger):
    allowed = state['allowed_actions']
    diagnostic = diagnostic_view(state)
    trusted = diagnostic.get('model_trusted') is True
    rc = (state.get('rc_only_assessment') or {}).get('feasible_with_margin') is True
    from backends.acceptance import initial_pass
    if 'finish' in allowed and initial_pass(backend):
        name, args, rule = ('finish', {}, 'current_full_measurement_pass_after_rejections')
    elif trusted and rc and ('rc_step' in allowed) and backend.room(3, 13, 2):
        name, args, rule = ('rc_step', {'radius': 'small'}, 'trusted_rc_feasible')
    elif trusted and 'coupled_step' in allowed and backend.room(4, 15, 3):
        name, args, rule = ('coupled_step', {'radius': 'small'}, 'trusted_coupled')
    elif diagnostic and 'calibrate' in allowed and backend.room(3, 13, 2):
        name, args, rule = ('calibrate', {}, 'independent_calibration')
    else:
        raise RecoveryFailure('no_safe_rule_action')
    if not state['evidence_ids'] or not state['knowledge_ids']:
        raise RecoveryFailure('no_rule_evidence')
    raw = json.dumps({'schema_version': 1, 'state_id': state['state_id'], 'diagnosis': {'category': 'unknown', 'summary': 'Deterministic rule fallback after contract rejection: ' + trigger, 'evidence_ids': [state['evidence_ids'][0]]}, 'action': {'name': name, 'args': args}, 'knowledge_ids': [state['knowledge_ids'][0]]})
    return (validate_decision(raw, state), rule, raw)

class Controller:

    def __init__(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('fallback_enabled must be boolean')
        self.enabled, self.rule_actions = (enabled, 0)

    def choose(self, state, backend, client, record, persist):
        record.update(attempts=[], recovery_trigger=None, decision_source=None)
        error, raw = (None, None)
        for index in range(1 + (MAX_CORRECTIONS if self.enabled else 0)):
            if index and api_remaining(backend, client) == 0:
                record['correction_stop'] = 'API_budget_exhausted'
                break
            kind = 'original' if index == 0 else 'repair_attempt%d' % index
            record['active_phase'] = 'correction_feedback' if index else 'original_request_preparation'
            feedback = feedback_for(state, error, raw) if index else None
            before = budget_view(backend, client)
            if feedback is not None:
                feedback['budget_remaining'] = before['remaining']
            attempt = {'request_kind': kind, 'feedback': feedback, 'budget_before': before, 'status': 'pending'}
            record['attempts'].append(attempt)
            persist()
            begin = time.monotonic()
            try:
                if api_remaining(backend, client) == 0:
                    raise ProviderFailure('cumulative_case_api_cap_reached')
                record['active_phase'] = 'provider_decision'
                raw = client.decide(state) if index == 0 else client.decide(state, feedback=feedback, request_kind=kind)
                record['active_phase'] = 'raw_persistence'
                save(backend.out / ('decision_raw_%03d%s.json' % (record.get('step', 0), '' if index == 0 else '_repair%d' % index)), {'content': raw})
                attempt['raw'] = clean(raw)
                attempt['raw_sha256'] = sha(raw.encode())
                record['active_phase'] = 'strict_decision_validation'
                decision = validate_decision(raw, state)
                attempt.update(status='validated', validation_error=None)
                record['decision_source'] = 'llm' if index == 0 else 'corrected_llm'
                record['decision'] = decision
                return decision
            except DecisionError as exc:
                error = exc
                attempt.update(status='decision_rejected', validation_error={'code': exc.code, 'nested': exc.nested})
                record.setdefault('initial_recovery_trigger', exc.code)
                record['recovery_trigger'] = exc.code
                if not self.enabled:
                    raise
            except ProviderFailure as exc:
                attempt.update(status='provider_failed', failure_code=exc.code)
                raise
            finally:
                attempt['wall_seconds'] = time.monotonic() - begin
                attempt['budget_after'] = budget_view(backend, client)
                start = before['used']['API_calls_new']
                attempt['API_cost_rows'] = clean(backend.engine.api_calls[start:])
                persist()
        if self.rule_actions >= MAX_RULE_ACTIONS:
            raise RecoveryFailure('rule_fallback_quota_exhausted')
        before = budget_view(backend, client)
        record['active_phase'] = 'rule_selection'
        decision, rule, rule_raw = select_rule(state, backend, error.code)
        self.rule_actions += 1
        record['active_phase'] = 'rule_feedback'
        record.update(decision_source='rule_fallback', decision=decision, rule_id=rule, rule_raw=rule_raw, rule_budget_before=before, rule_validation='passed_unchanged_contract', latest_error_feedback=feedback_for(state, error, raw))
        persist()
        return decision
