"""Measured acceptance plus bounded recovery; physical tools are unchanged."""
import time
from control.diagnostics import exception_view
from control.decisions import DecisionError
from provider.recovery import ProviderFailure
from control.recovery import Controller
from control.recovery import budget_view
from backends.acceptance import save
from backends.acceptance import ExecutionPaused
from backends.acceptance import CandidateBudgetExhausted
from backends.acceptance import ToolUnavailable
from backends.acceptance import initial_pass

def run_loop(backend, client, max_steps=None, *, fallback_enabled=False, current_pass_shortcut_enabled=False, identity=None):
    from runtime.integrity import verify
    verify()
    cap = backend.engine.protocol['max_llm_calls_per_case']
    max_steps = cap if max_steps is None else max_steps
    if type(max_steps) is not int or not 1 <= max_steps <= cap:
        raise ValueError('Decision count exceeds frozen API cap')
    if type(current_pass_shortcut_enabled) is not bool:
        raise ValueError('current_pass_shortcut_enabled must be boolean')
    identity = dict(identity or {})
    trace, controller = ([], Controller(fallback_enabled))
    phase, active_record, failures = ('initialization', None, [])
    backend.state = {'llm_enabled': not client.is_mock, 'mock_decisions': client.is_mock, 'decision_schema': 1, 'complete_loop_attempted': True}

    def note_failure(exc, at, record=None):
        failure = exception_view(exc, at, record)
        failures.append(failure)
        backend.stop = 'unexpected_exception: ' + failure['error_class'] + ' at ' + at
        backend.state['execution_failure'] = failures[0]
        if record is not None:
            record.update(status='unexpected_exception', execution_failure=failure)
        return failure

    def persist():
        try:
            save(backend.out / 'decisions.json', trace)
        except Exception as exc:
            note_failure(exc, 'decision_trace_persistence', active_record)
            raise
    try:
        backend.initialize()
        backend.state['initial_acceptance_short_circuit'] = initial_pass(backend)
        for step in range(max_steps):
            if initial_pass(backend) and (step == 0 or current_pass_shortcut_enabled):
                backend.state['current_acceptance_short_circuit'] = bool(step > 0 and current_pass_shortcut_enabled)
                backend.stop = 'Current full measurement passed; independent endpoint required'
                break
            phase = 'state_snapshot'
            state = backend.snapshot(step)
            record = {'step': step, 'state': state, 'status': 'decision_pending', 'budget_before': budget_view(backend, client)}
            active_record = record
            trace.append(record)
            persist()
            begin = time.monotonic()
            try:
                phase = 'decision_selection'
                decision = controller.choose(state, backend, client, record, persist)
                record['status'] = 'validated'
                record.pop('active_phase', None)
            except DecisionError as exc:
                record.update(failure_code=exc.code, nested_failure_code=exc.nested, status='rejected_fail_closed')
                raise
            except ProviderFailure as exc:
                record.update(failure_code=exc.code, status='provider_failed')
                raise
            except Exception as exc:
                note_failure(exc, record.get('active_phase', phase), record)
                raise
            finally:
                record['decision_seconds'] = time.monotonic() - begin
                persist()
            if decision['action']['name'] == 'finish':
                backend.stop = 'Validated decision requested independent endpoint'
                record['status'] = 'finish_requested'
                break
            record['tool_budget_before'] = budget_view(backend, client)
            try:
                phase = 'tool_execution'
                record['tool_result'] = backend.execute(decision)
                record['status'] = 'tool_completed'
                active_record = None
            except (ValueError, ToolUnavailable, RuntimeError) as exc:
                if isinstance(exc, (ExecutionPaused, CandidateBudgetExhausted)) or backend.engine.status == 'paused':
                    raise
                backend.last_result = {'status': 'tool_failed', 'error_class': type(exc).__name__, 'action': decision['action']['name'], 'reason': 'tool_exception: ' + type(exc).__name__}
                record.update(status='tool_failed', tool_result=backend.last_result)
                backend.event('failed_tool', backend.last_result)
                if record['decision_source'] == 'rule_fallback':
                    raise RuntimeError('rule_fallback_tool_failed') from None
            finally:
                record['budget_after'] = budget_view(backend, client)
                persist()
        else:
            backend.state['current_acceptance_short_circuit'] = bool(current_pass_shortcut_enabled and initial_pass(backend))
            backend.stop = 'Decision cap reached; independent endpoint required'
    except (DecisionError, ProviderFailure) as exc:
        backend.stop = 'fail_closed: ' + type(exc).__name__
        backend.state.update(decision_failure=type(exc).__name__, decision_failure_code=exc.code)
    except (ExecutionPaused, CandidateBudgetExhausted, ToolUnavailable, ValueError, RuntimeError) as exc:
        if not failures:
            backend.stop = 'execution_stopped: ' + type(exc).__name__
    except Exception as exc:
        if not failures:
            note_failure(exc, phase, active_record)
    finally:
        try:
            persist()
        except Exception:
            pass
        try:
            endpoint_before = budget_view(backend, client)
        except Exception as exc:
            note_failure(exc, 'endpoint_budget_before')
            endpoint_before = None
        try:
            result = backend.finish()
        except Exception as exc:
            note_failure(exc, 'endpoint_finalization')
            result = {'status': 'endpoint_unconfirmed', 'independent_endpoint_checked': False, 'reason': backend.stop}
        try:
            endpoint_after = budget_view(backend, client)
        except Exception as exc:
            note_failure(exc, 'endpoint_budget_after')
            endpoint_after = None
        try:
            calls = backend.engine.api_calls
            keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
            complete = all((all((type(c.get(k)) is int and c[k] >= 0 for k in keys)) for c in calls))
            known = {k: sum((c[k] for c in calls if type(c.get(k)) is int and c[k] >= 0)) for k in keys}
            failure = backend.state.get('decision_failure')
            execution = 'unexpected_exception' if failures else 'provider_failed' if failure == 'ProviderFailure' else 'decision_rejected' if failure else 'execution_stopped' if backend.stop.startswith('execution_stopped') else 'completed'
            method = bool(result['status'] == 'success' and result.get('independent_endpoint_checked') is True and (execution == 'completed'))
            corrected = any((r.get('decision_source') == 'corrected_llm' for r in trace))
            ruled = any((r.get('decision_source') == 'rule_fallback' for r in trace))
            any_llm = any((r.get('decision_source') in ('llm', 'corrected_llm') for r in trace))
            path = 'rule_fallback' if ruled else 'corrected_llm' if corrected else 'llm' if any_llm else 'measurement_shortcut'
            result.update(identity)
            result.update(method_version='interface_repair', repair_version='interface_repair', method_core_version='fallback', current_pass_shortcut_enabled=current_pass_shortcut_enabled, fallback_enabled=fallback_enabled, decision_count=len(trace), mock_decisions=client.is_mock, original_request_count=sum((a['request_kind'] == 'original' for r in trace for a in r.get('attempts', []) if a.get('raw') is not None or a.get('API_cost_rows'))), repair_attempt_count=sum((a['request_kind'].startswith('repair_attempt') for r in trace for a in r.get('attempts', []) if a.get('raw') is not None or a.get('API_cost_rows'))), rule_fallback_action_count=sum((r.get('decision_source') == 'rule_fallback' for r in trace)), recovery_episode_count=sum((bool(r.get('recovery_trigger')) for r in trace)), invalid_decision_count=sum((a.get('status') == 'decision_rejected' for r in trace for a in r.get('attempts', []))), early_stop_path='initial_full_measurement_pass' if backend.state.get('initial_acceptance_short_circuit') else 'current_full_measurement_pass' if backend.state.get('current_acceptance_short_circuit') else 'validated_finish' if trace and trace[-1]['status'] == 'finish_requested' else 'none', execution_status=execution, decision_failure_code=backend.state.get('decision_failure_code'), physical_endpoint_status=result['status'], method_success=method, completion_path=path, rule_fallback_success=method and ruled, corrected_llm_success=method and corrected and (not ruled), pure_llm_success=method and any_llm and (not corrected) and (not ruled), measurement_shortcut_success=method and (not trace), initial_acceptance_short_circuit=backend.state.get('initial_acceptance_short_circuit', False), current_acceptance_short_circuit=backend.state.get('current_acceptance_short_circuit', False), llm_loop_validated_with_real_provider=False, real_provider_exercised=bool(getattr(client, 'is_live_transport', False) and (not client.is_mock) and any((c.get('status') == 'completed' for c in calls))), endpoint_audit={'budget_before': endpoint_before, 'budget_after': endpoint_after, 'status': result['status'], 'independently_checked': result.get('independent_endpoint_checked')}, token_accounting={'known_subtotals': known, 'usage_complete': complete, 'exact_total_tokens': known['total_tokens'] if complete else None, 'subtotal_is_lower_bound': not complete, 'billing_status': 'unknown' if calls else 'no_requests', 'supplemental_details': 'api_NNN/usage_audit.json; missing reasoning remains null'})
        except Exception as exc:
            note_failure(exc, 'terminal_result_assembly')
            result = {'status': 'endpoint_unconfirmed', 'independent_endpoint_checked': False, 'physical_endpoint_status': 'endpoint_unconfirmed', 'method_version': 'interface_repair', 'method_core_version': 'fallback'}
        if failures:
            result.update(execution_status='unexpected_exception', method_success=False, rule_fallback_success=False, corrected_llm_success=False, pure_llm_success=False, measurement_shortcut_success=False, reason=backend.stop, execution_failure=failures[0], execution_failures=failures)
        try:
            save(backend.out / 'result.json', result)
        except Exception as exc:
            note_failure(exc, 'result_persistence')
            result.update(execution_status='unexpected_exception', method_success=False, rule_fallback_success=False, corrected_llm_success=False, pure_llm_success=False, measurement_shortcut_success=False, reason=backend.stop, execution_failure=failures[0], execution_failures=failures, terminal_result_persisted=False)
            try:
                save(backend.out / 'exception_audit.json', result)
            except Exception:
                result['exception_audit_persisted'] = False
    return result
