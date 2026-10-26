"""One explicit decision per tool dispatch. No model-to-code, hidden run() chain or fallback."""
import copy
import time

def run_loop(task, router, session_factory, client, *, arm, limits, digest, decode, sink=None, controller=None, stop_output=True, decision_details=None, tool_details=None):
    trace = []
    started = time.monotonic_ns()
    session = None
    error = None
    endpoint = None
    route = router.route(task)

    def record(kind, **data):
        row = dict(sequence=len(trace), kind=kind, monotonic_ns=time.monotonic_ns(), **copy.deepcopy(data))
        trace.append(row)
        if sink:
            sink(row)
    record('routing', route=route)
    if route['status'] != 'supported':
        return dict(schema_version='kira.full_result.v1', status='unsupported', method_success=False, complete_agent_pass=False, model_calls=0, decision_count=0, mock_decisions=bool(getattr(client, 'is_mock', True)), real_provider_exercised=False, LLM_incremental_evidence=False, trace=trace)
    try:
        session = session_factory(task, route)
        session.configure_arm(arm)
        record('initialization', output=session.initialize())
        for step in range(limits.decisions):
            state = session.snapshot(step)
            state['case'] = copy.deepcopy(task)
            state['run_id'] = session.run_id
            state['state_id'] = digest(dict(run_id=session.run_id, task=task, step=step, state=state))
            record('state', state=state)
            if controller is None:
                if client.calls >= limits.model_calls:
                    raise RuntimeError('model call allowance exhausted')
                receipts_before = len(getattr(client, 'receipts', []))
                raw = client.decide(copy.deepcopy(state))
                source = 'mock' if client.is_mock else 'llm'
                if not client.is_mock:
                    receipts = getattr(client, 'receipts', [])
                    if len(receipts) != receipts_before + 1 or receipts[-1].get('settlement') != 'settled' or receipts[-1].get('actual_attempts') != 1:
                        raise RuntimeError('one newly settled public provider receipt per LLM decision required')
            else:
                raw = controller.choose(copy.deepcopy(state), session)
                source = 'deterministic_controller'
            record('response', source=source, raw=raw)
            decision, details = decode(raw, state)
            if decision_details:
                details.update(decision_details(state, decision, source, client, controller, limits))
            session.validate(decision, state)
            record('decision', source=source, decision=decision, controller_audit=getattr(controller, 'last_record', None), accounting=copy.deepcopy(getattr(client, 'receipts', [])[-1:]), **details)
            action = decision['action']
            if action['name'] == 'stop':
                extra = {'output': session.stop()} if stop_output else {}
                record('stopped', reason='explicit decision', **extra)
                break
            if action['name'] == 'finish':
                endpoint = session.finish(action['args'])
                record('independent_endpoint', output=endpoint, **tool_details(state, action, endpoint) if tool_details else {})
                break
            before = session.capture() if hasattr(session, 'capture') else None
            try:
                output = session.execute(action['name'], action['args'])
            except Exception as exc:
                if session.is_control_exception(exc) or not hasattr(session, 'recover_tool_error') or (not session.recover_tool_error(exc)):
                    raise
                output = dict(status='tool_failed', error_class=type(exc).__name__, recovery='next explicit decision; no automatic fallback')
            after = session.capture() if hasattr(session, 'capture') else None
            record('feedback', action=action, output=output, before=before, after=after, rollback=output.get('rollback') if isinstance(output, dict) else None, **tool_details(state, action, output, before=before, after=after) if tool_details else {})
        else:
            raise RuntimeError('decision allowance exhausted; no unrequested endpoint launch')
        if endpoint is not None:
            if session.verify_receipts(endpoint) is not True:
                raise ValueError('current endpoint receipts rejected')
            record('receipts_verified')
            if session.verify_endpoint(endpoint, task) is not True:
                raise ValueError('independent family verifier rejected')
            record('family_verified')
    except Exception as exc:
        error = type(exc).__name__
        record('failure', error_class=error)
        if session is not None and session.is_control_exception(exc):
            raise
    finally:
        if session is not None:
            session.persist_trace(trace)
    decisions = [r for r in trace if r['kind'] == 'decision']
    mock = bool(getattr(client, 'is_mock', True))
    passed = endpoint is not None and error is None and any((r['kind'] == 'family_verified' for r in trace))
    return dict(schema_version='kira.full_result.v1', status='completed' if error is None else 'failed', error_class=error, arm=arm, endpoint=endpoint, method_success=passed, complete_agent_pass=passed and (not mock) and (not getattr(session, 'is_mock', True)) and (controller is None), mock_decisions=mock, real_provider_exercised=any((r['source'] == 'llm' for r in decisions)), LLM_incremental_evidence=False, decision_count=len(decisions), model_calls=client.calls, provider_accounting=copy.deepcopy(getattr(client, 'all_attempts', getattr(client, 'receipts', []))), model_call_count_scope='decision invocations; accounting records retain failed and unknown dispatches', endpoint_receipt_proof=copy.deepcopy(getattr(getattr(session, 'endpoint_proof', None), 'receipt_proof', None)), administrative_or_mock_success_is_physical=False, trace=trace, wall_ns=time.monotonic_ns() - started)
