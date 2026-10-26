"""Topology-specific decision validation and error classification."""
REASONS = {'Unexpected or missing schema fields': 'schema_fields', 'Decision must be bounded JSON text': 'decision_text_size_or_type', 'Duplicate JSON key': 'duplicate_json_key', 'Invalid JSON decision': 'invalid_json', 'Unsupported schema': 'schema_version', 'Stale evidence': 'state_id_mismatch', 'Unknown diagnosis': 'diagnosis_category', 'Bounded diagnosis required': 'diagnosis_summary', 'Unverified diagnosis evidence': 'unavailable_evidence_reference', 'Unretrieved knowledge card': 'knowledge_reference', 'Action is not currently authorized by evidence': 'action_not_currently_allowed', 'Unreviewed trust radius': 'invalid_trust_radius', 'Invalid bounded probe': 'invalid_probe'}

def contract(type_id):
    if type_id == 'T1':
        import backends.lowpass_contracts as contracts
    elif type_id in ('T2',):
        import backends.highpass_contracts as contracts
    else:
        raise ValueError('Unsupported frozen method')
    return contracts

def original_payload(state, protocol):
    module = contract(state['case']['type_id'])
    return {'model': 'qwen3.8-flash', 'messages': [{'role': 'system', 'content': module.SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps(state, allow_nan=False)}], 'temperature': 0, 'max_tokens': protocol['max_completion_tokens_per_call'], 'stream': False, 'response_format': {'type': 'json_object'}, 'enable_thinking': True}

def validate_and_audit(raw, state):
    module = contract(state['case']['type_id'])
    try:
        decision = module.validate_decision(raw, state)
        return {'status': 'accepted', 'failure_type': None, 'decision': decision, 'action_executed': False}
    except module.DecisionError as exc:
        reason = REASONS.get(str(exc), 'unclassified_contract_rejection')
        cause = REASONS.get(str(exc.__cause__)) if exc.__cause__ else None
        return {'status': 'rejected', 'failure_type': 'DecisionError', 'schema_error': reason, 'nested_schema_error': cause, 'decision': None, 'action_executed': False, 'action_repaired_or_guessed': False}
