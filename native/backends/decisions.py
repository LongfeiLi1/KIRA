"""Fail-closed decision contract. The model never supplies code or circuit values."""
import json
ACTIONS = {'rc_step', 'coupled_step', 'calibrate', 'probe', 'finish'}
DIAGNOSES = {'model_mismatch', 'realization_gap', 'planning_gap', 'constraints', 'success', 'unknown'}
AXES = ('r1', 'r2', 'cg', 'cf', 'current_scale', 'output_scale', 'cc_pf')

class DecisionError(ValueError):
    pass

def exact_keys(value, keys, error_type=DecisionError):
    DecisionError = error_type
    if not isinstance(value, dict) or set(value) != set(keys):
        raise DecisionError('Unexpected or missing schema fields')

def validate_decision(raw, state, *, state_axes=False, error_type=DecisionError):
    DecisionError = error_type
    if not isinstance(raw, str) or len(raw.encode()) > 32768:
        raise DecisionError('Decision must be bounded JSON text')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DecisionError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(DecisionError('Nonfinite JSON')))
    except (ValueError, TypeError, RecursionError) as exc:
        raise DecisionError('Invalid JSON decision') from exc
    exact_keys(value, ('schema_version', 'state_id', 'diagnosis', 'action', 'knowledge_ids'), error_type)
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise DecisionError('Unsupported schema')
    if value['state_id'] != state['state_id']:
        raise DecisionError('Stale evidence')
    diagnosis = value['diagnosis']
    exact_keys(diagnosis, ('category', 'summary', 'evidence_ids'), error_type)
    if not isinstance(diagnosis['category'], str) or diagnosis['category'] not in DIAGNOSES:
        raise DecisionError('Unknown diagnosis')
    if not isinstance(diagnosis['summary'], str) or not 1 <= len(diagnosis['summary']) <= 1600:
        raise DecisionError('Bounded diagnosis required')
    evidence = diagnosis['evidence_ids']
    if not isinstance(evidence, list) or not evidence or any((not isinstance(x, str) or x not in state['evidence_ids'] for x in evidence)):
        raise DecisionError('Unverified diagnosis evidence')
    ids = value['knowledge_ids']
    if not isinstance(ids, list) or not ids or any((not isinstance(x, str) or x not in state['knowledge_ids'] for x in ids)) or (len(ids) != len(set(ids))):
        raise DecisionError('Unretrieved knowledge card')
    action = value['action']
    exact_keys(action, ('name', 'args'), error_type)
    name, args = (action['name'], action['args'])
    if not isinstance(name, str) or name not in ACTIONS or name not in state['allowed_actions']:
        raise DecisionError('Action is not currently authorized by evidence')
    if name in {'rc_step', 'coupled_step'}:
        exact_keys(args, ('radius',), error_type)
        if not isinstance(args['radius'], str) or args['radius'] not in {'small', 'normal'}:
            raise DecisionError('Unreviewed trust radius')
    elif name == 'probe':
        exact_keys(args, ('axis', 'direction'), error_type)
        if not isinstance(args['axis'], str) or args['axis'] not in (state['parameter_names'] if state_axes else AXES) or type(args['direction']) is not int or (args['direction'] not in {-1, 1}):
            raise DecisionError('Invalid bounded probe')
    else:
        exact_keys(args, (), error_type)
    return value
