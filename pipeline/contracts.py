import hashlib
import json
import math
from dataclasses import dataclass

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def typed(arguments, schema):
    if type(arguments) is not dict or set(arguments) != set(schema):
        raise ValueError('exact argument keys required')
    for name, kind in schema.items():
        value = arguments[name]
        good = type(value) in (int, float) and math.isfinite(value) if kind == 'number' else type(value) is {'string': str, 'array': list, 'boolean': bool, 'object': dict}[kind]
        if not good:
            raise ValueError('argument type: ' + name)
    canonical(arguments)

def validate_decision(raw, state):
    value = json.loads(raw) if type(raw) is str else raw
    if type(value) is not dict or set(value) != {'schema_version', 'state_id', 'diagnosis', 'action', 'knowledge_ids'}:
        raise ValueError('decision schema mismatch')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['state_id'] != state['state_id']:
        raise ValueError('stale or wrong decision state')
    diagnosis = value['diagnosis']
    if type(diagnosis) is not dict or set(diagnosis) != {'category', 'summary', 'evidence_ids'}:
        raise ValueError('diagnosis schema')
    if diagnosis['category'] not in {'model_mismatch', 'realization_gap', 'planning_gap', 'constraints', 'success', 'unknown'} or type(diagnosis['summary']) is not str or (not 1 <= len(diagnosis['summary']) <= 1600):
        raise ValueError('diagnosis value')
    for values, allowed in [(diagnosis['evidence_ids'], state['evidence_ids']), (value['knowledge_ids'], state['knowledge_ids'])]:
        if type(values) is not list or not values or (not all((type(x) is str for x in values))) or (len(set(values)) != len(values)) or (not set(values) <= set(allowed)):
            raise ValueError('unbound decision evidence')
    action = value['action']
    if type(action) is not dict or set(action) != {'name', 'args'} or action['name'] not in state['allowed_actions']:
        raise ValueError('illegal action')
    typed(action['args'], state['tool_interface'][action['name']]['schema'])
    return json.loads(canonical(value))

@dataclass(frozen=True)
class Limits:
    decisions: int = 32
    model_calls: int = 32

    def __post_init__(self):
        if any((type(v) is not int or v < 1 for v in (self.decisions, self.model_calls))):
            raise ValueError('positive bounded decision/model limits')
