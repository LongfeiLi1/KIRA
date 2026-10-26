"""Offline, fail-closed capability dispatch. No provider or simulator imports."""
from dataclasses import dataclass, field
from typing import Protocol
import json
import copy
import math
import time
VERSION = 'kira.router.v1'
FIELDS = ('family', 'topology', 'response', 'order')
DEVICES = {'active_rc': {'R', 'C', 'OPAMP'}}

class AdvisoryClassifier(Protocol):

    def propose(self, task: dict) -> dict:
        """Optional untrusted proposal; never used to authorize dispatch."""

def validate(value, schema):
    if not isinstance(value, dict) or set(value) != set(schema):
        raise ValueError('schema keys mismatch')
    types = {'string': str, 'boolean': bool, 'array': list}
    for key, kind in schema.items():
        v = value[key]
        if kind == 'number':
            valid = type(v) in (int, float) and math.isfinite(v)
        else:
            valid = kind in types and type(v) is types[kind]
        if not valid:
            raise ValueError('invalid field: ' + key)

@dataclass(frozen=True)
class Package:
    package_id: str
    family: str
    topology: str
    response: str
    order: int
    required_devices: frozenset
    required_models: frozenset
    dependencies: frozenset
    schemas: dict
    output_schema: dict
    handlers: dict
    verifier: object
    input_version: str = '1.0.0'
    output_version: str = '1.0.0'
    allowed_actions: object = None
    task_validator: object = None
    action_dependencies: dict = field(default_factory=dict)
    result_adapter: object = None
    argument_validator: object = None

@dataclass
class Context:
    remaining_actions: int
    dependencies: frozenset = frozenset()
    events: list = field(default_factory=list)
    status: str = 'ready'
    active_task: object = None

class Router:

    def __init__(self, packages):
        self.packages = {}
        signatures = set()
        for original in packages:
            p = copy.deepcopy(original)
            signature = tuple((getattr(p, k) for k in FIELDS))
            if p.package_id in self.packages or signature in signatures or p.family not in DEVICES or (type(p.order) is not int) or (p.order < 1) or (not DEVICES[p.family] <= p.required_devices) or (not callable(p.verifier)) or (not set(p.schemas)) or (not set(p.schemas) <= {'initialize', 'measure', 'repair'}) or (set(p.handlers) != set(p.schemas)) or (not all((callable(h) for h in p.handlers.values()))) or (p.input_version != '1.0.0') or (p.output_version != '1.0.0'):
                raise ValueError('invalid or ambiguous registration')
            if p.allowed_actions is not None and (not set(p.allowed_actions) <= set(p.schemas)) or (p.task_validator is not None and (not callable(p.task_validator))) or (not set(p.action_dependencies) <= set(p.schemas)) or (p.result_adapter is not None and (not callable(p.result_adapter))) or (p.argument_validator is not None and (not callable(p.argument_validator))):
                raise ValueError('invalid host policy')
            for schema in [*p.schemas.values(), p.output_schema]:
                if not isinstance(schema, dict) or any((t not in {'number', 'string', 'boolean', 'array'} for t in schema.values())):
                    raise ValueError('unsupported schema')
            signatures.add(signature)
            self.packages[p.package_id] = p

    def route(self, task):
        started = time.perf_counter_ns()
        result = {'status': 'needs_confirmation', 'reason': 'missing or invalid evidence', 'package_id': None, 'source': None}
        if isinstance(task, dict) and task.get('schema_version') == VERSION:
            d, e = (task.get('declaration'), task.get('evidence'))
            if isinstance(d, dict) and isinstance(e, dict):
                source = e.get('provenance')
                result['source'] = source
                if type(source) is str and source in {'structured_topology', 'benchmark_metadata'} and all((k in d and k in e and (d[k] is not None) and (e[k] is not None) for k in FIELDS)) and isinstance(e.get('devices'), list) and all((type(x) is str for x in e['devices'])) and isinstance(task.get('available_models'), list) and all((type(x) is str for x in task['available_models'])) and (type(d['order']) is int) and (type(e['order']) is int) and (d['order'] > 0) and (e['order'] > 0) and all((type(d[k]) is str and type(e[k]) is str for k in FIELDS[:3])):
                    if any((d[k] != e[k] for k in FIELDS)):
                        result['reason'] = 'declaration/evidence conflict'
                    else:
                        result.update(status='unsupported', reason='unregistered capability')
                        for p in self.packages.values():
                            if all((d[k] == getattr(p, k) for k in FIELDS)):
                                devices = set(e['devices'])
                                if devices != set(p.required_devices):
                                    result['reason'] = 'device capability mismatch'
                                elif not p.required_models <= set(task['available_models']):
                                    result['reason'] = 'missing model capability'
                                elif p.task_validator is not None and (not self._trusted(p, task)):
                                    result['reason'] = 'unrecognized trusted task structure'
                                else:
                                    result.update(status='supported', reason='exact verified metadata match', package_id=p.package_id)
                                break
        result['route_ns'] = time.perf_counter_ns() - started
        return result

    @staticmethod
    def _trusted(p, task):
        try:
            return p.task_validator(copy.deepcopy(task)) is True
        except Exception:
            return False

    @staticmethod
    def _allowed(p, action):
        return action in p.schemas and (p.allowed_actions is None or action in p.allowed_actions)

    def expose(self, task):
        r = self.route(task)
        if r['status'] != 'supported':
            return []
        p = self.packages[r['package_id']]
        return [{'package_id': p.package_id, 'action': a, 'input_version': p.input_version, 'output_version': p.output_version, 'schema': dict(s), 'output_schema': dict(p.output_schema), 'description': 'Execute registered ' + a + '; physical verification required.'} for a, s in p.schemas.items() if self._allowed(p, a)]

    def execute(self, task, package_id, action, args, context, input_version='1.0.0'):
        r = self.route(task)
        outcome = {'status': 'rejected', 'reason': 'illegal dispatch'}
        if r['status'] == 'supported' and package_id == r['package_id'] and (context.status in {'ready', 'running'}):
            p = self.packages[package_id]
            if self._allowed(p, action) and input_version == p.input_version:
                try:
                    validate(args, p.schemas[action])
                    if p.argument_validator is not None and p.argument_validator(action, copy.deepcopy(args), copy.deepcopy(task)) is not True:
                        raise ValueError('arguments conflict with trusted task')
                    if not p.action_dependencies.get(action, p.dependencies) <= context.dependencies:
                        raise ValueError('missing dependency')
                    if type(context.remaining_actions) is not int or context.remaining_actions <= 0:
                        context.status = 'budget_exhausted'
                        raise ValueError('budget exhausted')
                    context.remaining_actions -= 1
                    context.status = 'running'
                    bound_task = copy.deepcopy(task)
                    context.active_task = copy.deepcopy(bound_task)
                    output = p.handlers[action](copy.deepcopy(args), context)
                    validate(output, p.output_schema)
                    if p.verifier(output, bound_task) is not True:
                        raise ValueError('physical verification failed')
                    outcome = {'status': 'verified', 'output': output, 'output_version': p.output_version}
                    if p.result_adapter is not None:
                        outcome['result'] = p.result_adapter(action, output)
                except Exception as exc:
                    if context.status != 'budget_exhausted':
                        context.status = 'failed'
                    outcome = {'status': 'failed', 'reason': type(exc).__name__}
        context.events.append({'event': 'dispatch', 'task_id': task.get('task_id') if isinstance(task, dict) else None, 'package_id': package_id, 'action': action, 'status': outcome['status'], 'remaining_actions': context.remaining_actions})
        return outcome

    def terminate(self, context, reason='completed'):
        if reason not in {'completed', 'cancelled'}:
            raise ValueError('invalid termination reason')
        if context.status in {'ready', 'running'}:
            context.status = reason
            context.events.append({'event': 'terminate', 'status': reason})
        return context.status

    def telemetry(self, task, expected_package, token_counter=None):
        start = time.perf_counter_ns()
        r = self.route(task)
        payload = json.dumps(self.expose(task), sort_keys=True, separators=(',', ':'))
        selected = r['package_id']
        return {**r, 'expected_package': expected_package, 'correct': selected == expected_package, 'wrong_dispatch': selected is not None and selected != expected_package, 'prompt_bytes': len(payload.encode('utf-8')), 'prompt_tokens': token_counter(payload) if token_counter else None, 'end_to_end_router_ns': time.perf_counter_ns() - start}
