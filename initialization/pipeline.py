"""Explicit topology routing, tool applicability and honest diagnostic envelope.

Routing is metadata validation, not natural-language topology inference. This
module adds no optimizer and never turns unavailable residuals into zeros.
"""
import math
from .synthesis import InitializationError, digest
DECLARATIONS = {'t1_sk_lp': ('lowpass', 2, 'natural_frequency', 'native'), 't2_sk_hp': ('highpass', 2, 'natural_frequency', 'native')}
TOOLS = {'native': {'probe': 'native.bounded_coordinate_probe', 'calibrate': 'native.independent_ota_calibrate', 'rc_step': 'native.bounded_passive_allocation', 'coupled_step': 'native.bounded_joint_allocation', 'finish': 'native.fresh_endpoint'}}

def declaration_for(spec):
    if spec.get('topology') not in DECLARATIONS:
        raise InitializationError('unsupported declared topology; no guessed fallback')
    response, order, frequency_definition, family = DECLARATIONS[spec['topology']]
    return dict(topology=spec['topology'], response=response, order=order, frequency_definition=frequency_definition, physical_family=family)

def _unbound(*args, **kwargs):
    raise RuntimeError('execution handler has not been bound; metadata is not execution permission')

def route(spec, declaration, *, evidence, available_models, host=None):
    """Reuse the audited router.v1 implementation, with actual host bindings.

    Caller supplies independent trusted structure evidence and available model
    capabilities. No LLM proposal can fill either field. A metadata-only route
    exposes zero executable schemas until its physical host is bound.
    """
    from router import Package
    from router import Router
    from router import VERSION
    expected = declaration_for(spec)
    if declaration != expected:
        raise InitializationError('task/declaration conflict: response/order/frequency semantics')
    family = expected['physical_family']
    key = dict(family='active_rc', topology=spec['topology'], response=expected['response'], order=expected['order'])
    devices = frozenset(('R', 'C', 'OPAMP'))
    model = 'sky130_tt_two_stage_ota_v1'
    output = {'stage_verified': 'boolean', 'receipt_sha256': 'string', 'physical_acceptance': 'boolean'}
    schemas = {'initialize': {'task_id': 'string'}, 'measure': {'task_id': 'string', 'endpoint': 'boolean'}, 'repair': {'task_id': 'string', 'tool': 'string', 'arguments_json': 'string'}}
    handlers = {a: getattr(host, a) if host is not None else _unbound for a in schemas}
    package = Package(package_id='initialization-v2.' + spec['topology'], required_devices=devices, required_models=frozenset((model,)), dependencies=frozenset(('bound_runtime', 'shared_budget')), schemas=schemas, output_schema=output, handlers=handlers, verifier=host.verify_stage if host is not None else _unbound, allowed_actions=tuple(schemas) if host is not None else (), task_validator=lambda task: task.get('specification') == spec, argument_validator=lambda action, args, task: args['task_id'] == spec['task_id'], **key)
    router = Router([package])
    task = dict(schema_version=VERSION, task_id=spec['task_id'], declaration=key, evidence=evidence, available_models=available_models, specification=spec)
    routed = router.route(task)
    if routed['status'] != 'supported':
        raise InitializationError('router.v1 rejected: ' + routed['reason'])
    return dict(schema='kira.route.v2', recognition='exact_declared_topology', router_contract=VERSION, declaration=expected, tool_catalog=dict(TOOLS[family]), LLM_route=False, unsupported_action_fallback=False, route_sha256=digest(expected), runtime_bound=host is not None, exposed_schemas=router.expose(task), router_result=routed, _router=router, _task=task)

def select_tool(routing, action, current_allowed):
    catalog = routing['tool_catalog']
    if action not in catalog or action not in current_allowed:
        raise InitializationError('tool is not applicable and currently legal: ' + str(action))
    return catalog[action]

def _part(status, scope, value=None, evidence=()):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise InitializationError('finite diagnostic scalar required')
    if status in ('unavailable', 'not_applicable') and value is not None:
        raise InitializationError('missing diagnostic cannot be filled with a scalar')
    return dict(status=status, value=value, scope=scope, evidence_ids=list(evidence))

def diagnostic_envelope(family, raw, *, candidate_id, evidence_ids):
    """Normalize semantic status; retain raw family diagnostics separately.

    For extensions, measured constraint errors are not invented causal residuals.
    No common scalar sum or complete three-term decomposition is claimed there.
    """
    if family not in TOOLS:
        raise InitializationError('unsupported diagnostic family')
    if not candidate_id or not evidence_ids:
        raise InitializationError('diagnostic provenance required')
    base = dict(schema='kira.diagnostics.v2', family=family, candidate_id=candidate_id, raw_diagnostic_present=raw is not None, raw_diagnostic=raw, performance_gain_established=False, scope='local nominal measured operating point')
    if family == 'native' and raw is not None:
        components = {k: _part('derived_from_measured_and_modelled_responses', scope, raw.get(key), evidence_ids) for k, key, scope in [('model', 'model_error', 'H-Ha'), ('realization', 'realization_error', 'Ha-Hp'), ('planning', 'planning_error', 'Hp-Ht')]}
        if any((p['value'] is None for p in components.values())):
            raise InitializationError('native three-term diagnostic incomplete')
        base.update(components=components, complex_response_identity=True, scalar_scores_additive=False)
    else:
        base.update(complex_response_identity=False, scalar_scores_additive=False, components=dict(model=_part('available_in_family_report' if raw is not None else 'unavailable', 'family-specific prediction/measurement comparison; inspect raw report', evidence=evidence_ids), realization=_part('available_in_family_report' if raw is not None else 'unavailable', 'loaded section response errors, not separate transistor causal residual', evidence=evidence_ids), planning=_part('not_applicable', 'fixed prototype; no separately allocated response target', evidence=evidence_ids)))
    base['diagnostic_sha256'] = digest(base)
    return base
