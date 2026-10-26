"""Instance-owned refinement policy. All physical observations stay in CommonExecution."""
import copy
import math
from dataclasses import asdict
from types import MethodType
from control.policy import BASE_SHA
from control.policy import digest
from control.policy import sha
HIDDEN = frozenset(('model_error', 'realization_error', 'planning_error', 'trust_band_component_errors', 'model_jacobian', 'predicted_constraints_with_margin', 'linearized_hard_constraints', 'required_route_margin', 'rc_only_assessment', 'repair_guidance', 'proposal', 'planned_tau', 'target', 'ota_target', 'OTA_target', 'history', 'target_realized_within_tolerances'))

def project(value):
    if isinstance(value, dict):
        result = {k: project(v) for k, v in value.items() if k not in HIDDEN}
        if 'model_trusted' in value and ('model_id' in value or 'model_error' in value):
            allowed = ('candidate_id', 'type_id', 'model_trusted', 'matched_operating_point', 'qualification_blocker', 'dc_match_tolerance_v', 'dc_output_difference_v', 'nominal_dc_conditions_equal', 'acceptance')
            result = {k: copy.deepcopy(value[k]) for k in allowed if k in value}
        if 'target_realized_within_tolerances' in value:
            result = {k: copy.deepcopy(value[k]) for k in ('system_acceptance', 'optimizer', 'max_objective_calls') if k in value}
        return result
    if isinstance(value, (list, tuple)):
        return [project(v) for v in value]
    return copy.deepcopy(value)

def valid(row):
    return bool(row and row.get('evidence') and (row.get('measurement_status', 'valid') == 'valid'))

def context(backend):
    from rc_agent import physical
    from control.policy import model_binding
    from control.policy import read
    binding=model_binding()
    dependencies=binding['files']
    return {'case': backend.case, 'topology_binding': getattr(backend, 'binding', None), 'task': getattr(backend, 'task', None), 'bounds': backend.bounds, 'acceptance': backend.common, 'protocol': backend.engine.protocol, 'optimizer': backend.optimizer_settings, 'base_source': BASE_SHA, 'PDK_dependencies': dependencies, 'PDK_path': str(physical.PDK), 'execution_seed': 0, 'simulator': binding['simulator'], 'conditions': 'TT; VDD1.8; VinVref1.2; unchanged source/load and analysis decks'}

def key(backend):
    row = backend.current
    return digest({'context': backend._refinement_context, 'observation': row, 'descriptor_revision': len(backend.auxiliary), 'descriptor_anchor': backend.descriptor_anchor, 'current_paid_descriptor': None if not row or descriptor_raw_key(backend, row['parameters']) not in backend.descriptors else asdict(backend.descriptors[descriptor_raw_key(backend, row['parameters'])][0]), 'plan_eta': None if backend.plan_eta is None else asdict(backend.plan_eta), 'model_instance_revision': id(backend.plan_calibration), 'calibration_revision': backend.calibrations, 'fixed_targets': getattr(backend, 'fixed_allocation_targets', None), 'adaptive_target_feedback': {k: getattr(backend, 'adaptive_target_feedback', {}).get(k) for k in ('revision', 'eta_penalty_weights')}})

def model_from_bench(backend, x, eta, evidence):
    if backend.case['type_id'] != 'T1':
        return backend.calibration_models[backend.descriptor_key(x)]
    from rc_agent import physical
    return physical.calibration_from_evidence(eta, evidence)

def descriptor_raw_key(backend, x):
    if backend.case['type_id'] != 'T1':
        return backend.descriptor_key(x)
    from backends.lowpass import value_hash
    from backends.lowpass import DEVICE_NAMES
    return value_hash({n: float(x[n]) for n in DEVICE_NAMES})

def descriptor_deck(backend, x):
    if backend.case['type_id'] == 'T1':
        from execution.device_model.characterization import deck
        return deck(backend.devices(x))
    from backends.highpass_characterization import deck
    return deck(backend.devices(x), backend.case, x)

def descriptor_render_key(backend, x):
    import hashlib
    return digest({'context': backend._refinement_context, 'exact_rendered_deck_sha256': hashlib.sha256(descriptor_deck(backend, x).encode()).hexdigest()})

def install_rendered_descriptor_cache(backend):
    native = backend.characterize
    cache = {}
    backend._rendered_descriptor_cache = cache

    def characterize(self, x, *, fresh=False):
        rendered = descriptor_render_key(self, x)
        raw = descriptor_raw_key(self, x)
        if not fresh and rendered in cache:
            result, model = cache[rendered]
            self.descriptors[raw] = result
            if self.case['type_id'] != 'T1':
                self.calibration_models[raw] = model
            self.refinement_audit['rendered_OTA_cache_hits'] += 1
            self.event('exact_rendered_OTA_cache_reuse', {'render_key': rendered, 'source_evidence': result[1]['evidence'], 'new_physical_candidate': False, 'new_SPICE': 0, 'fresh': False, 'no_float_rounding': True})
            return result
        result = native(x, fresh=fresh)
        model = None if self.case['type_id'] == 'T1' else self.calibration_models[raw]
        cache[rendered] = (result, model)
        return result
    backend.characterize = MethodType(characterize, backend)

def install_bootstrap(backend):
    from backends.lowpass import analytic_initial
    from backends.acceptance import initial_pass
    audit = {'enabled': True, 'triggered': False, 'skipped_OTA_requests': 0, 'endpoint_still_requires_fresh_measurement': True, 'planning_bootstrap_not_available_when_skipped': False, 'reason': None}
    backend.bootstrap_optimization_audit = audit

    def initialize(self):
        typed = self.case['type_id'] != 'T1'
        if not self.room(3, 13 if typed else 12, 2):
            from backends.lowpass import ToolUnavailable
            raise ToolUnavailable('initial_shared_allowance_unavailable')
        x = dict(self.binding['initial']) if typed else analytic_initial(self.task, self.bounds)
        self.current = self.evaluate(x)
        self.event('unchanged_analytic_initial_full_measurement', self.current)
        if initial_pass(self):
            audit.update(triggered=True, skipped_OTA_requests=1, planning_bootstrap_not_available_when_skipped=True, reason='unchanged complete initial_PASS; fresh endpoint remains mandatory')
            self.event('planning_bootstrap_deferred_after_full_initial_PASS', audit)
        elif valid(self.current):
            bootstrap(self, self.current)
        return self.current
    backend.initialize = MethodType(initialize, backend)
    native_accept = backend.accept_trial

    def accept_trial(self, row, *, planned_eta=None, planned_model=None, force=False):
        from backends.acceptance import initial_pass
        from types import SimpleNamespace
        if self.engine.status != 'paused' and initial_pass(SimpleNamespace(current=row)):
            previous = self.current
            cached = descriptor_raw_key(self, row['parameters']) in self.descriptors or descriptor_render_key(self, row['parameters']) in self._rendered_descriptor_cache
            self.current = row
            self.plan_eta = planned_eta if planned_eta is not None else self.plan_eta
            self.plan_calibration = planned_model if planned_model is not None else self.plan_calibration
            self.last_diagnostic = None
            reason = 'complete valid full PASS and frozen native next-actions exactly finish; unused planning-only diagnosis deferred; fresh endpoint mandatory'
            self.refinement_audit['terminal_PASS_diagnostic_deferrals'] += 1
            self.event('complete_PASS_trial_diagnosis_deferred', {'candidate_id': row['candidate_id'], 'descriptor_request_skipped': True, 'paid_descriptor_already_cached': cached, 'actual_new_OTA': 0, 'skipped_reason': reason, 'previous_costs_not_modified': True, 'full_filter_evaluation_already_complete': True, 'fresh_endpoint_required': True})
            return {'selected': True, 'rollback': False, 'previous_candidate': previous['candidate_id'], 'trial': row, 'trial_diagnostics': None, 'trial_diagnostic_error': None, 'diagnostic_deferred': True, 'diagnostic_skipped_reason': reason, 'next_legal_actions': ['finish']}
        if valid(row) and planned_model is None:
            self._capture_bootstrap_target = self.plan_calibration is None
            self._bootstrap_capture_row = row
            try:
                eta, evidence = self.characterize(row['parameters'])
                planned_model = model_from_bench(self, row['parameters'], eta, evidence)
                planned_eta = planned_eta or eta
            finally:
                self._capture_bootstrap_target = False
                self._bootstrap_capture_row = None
        return native_accept(row, planned_eta=planned_eta, planned_model=planned_model, force=force)
    backend.accept_trial = MethodType(accept_trial, backend)

def bootstrap(backend, row):
    backend._capture_bootstrap_target = True
    backend._bootstrap_capture_row = row
    try:
        eta, evidence = backend.characterize(row['parameters'])
        backend.plan_eta = eta
        backend.plan_calibration = model_from_bench(backend, row['parameters'], eta, evidence)
    finally:
        backend._capture_bootstrap_target = False
        backend._bootstrap_capture_row = None
    backend.last_diagnostic = None
    backend.event('first_valid_independent_model_bootstrap', {'candidate_id': row['candidate_id'], 'independent_OTA_evidence': evidence['evidence'], 'full_filter_fit': False, 'extra_SPICE': 0, 'one_pole_diagnostic_before_known_A_Z_removed': True})
    return backend.diagnose()

def install_diagnosis_cache(backend):
    native = backend.diagnose
    cache = {}

    def diagnose(self):
        k = key(self)
        if k in cache:
            self.refinement_audit['diagnostic_math_cache_hits'] += 1
            self.last_diagnostic = copy.deepcopy(cache[k])
            return self.last_diagnostic
        value = native()
        cache[key(self)] = copy.deepcopy(value)
        self.refinement_audit['diagnostic_math_computations'] += 1
        return value
    backend.diagnose = MethodType(diagnose, backend)

def install_sizing_stop(backend):
    from provider.stream import injected
    from backends.lowpass import SizingBlockComplete
    native = backend.size_devices.__func__
    powell = native.__globals__['native_powell']

    def bounded_powell(objective, z0, *args, **kwargs):
        import numpy as np
        from control.action_preview import paid_power_seed
        raw = descriptor_raw_key(backend, backend.current['parameters'])
        preview = paid_power_seed(backend, backend._active_size_plan, raw, z0=z0)
        initial = np.asarray(preview['seed_log_parameters'], float)
        if preview['power_ratio_rule_available']:
            backend.refinement_audit['power_ratio_seeds'] += int(preview['power_ratio_rule_applied'])
            backend.event('generic_paid_power_ratio_device_seed', preview)

        def measured_objective(z):
            value = objective(z)
            import numpy as np
            cells = dict(zip(objective.__code__.co_freevars, objective.__closure__))
            best = cells['best'].cell_contents
            plan = backend._active_size_plan
            ratio = np.abs(np.log(np.array([best['eta'][k] for k in ('a0', 'wp_rad_s', 'ro_ohm')]) / np.array([plan['ota_target'][k] for k in ('a0', 'wp_rad_s', 'ro_ohm')])))
            ready = bool(np.all(ratio <= [0.25, 0.25, 0.35]) and best['power_w'] <= plan['power_target_w'] and (best['minimum_saturation_margin_v'] >= plan['saturation_target_v']) and (backend.common['dc_out_range_v'][0] <= best['dc_out'] <= backend.common['dc_out_range_v'][1]))
            if ready:
                backend.refinement_audit['device_target_early_stops'] += 1
                raise SizingBlockComplete('independently_measured_device_target_ready_full_filter_still_required')
            return value
        return powell(measured_objective, initial, *args, **kwargs)
    sized = injected(native, native_powell=bounded_powell).__get__(backend, type(backend))

    def size_devices(self, plan, max_objectives=12):
        self._active_size_plan = plan
        try:
            return sized(plan, max_objectives=max_objectives)
        finally:
            self._active_size_plan = None
    backend.size_devices = MethodType(size_devices, backend)

def measured_coordinate_plan(backend, coupled, radius):
    """No residual, model response, Jacobian, propose or bounded_update access.

    Global rules: f high => raise all passive values; f low => lower them.
    Q correction uses topology's known passive-ratio sign, within same .04/.08.
    OTA target stays measured; power allocation uses measured excess only.
    Missing f/Q produce unchanged coordinates and retain probe/finish choices.
    """
    from backends.lowpass import ToolUnavailable
    import numpy as np
    if radius not in ('small', 'normal'):
        raise ValueError('unchanged_radius_domain')
    diagnostic = backend.last_diagnostic or backend.diagnose()
    if not diagnostic['model_trusted']:
        raise ToolUnavailable('shared_independent_model_safety_gate')
    eta, _ = backend.characterize(backend.current['parameters'])
    if coupled:
        anchor = np.asarray(backend.descriptor_anchor, float)
        achieved = np.asarray(eta.values, float)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)) or np.any(anchor <= 0):
            raise ValueError('shared_first_characterization_anchor_invalid')
        if np.any(achieved < 0.5 * anchor) or np.any(achieved > 2.0 * anchor):
            raise ValueError('Measured ETA outside fixed first-characterization domain')
    typed = backend.case['type_id'] != 'T1'
    if typed:
        from backends.highpass_numerics import constraints
        from execution.highpass.adapter import PASSIVE_NAMES
        names = PASSIVE_NAMES[backend.case['type_id']]
        actual = constraints(backend.current['metrics'], backend.case, backend.common).tolist()
        qtarget = backend.case['q_target']
    else:
        from backends.lowpass_numerics import constraints
        from backends.lowpass_numerics import PASSIVE_NAMES
        names = PASSIVE_NAMES
        actual = constraints(backend.current['metrics'], backend.task, backend.common).tolist()
        qtarget = backend.task['q']
    delta = {'small': 0.04, 'normal': 0.08}[radius]
    metrics = backend.current['metrics']
    x = backend.current['parameters']
    target_f = backend.case['f0_hz']
    measured_f = metrics.get('f0_hz')
    shift = float(np.clip(0.5 * math.log(measured_f / target_f), -delta, delta)) if isinstance(measured_f, (int, float)) and measured_f > 0 else 0.0
    shifts = {n: shift for n in names}
    q = metrics.get('q')
    qshift = float(np.clip(math.log(qtarget / q), -delta, delta)) if isinstance(q, (int, float)) and q > 0 else 0.0
    if backend.case['type_id'] == 'T1':
        shifts['cf'] = float(np.clip(shift + qshift, -delta, delta))
        shifts['cg'] = float(np.clip(shift - qshift, -delta, delta))
    passives = {n: float(np.clip(x[n] * math.exp(shifts[n]), *backend.bounds[n])) for n in names}
    limit = backend.case['power_max_uW'] * 1e-06
    power = metrics['power_w']
    target_power = min(power, 0.95 * limit) if coupled else power
    return {'passives': passives, 'ota_target': asdict(eta), 'power_target_w': target_power, 'saturation_target_v': backend.common['minimum_saturation_margin_v'] + 0.005, 'actual_constraints': actual, 'rc_only_feasible': not coupled, 'predicted_constraints_with_margin': None, 'coordinate_plan_requires_actual_validation': True, 'predicted_only': True, 'repair_policy': 'predeclared_measured_coordinate_without_residual_or_Jacobian', 'coordinate_log_shifts': shifts, 'radius': radius, 'descriptor_domain_anchor': list(backend.descriptor_anchor), 'descriptor_domain_scale': [0.5, 2.0]}

def install_allocation(backend, no_diagnostic):
    native = backend.allocation
    cache = {}

    def allocation(self, coupled, radius='normal'):
        k = (key(self), bool(coupled), radius)
        if k in cache:
            self.refinement_audit['allocation_math_cache_hits'] += 1
            return copy.deepcopy(cache[k])
        result = measured_coordinate_plan(self, coupled, radius) if no_diagnostic else native(coupled, radius)
        cache[key(self), bool(coupled), radius] = copy.deepcopy(result)
        self.refinement_audit['allocation_math_computations'] += 1
        return result
    backend.allocation = MethodType(allocation, backend)

def guidance(backend, state):
    from backends.lowpass import ToolUnavailable
    diagnostic = state.get('diagnostics') or {}
    plans = []
    if diagnostic.get('model_trusted'):
        for name, coupled in (('rc_step', False), ('coupled_step', True)):
            if name not in state['allowed_actions']:
                continue
            for radius in ('small', 'normal'):
                try:
                    p = backend.allocation(coupled, radius)
                    from control.action_preview import compare_allocation
                    from control.action_preview import paid_power_seed
                    comparison = compare_allocation(p)
                    implementation = None if not coupled else paid_power_seed(backend, p, descriptor_raw_key(backend, backend.current['parameters']))
                    n = len(p['passives'])
                    jac = p['model_jacobian']
                    plans.append({'action': name, 'radius': radius, **comparison, 'RC_certified_margin': p['rc_only_feasible'] if not coupled else None, 'coupled_device_implementation_preview': implementation, 'coupled_legality_is_model_domain_budget_not_RC_feasibility': coupled, 'passive_column_support': [sum((abs(row[k]) for row in jac)) for k in range(n)], 'OTA_power_column_support': [sum((abs(row[k]) for row in jac)) for k in range(n, len(jac[0]))], 'candidate_passives': p['passives'], 'OTA_target': p['ota_target'], 'power_target_w': p['power_target_w'], 'local_delta_z': p['proposal']['delta_z'], 'unsupported_gates_require_real_SPICE': True})
                except (ValueError, RuntimeError, OSError, ToolUnavailable) as exc:
                    plans.append({'action': name, 'radius': radius, 'available': False, 'reason': type(exc).__name__})
    components = diagnostic.get('trust_band_component_errors')
    dominant = None if not components else ['model', 'realization', 'planning'][max(range(3), key=lambda i: components[i])]
    ranked = sorted((p for p in plans if 'predicted_reduction' in p), key=lambda p: (-p['predicted_reduction'], p['action'] != 'rc_step', p['radius'] != 'small'))
    return {'dominant_trust_band_residual': dominant, 'alternatives': plans, 'predicted_candidate_order': [{'action': p['action'], 'radius': p['radius']} for p in ranked], 'controller_instructions': 'Use measured residuals and alternative sensitivities to choose RC versus OTA and small versus normal. Ranking compares the same route margin before/after. Coupled linear allocation and the device realization goal/first seed are separate stages: an allocated power goal and proportional seed estimate are not a measured OTA or complete circuit prediction. RC certification applies only to rc_step. The seed may fail and the native device block may use up to 12 objective calls; no extra SPICE was run to preview it. Predictions are not acceptance. Prefer RC when its margin is certified. Characterize reuses only the exact already paid independent bench; calibrate means fresh independent OTA plus same-point full repeat and should address mismatch, not a missing initial model. A device target-ready stop still requires actual full circuit and fresh endpoint.', 'rejected_last_step': state.get('last_tool_result', {}).get('rollback') is True}

def install_snapshot(backend, no_diagnostic):
    native = backend.snapshot

    def snapshot(self, step):
        state = native(step)
        if no_diagnostic:
            state = project(state)
            state['diagnostic_policy_scope'] = 'Residual/Jacobian repair guidance absent; measured coordinate policy; model_trusted only safety gate'
        else:
            state['repair_guidance'] = guidance(self, state)
        from control.action_preview import measured_probe_preview
        state['shared_measured_probe_preview'] = measured_probe_preview(self)
        state.pop('state_id', None)
        state['state_id'] = digest(state)
        return state
    backend.snapshot = MethodType(snapshot, backend)

def install(backend, factors):
    from control.adaptive import install as install_adaptive
    install_adaptive(backend, enabled=not (factors['fixed_allocation_targets'] or factors['no_diagnostic_repair']))
    backend._refinement_context = context(backend)
    backend.refinement_audit = {'diagnostic_math_cache_hits': 0, 'diagnostic_math_computations': 0, 'allocation_math_cache_hits': 0, 'allocation_math_computations': 0, 'device_target_early_stops': 0, 'rendered_OTA_cache_hits': 0, 'power_ratio_seeds': 0, 'terminal_PASS_diagnostic_deferrals': 0, 'cache_scope': 'new worker instance; no historical/cross-case observations', 'physical_cache_changed': 'exact rendered OTA deck equivalence, fresh bypass', 'accounting_changed': False}
    install_rendered_descriptor_cache(backend)
    install_bootstrap(backend)
    install_diagnosis_cache(backend)
    install_sizing_stop(backend)
    if factors['fixed_allocation_targets']:
        from control.targets import install_targets
        install_targets(backend)
    install_allocation(backend, factors['no_diagnostic_repair'])
    install_snapshot(backend, factors['no_diagnostic_repair'])
    from control.scheduler import install as install_scheduler
    install_scheduler(backend)
    return backend
