"""Fixed equality targets: same mathematical tool, immutable first valid independent anchor."""
import copy
import math
from types import MethodType
from control.policy import read
from control.policy import digest

def fixed_update(update, target_getter, passive_count):
    """Equality-constrained original linear allocation, passive coordinates only.

    Actual constraint vector and full Jacobian stay measurement-anchored.
    Fixed target shift enters the intercept; OTA/power columns cannot optimize.
    The .04/.08 trust radius applies to passives and ETA. Power is the
    native output goal, including its independent .95*spec clamp.
    """

    def bounded(q, domain, c, jacobian, **kwargs):
        import numpy as np
        from rc_agent.allocation import LogDomain, Proposal
        q = np.asarray(q, float)
        j = np.asarray(jacobian, float)
        if len(q) == passive_count:
            return update(q, domain, c, jacobian, **kwargs)
        fixed = np.asarray(target_getter(), float)
        if len(q) != passive_count + 4 or fixed.shape != (4,) or np.any(fixed <= 0):
            raise ValueError('fixed_allocation_shape')
        ref, lo, hi = domain.arrays()
        dz = np.log(fixed / q[passive_count:])
        scales = np.asarray(kwargs['scale'], float)[passive_count:]
        reasons = fixed_coordinate_reasons(q[passive_count:], fixed, lo[passive_count:], hi[passive_count:], scales, kwargs['delta'])
        if reasons:
            from backends.lowpass import ToolUnavailable
            raise ToolUnavailable('fixed_target_outside_original_current_trustbox')
        reduced_domain = LogDomain(tuple(ref[:passive_count]), tuple(lo[:passive_count]), tuple(hi[:passive_count]))
        reduced_kwargs = {**kwargs, 'scale': np.asarray(kwargs['scale'])[:passive_count]}
        r = update(q[:passive_count], reduced_domain, np.asarray(c) + j[:, passive_count:] @ dz, j[:, :passive_count], **reduced_kwargs)
        constant = kwargs['regularization'] * float(np.sum((scales * dz) ** 2))
        return Proposal(tuple(r.values) + tuple(fixed), tuple(r.delta_z) + tuple(dz), r.objective_before + constant, r.objective_after + constant, r.iterations)
    return bounded

def fixed_coordinate_reasons(achieved, fixed, lower, upper, scale, radius):
    """One common equality-feasibility check for solver and predecision view."""
    import numpy as np
    a, t, lo, hi, scale = (np.asarray(v, float) for v in (achieved, fixed, lower, upper, scale))
    if any((v.shape != (4,) for v in (a, t, lo, hi, scale))) or any((not np.all(np.isfinite(v)) for v in (a, t, lo, hi, scale))) or np.any(a <= 0) or np.any(t <= 0):
        return ['fixed_target_or_achieved_anchor_missing_or_invalid']
    reasons = []
    if np.any(a[:3] < lo[:3]) or np.any(a[:3] > hi[:3]):
        reasons.append('achieved_OTA_outside_original_descriptor_domain')
    if np.any(t[:3] < lo[:3]) or np.any(t[:3] > hi[:3]):
        reasons.append('fixed_OTA_target_outside_original_descriptor_domain')
    if np.any(np.abs(np.log(t[:3] / a[:3])) * scale[:3] > radius + 1e-12):
        reasons.append('fixed_output_target_outside_original_current_local_radius')
    return reasons

def install_targets(backend):
    from provider.stream import injected
    import backends.lowpass_numerics as t1n
    import backends.highpass_numerics as typen
    from backends.lowpass import T1Backend
    from backends.highpass import HighpassBackend
    from dataclasses import asdict
    native_initialize = backend.initialize
    native_characterize = backend.characterize
    native_calibrate = backend.calibrate
    locked = {'eta': None, 'power': None, 'target_status': 'initialization_not_attempted'}
    phase = {'initializing': False}

    def initialize(self):
        phase['initializing'] = True
        try:
            return native_initialize()
        finally:
            phase['initializing'] = False
            if locked['eta'] is None:
                from backends.acceptance import initial_pass
                locked['target_status'] = 'not_needed_initialPASS_lazy' if initial_pass(self) else 'awaiting_first_valid_independent_bootstrap'
    backend.initialize = MethodType(initialize, backend)

    def characterize(self, *a, **k):
        eta, evidence = native_characterize(*a, **k)
        if locked['eta'] is None and getattr(backend, '_capture_bootstrap_target', False):
            capture = backend._bootstrap_capture_row
            if capture.get('measurement_status', 'valid') != 'valid':
                return (eta, evidence)
            p = float(capture['metrics']['power_w'])
            if not math.isfinite(p) or p <= 0:
                raise ValueError('initial_measured_power_unavailable_for_fixed_target')
            limit = backend.task['power_limit_w'] if backend.case['type_id'] == 'T1' else backend.case['power_max_uW'] * 1e-06
            locked['power'] = min(p, 0.95 * limit)
            locked['eta'] = asdict(eta)
            locked['target_status'] = 'captured_only_at_first_valid_independent_bootstrap'
            backend.event('fixed_initial_allocation_targets', copy.deepcopy(locked))
        return (eta, evidence)
    backend.characterize = MethodType(characterize, backend)
    previous_eta = backend.plan_eta
    backend._ablation_plan_eta = previous_eta
    original_type = type(backend)

    def get_eta(self):
        if locked['eta'] is not None:
            from rc_agent.models import OTA
            return OTA(**locked['eta'])
        return self._ablation_plan_eta

    def set_eta(self, value):
        self._ablation_plan_eta = value
    backend.__class__ = type('FixedTarget' + original_type.__name__, (original_type,), {'plan_eta': property(get_eta, set_eta)})

    def calibrate(self):
        result = native_calibrate()
        if locked['eta'] is not None:
            result.update(plan_rebased=False, calibration_model_updated=True, frozen_planning_target_preserved=True)
        return result
    backend.calibrate = MethodType(calibrate, backend)

    def target():
        if locked['eta'] is None:
            from backends.lowpass import ToolUnavailable
            raise ToolUnavailable('first_valid_independent_bootstrap_target_not_yet_available')
        return [locked['eta'][k] for k in ('a0', 'wp_rad_s', 'ro_ohm')] + [locked['power']]
    typed = backend.case['type_id'] != 'T1'
    module = typen if typed else t1n
    names = backend.names if typed else t1n.NAMES
    n = len(names) - 3
    propose = injected(module.propose, bounded_update=fixed_update(module.bounded_update, target, n))
    native = HighpassBackend.allocation if typed else T1Backend.allocation
    allocate = injected(native, propose=propose).__get__(backend, type(backend))

    def allocation(self, coupled, radius='normal'):
        plan = allocate(coupled, radius)
        if coupled:
            plan['ota_target'] = copy.deepcopy(locked['eta'])
            plan['power_target_w'] = locked['power']
            plan['fixed_target_equality_constraint'] = True
        return plan
    backend.allocation = MethodType(allocation, backend)
    backend.fixed_allocation_targets = locked
    native_snapshot = backend.snapshot
    availability_audit = []
    backend.fixed_target_availability_audit = availability_audit

    def snapshot(self, step):
        from evidence import save
        import numpy as np
        state = native_snapshot(step)
        diagnostic = state.get('diagnostics') or {}
        available = {}
        for radius, delta in (('small', 0.04), ('normal', 0.08)):
            reasons = []
            if locked['eta'] is None:
                reasons = ['first_valid_independent_bootstrap_target_not_yet_available']
            elif 'coupled_step' not in state['allowed_actions']:
                reasons = ['native_coupled_model_or_budget_precondition_unavailable']
            else:
                try:
                    eta = diagnostic['achieved_eta']
                    power = float(state['observation']['metrics']['power_w'])
                    a = [eta[k] for k in ('a0', 'wp_rad_s', 'ro_ohm')] + [power]
                    anchor = np.asarray(self.descriptor_anchor, float)
                    lower = list(anchor * 0.5) + [power * math.exp(-0.16)]
                    upper = list(anchor * 2.0) + [power * math.exp(0.16)]
                    reasons = fixed_coordinate_reasons(a, target(), lower, upper, [1.0] * 4, delta)
                except (KeyError, TypeError, ValueError):
                    reasons = ['fixed_target_or_achieved_anchor_missing_or_invalid']
            available[radius] = {'available': not reasons, 'reasons': reasons}
        view = {'by_radius': available, 'coupled_action_retained': True, 'device_refinement_still_in_coupled_tool': True, 'guidance': 'Prefer an available radius. An unavailable fixed allocation is a tool failure, not a contract rejection. RC/probe/calibration retain their native paths.'}
        state['fixed_target_allocation_availability'] = view
        state.pop('state_id', None)
        state['state_id'] = digest(state)
        availability_audit.append({'step': step, 'candidate_id': state['observation']['candidate_id'], **view})
        save(self.out / 'fixed_target_availability.json', availability_audit)
        return state
    backend.snapshot = MethodType(snapshot, backend)
