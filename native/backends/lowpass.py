from runtime.paths import CODE_ROOT
'M4-owned bridge. Public T1 evaluate/finish and CommonExecution stay unmodified.'
from dataclasses import asdict
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
ROOT = CODE_ROOT
sys.path.insert(0, str(ROOT / 'execution'))
from t1_adapter import T1Run, task_from_case, measured_runner, read
from common_execution import CommonExecution, ExecutionPaused, CandidateBudgetExhausted, save, value_hash
from rc_agent import physical
from rc_agent.models import OTA
from rc_agent.relations import Passives
from rc_agent.diagnostics import Response, decompose, model_trusted
from rc_agent.knowledge import KnowledgeBase, RetrievalContext
from rc_agent.native_powell import native_powell
from rc_agent.baseline import task_metrics
from runtime.integrity import manifest as source_manifest
from backends.lowpass_numerics import NAMES
from backends.lowpass_numerics import PASSIVE_NAMES
from backends.lowpass_numerics import DEVICE_NAMES
from backends.lowpass_numerics import analytic_initial
from backends.lowpass_numerics import model_response
from backends.lowpass_numerics import propose
from backends.lowpass_numerics import constraints
VERSION = 'm4_t1_m1_aligned_development_v2'
PORT_ID = 'SKLP:vin=1.2V;VDD=1.8V;ideal_source;TT;Cg_y_ground;Cf_x_out;unity_feedback'

class ToolUnavailable(RuntimeError):
    pass

class SizingBlockComplete(RuntimeError):
    pass

class T1Backend(T1Run):
    """Inherit verified public evaluate/finish without spoofing another method.

    M4 has its own constructor because the public constructor's method whitelist
    predates M4. Auxiliary characterization candidates are conservatively counted
    in the same candidate cap and consume the same search process/analysis pool.
    They can never become the selected full-filter endpoint.
    """

    def __init__(self, case, out, protocol, ngspice, *, launch_authorized=False):
        self.case, self.method = (case, 'M4_kira')
        legacy = read(ROOT / 'resources/configs/sallen_key.json')
        self.task, self.common = task_from_case(case, legacy)
        self.legacy = {'physical_bounds': legacy['physical_bounds'], 'initial': {'rz_ohm': legacy['initial']['rz_ohm']}}
        self.engine = CommonExecution(out, protocol, case['level'], ngspice, launch_authorized=launch_authorized)
        self.out = self.engine.directory
        self.rows, self.best, self.first, self.endpoint = ([], None, None, None)
        self.started, self.state, self.stop = (time.monotonic(), {}, 'not_started')
        self.bounds = self.legacy['physical_bounds']
        self.optimizer_settings = legacy['optimizer']
        self.descriptor_anchor = None
        self.current = None
        self.descriptors = {}
        self.auxiliary = []
        self.calibration = None
        self.plan_eta = None
        self.plan_calibration = None
        self.calibrations = 0
        self.probes = 0
        self.last_diagnostic = None
        self.last_result = {'status': 'not_started'}
        self.kb = KnowledgeBase()
        self.events = []
        self.sizing_active = False
        save(self.out / 'source_manifest.json', source_manifest())
        save(self.out / 'input.json', {'case': case, 'method': self.method, 'task': self.task, 'common_acceptance': self.common, 'domain': self.bounds, 'initialization': 'analytic equal-R SK plus public Devices defaults, same device seed as M1; no result reads', 'protocol': protocol, 'version': VERSION, 'classification': 'development pilot only; real feasibility unverified', 'auxiliary_budget': 'OTA characterizations count in shared candidate/process/analysis caps'})

    def costs(self):
        result = super().costs()
        result.update(full_filter_search_candidates=sum((not r['endpoint_check'] for r in self.rows)), ota_characterization_candidates=len(self.auxiliary), api_seconds=sum((x.get('wall_seconds') or 0 for x in self.engine.api_calls)), prompt_tokens=sum((x.get('prompt_tokens') or 0 for x in self.engine.api_calls)), completion_tokens=sum((x.get('completion_tokens') or 0 for x in self.engine.api_calls)), total_tokens=sum((x.get('total_tokens') or 0 for x in self.engine.api_calls)), token_usage_complete=all((x.get('total_tokens') is not None for x in self.engine.api_calls)), monetary_cost=None, monetary_cost_status='not_estimated; no frozen tariff or billing receipt')
        return result

    def event(self, kind, data):
        self.events.append({'index': len(self.events), 'kind': kind, 'data': data, 'cost': self.costs()})
        save(self.out / 'tool_events.json', self.events)

    def room(self, slots, analyses, candidates=1):
        e = self.engine
        search = [s for s in e.slots if not s['endpoint']]
        return e.status != 'paused' and len(e.cache) + candidates <= e.limits['max_new_candidates'] and (len(search) + slots <= e.limits['search_base_process_slots']) and (sum((s['analysis_count'] for s in search)) + analyses <= e.limits['search_base_analysis_allowance'])

    def devices(self, x):
        return physical.Devices(*(x[n] for n in DEVICE_NAMES), self.legacy['initial']['rz_ohm'])

    def characterize(self, x, *, fresh=False):
        key = value_hash({n: float(x[n]) for n in DEVICE_NAMES})
        if key in self.descriptors and (not fresh):
            return self.descriptors[key]
        enough = self.room(3, 12, 2) if self.sizing_active else self.room(1, 2)
        if not enough:
            raise ToolUnavailable('Shared auxiliary allowance exhausted; endpoint reserve preserved')
        parameters = {'measurement_kind': 'ota_descriptor', 'devices': asdict(self.devices(x))}
        if fresh:
            parameters['independent_repeat'] = len(self.auxiliary)
        candidate, cached = self.engine.candidate(parameters)
        if cached:
            raise ToolUnavailable('Failed auxiliary point cannot be silently retried')
        record = {'candidate_id': candidate['id'], 'devices': parameters['devices'], 'status': 'started', 'fresh': fresh}
        self.auxiliary.append(record)
        folder = self.out / ('ota_%03d' % candidate['id'])
        measured = False
        try:
            with measured_runner(self.engine, candidate):
                budgeted_run = physical.run_deck

                def wide_ota_run(deck, directory, budget, tag):
                    expected = 'ac dec 40 1 1e9'
                    if tag != 'ota_descriptor' or expected not in deck:
                        raise ValueError('Unexpected independent characterization deck')
                    return budgeted_run(deck.replace(expected, 'ac dec 40 0.001 1e9'), directory, budget, tag)
                physical.run_deck = wide_ota_run
                try:
                    eta, evidence = physical.characterize(self.devices(x), folder, None)
                finally:
                    physical.run_deck = budgeted_run
            evidence['m4_characterization_grid'] = '0.001Hz to 1GHz, 40/decade; identical to M1 independent OTA A/Z'
            measured = True
            record.update(status='measured', eta=asdict(eta), evidence=evidence)
            result = (eta, evidence)
            self.descriptors[key] = result
            if self.descriptor_anchor is None:
                self.descriptor_anchor = tuple(eta.values)
            return result
        except (ValueError, RuntimeError) as exc:
            record.update(status='unknown', error_class=type(exc).__name__)
            raise
        finally:
            self.engine.finish_candidate(candidate, measured=measured, accepted=False)
            save(self.out / 'auxiliary.json', self.auxiliary)
            self.event('independent_ota_characterization', record)
            if self.engine.status == 'paused':
                raise ExecutionPaused('Auxiliary infrastructure failure paused the common engine')

    def initialize(self):
        x = analytic_initial(self.task, self.bounds)
        if not self.room(3, 12, 2):
            raise ToolUnavailable('Insufficient shared budget for initial full response and OTA characterization')
        self.current = self.evaluate(x)
        self.event('analytic_initial_full_measurement', self.current)
        if self.current.get('evidence'):
            eta, evidence = self.characterize(x)
            self.plan_eta = eta
            self.event('initial_one_pole_diagnostic', self.diagnose())
            self.plan_calibration = physical.calibration_from_evidence(eta, evidence)
            self.last_diagnostic = None
            self.event('common_model_bootstrap', {'model': 'independent_unloaded_OTA_sampled_A_Z_local_shape', 'descriptor_anchor': list(self.descriptor_anchor), 'descriptor_scale_bounds': [0.5, 2.0], 'evidence': evidence['evidence'], 'extra_spice': 0, 'source': 'same independent characterization, no full-filter fit'})
        return self.current

    def wave(self, row):
        if not row.get('evidence'):
            raise ToolUnavailable('Current complete response missing')
        a = physical.table(ROOT / row['evidence'], 'filter_ac.dat', 3)
        return (a[:, 0], a[:, 1] + 1j * a[:, 2])

    def diagnose(self):
        row = self.current
        f, h = self.wave(row)
        eta, evidence = self.characterize(row['parameters'])
        if abs(evidence['dc_out'] - row['metrics']['dc_out']) > 1e-07:
            raise ToolUnavailable('Independent OTA/full-circuit DC conditions do not match')
        p = Passives(*(row['parameters'][n] for n in PASSIVE_NAMES))
        planned = self.plan_eta or eta
        model = self.plan_calibration
        ha = model_response(p, eta, f, self.task['load_f'], model)
        hp = model_response(p, planned, f, self.task['load_f'], model)
        s = 1j * f / self.task['f0_hz']
        ht = 1 / (1 + s / self.task['q'] + s * s)

        def response(values, source):
            return Response(str(row['candidate_id']), p, tuple(f), tuple(values), source)
        residual = decompose(response(h, 'full_spice'), response(ha, 'achieved_model'), response(hp, 'planned_model'), response(ht, 'target'), 0.01)
        actual_constraints = constraints(row['metrics'], self.task, self.common)
        trust_band = (f >= self.task['f0_hz'] * 0.1) & (f <= self.task['f0_hz'] * 10)
        if np.count_nonzero(trust_band) < 20:
            raise ToolUnavailable('Insufficient shared M1/M4 model trust band')
        band_errors = [float(np.max(np.abs(np.asarray(part)[trust_band]) / np.maximum(np.abs(ht[trust_band]), 0.01))) for part in (residual.model, residual.realization, residual.planning)]
        diagnostic = {'candidate_id': row['candidate_id'], 'port_id': PORT_ID, 'load_f': self.task['load_f'], 'dc_match_tolerance_v': 1e-07, 'frequency_grid_sha256': value_hash(f.tolist()), 'actual_passives': asdict(p), 'achieved_eta': asdict(eta), 'planned_tau': asdict(planned), 'model_id': 'one_pole_series_ro' if model is None else 'independent_ota_shape_local_frozen', 'model_error': residual.errors[0], 'realization_error': residual.errors[1], 'planning_error': residual.errors[2], 'total_error': residual.total_error, 'normalization_floor': 0.01, 'model_trust_threshold': 0.08, 'trust_band_hz': [self.task['f0_hz'] * 0.1, self.task['f0_hz'] * 10], 'trust_band_component_errors': band_errors, 'model_trusted': bool(residual.measured and band_errors[0] <= 0.08), 'full_grid_errors_remain_reported': True, 'descriptor_domain_anchor': list(self.descriptor_anchor), 'descriptor_domain_scale': [0.5, 2.0], 'actual_constraints': actual_constraints.tolist(), 'acceptance': 'only public full circuit evaluate/independent finish', 'cancellation_is_trust': False}
        raw = np.column_stack([f, h.real, h.imag, ha.real, ha.imag, hp.real, hp.imag, ht.real, ht.imag, np.real(residual.model), np.imag(residual.model), np.real(residual.realization), np.imag(residual.realization), np.real(residual.planning), np.imag(residual.planning)])
        np.savetxt(self.out / ('residual_%03d_%02d.csv' % (row['candidate_id'], self.calibrations)), raw, delimiter=',', header='f,h_re,h_im,ha_re,ha_im,hp_re,hp_im,target_re,target_im,rmod_re,rmod_im,rreal_re,rreal_im,rplan_re,rplan_im')
        self.last_diagnostic = diagnostic
        return diagnostic

    def allocation(self, coupled, radius='normal'):
        diagnostic = self.last_diagnostic or self.diagnose()
        f, _ = self.wave(self.current)
        eta, _ = self.characterize(self.current['parameters'])
        return propose(self.current, eta, f, self.task, self.common, self.bounds, calibration=self.plan_calibration, coupled=coupled, radius=radius, trusted=diagnostic['model_trusted'], descriptor_anchor=self.descriptor_anchor)

    def snapshot(self, step):
        diagnostic, error, rc = (None, None, None)
        try:
            diagnostic = self.diagnose()
            if diagnostic['model_trusted']:
                rc = self.allocation(False)
        except (ValueError, ToolUnavailable, RuntimeError) as exc:
            if isinstance(exc, ExecutionPaused) or self.engine.status == 'paused':
                raise
            error = type(exc).__name__ + ': ' + str(exc)
        allowed = ['finish']
        if self.current and self.current['status'] != 'success':
            if self.probes < 6 and self.room(3, 12, 2):
                allowed.append('probe')
            if diagnostic and self.calibrations < 2 and self.room(3, 12, 2):
                allowed.append('calibrate')
            if diagnostic and diagnostic['model_trusted'] and self.room(3, 12, 2):
                if rc and rc['rc_only_feasible']:
                    allowed.append('rc_step')
                if self.room(4, 14, 3):
                    allowed.append('coupled_step')
        cid = str(self.current['candidate_id']) if self.current else 'none'
        stable = self.current.get('metrics', {}).get('stable') if self.current else None
        ctx = RetrievalContext(self.case['case_id'], cid, PORT_ID, diagnostic['model_id'] if diagnostic else 'unqualified', consumer='framework_llm_interface', evidence=tuple(((name, 'observation:' + cid, 'measured_or_static') for name in ['node_mapping', 'port_identity', 'source_load_known', 'legal_domain', 'ideal_reference_assumptions'] + (['response_validity', 'actual_power', 'amp_complex_response'] if diagnostic else []) + (['model_validity'] if diagnostic and diagnostic['model_trusted'] else []))), stability_status='pass' if stable is True else 'fail' if stable is False else 'unknown', remaining_spice_processes=self.engine.limits['search_base_process_slots'] - sum((not s['endpoint'] for s in self.engine.slots)))
        tags = ('model_residual', 'realization_residual', 'planning_residual', 'power', 'frequency', 'q')
        knowledge = self.kb.retrieve(ctx, tags)
        evidence_ids = ['observation:' + cid, 'tool_result:' + str(step)]
        if diagnostic:
            evidence_ids.append('diagnostics:' + cid)
        payload = {'step': step, 'case': self.case, 'observation': self.current, 'diagnostics': diagnostic, 'diagnostic_error': error, 'rc_only_assessment': None if rc is None else {'feasible_with_margin': rc['rc_only_feasible'], 'predicted_constraints': rc['predicted_constraints_with_margin']}, 'allowed_actions': allowed, 'evidence_ids': evidence_ids, 'knowledge': knowledge, 'knowledge_ids': [c['id'] for c in knowledge['cards']], 'cost': self.costs(), 'limits': self.engine.limits, 'last_tool_result': self.last_result, 'tool_history': self.events[-6:], 'calibration_count': self.calibrations, 'probe_count': self.probes}
        payload['state_id'] = value_hash(payload)
        return payload

    def accept_trial(self, row, *, planned_eta=None, planned_model=None, force=False):
        previous = self.current
        old_plan, old_model, old_diagnostic = (self.plan_eta, self.plan_calibration, self.last_diagnostic)
        accepted = bool(row.get('evidence') and (force or row['status'] == 'success' or row['loss'] < previous['loss']))
        diagnostic, diagnostic_error = (None, None)
        eta = None
        if row.get('evidence'):
            try:
                eta, _ = self.characterize(row['parameters'])
                self.current, self.plan_eta = (row, planned_eta or eta)
                self.plan_calibration, self.last_diagnostic = (planned_model, None)
                diagnostic = self.diagnose()
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, ExecutionPaused) or self.engine.status == 'paused':
                    raise
                diagnostic_error = type(exc).__name__ + ': ' + str(exc)
            finally:
                self.current, self.plan_eta = (previous, old_plan)
                self.plan_calibration, self.last_diagnostic = (old_model, old_diagnostic)
        if accepted:
            self.current = row
            self.plan_eta = planned_eta or eta
            self.plan_calibration = planned_model
            self.last_diagnostic = diagnostic
        return {'selected': accepted, 'rollback': not accepted, 'previous_candidate': previous['candidate_id'], 'trial': row, 'trial_diagnostics': diagnostic, 'trial_diagnostic_error': diagnostic_error}

    def calibrate(self):
        self.calibrations += 1
        eta, evidence = self.characterize(self.current['parameters'], fresh=True)
        calibration = physical.calibration_from_evidence(eta, evidence)
        row = self.repeat_full(self.current['parameters'])
        self.current = row
        self.plan_eta = eta
        self.plan_calibration = calibration
        self.last_diagnostic = None
        diagnostic = self.diagnose()
        return {'status': 'calibrated' if diagnostic['model_trusted'] else 'still_untrusted', 'diagnostic': diagnostic, 'plan_rebased': True, 'total_error_unchanged_by_model_definition': True}

    def repeat_full(self, x):
        if not self.room(2, 10):
            raise ToolUnavailable('No same-point full remeasurement allowance')
        candidate, cached = self.engine.candidate({'measurement_kind': 'independent_full_repeat', 'parameters': {n: float(x[n]) for n in NAMES}, 'repeat': self.calibrations})
        if cached:
            raise ToolUnavailable('Independent remeasurement already attempted')
        folder = self.out / ('remeasured_%03d' % candidate['id'])
        row = {'candidate_id': candidate['id'], 'parameters': dict(x), 'endpoint_check': False, 'independent_calibration_check': True, 'status': 'unknown', 'failed_constraints': ['measurement_invalid'], 'loss': 1000000000.0, 'metrics': {}}
        measured = False
        try:
            with measured_runner(self.engine, candidate):
                f, h, result = physical.measure_filter(Passives(*(x[n] for n in PASSIVE_NAMES)), self.devices(x), self.task['f0_hz'], folder, None, load_f=self.task['load_f'])
            metrics, status, failed, loss = task_metrics(f, h, result['metrics'], self.task, self.common)
            measured = len(candidate['slot_ids']) == 2 and all((self.engine.slots[i]['status'] == 'completed' for i in candidate['slot_ids']))
            row.update(metrics=metrics, status=status, failed_constraints=failed, loss=loss, evidence=result['evidence'])
        finally:
            self.engine.finish_candidate(candidate, measured=measured, accepted=row['status'] == 'success')
            row['cost_to_observation'] = self.costs()
            self.rows.append(row)
            save(self.out / 'trajectory.json', self.rows)
        if self.best is None or row['loss'] < self.best['loss']:
            self.best = row
        if row['status'] == 'success' and self.first is None:
            self.first = row
        return row

    def size_devices(self, plan, max_objectives=12):
        initial = dict(self.current['parameters'])
        target = np.array([plan['ota_target'][k] for k in ('a0', 'wp_rad_s', 'ro_ohm')])
        history, best = ([], None)
        native_state = {}

        def objective(z):
            nonlocal best
            if len(history) >= max_objectives:
                raise SizingBlockComplete('Finite M4 sizing block complete')
            trial = dict(initial)
            trial.update({n: float(v) for n, v in zip(DEVICE_NAMES, np.exp(z))})
            eta, evidence = self.characterize(trial)
            value = float(np.sum(np.maximum(np.abs(np.log(np.array(eta.values) / target)) / np.array([0.25, 0.25, 0.35]) - 1, 0) ** 2))
            value += max(math.log(evidence['power_w'] / plan['power_target_w']) / 0.05, 0) ** 2
            value += 1000 * max(plan['saturation_target_v'] - evidence['minimum_saturation_margin_v'], 0) ** 2
            record = {'parameters': trial, 'eta': asdict(eta), 'loss': value, 'power_w': evidence['power_w'], 'minimum_saturation_margin_v': evidence['minimum_saturation_margin_v'], 'dc_out': evidence['dc_out'], 'evidence': evidence['evidence']}
            history.append(record)
            if best is None or value < best['loss']:
                best = record
            return value
        result = None
        self.sizing_active = True
        try:
            result = native_powell(objective, np.log([initial[n] for n in DEVICE_NAMES]), np.log([self.bounds[n] for n in DEVICE_NAMES]), self.optimizer_settings, native_state, lambda *_: None)
        except (ToolUnavailable, SizingBlockComplete):
            pass
        finally:
            self.sizing_active = False
            self.event('device_target_realization', {'target': plan['ota_target'], 'power_target_w': plan['power_target_w'], 'saturation_target_v': plan['saturation_target_v'], 'history': history, 'best': best, 'target_realized_within_tolerances': bool(best is not None and np.all(np.abs(np.log(np.array([best['eta'][k] for k in ('a0', 'wp_rad_s', 'ro_ohm')]) / target)) <= np.array([0.25, 0.25, 0.35])) and (best['power_w'] <= plan['power_target_w']) and (best['minimum_saturation_margin_v'] >= plan['saturation_target_v']) and (self.common['dc_out_range_v'][0] <= best['dc_out'] <= self.common['dc_out_range_v'][1])), 'unmet_target_meaning': 'not realized within this finite sizing allocation; not global infeasibility', 'system_acceptance': False, 'optimizer': 'shared rc_agent.native_powell.native_powell', 'native_state': native_state, 'max_objective_calls': max_objectives, 'loss_alignment': 'M1 descriptor dead-zone + log-power + headroom; M4 allocated targets'})
        if best is None:
            raise ToolUnavailable('No measured device implementation')
        return best['parameters']

    def execute(self, decision):
        name, args = (decision['action']['name'], decision['action']['args'])
        if name == 'finish':
            return {'status': 'finish_requested'}
        if name == 'calibrate':
            result = self.calibrate()
        elif name == 'probe':
            self.probes += 1
            x = dict(self.current['parameters'])
            axis = args['axis']
            value = x[axis] * math.exp(0.04 * args['direction'])
            if not self.bounds[axis][0] <= value <= self.bounds[axis][1]:
                raise ToolUnavailable('Probe outside fixed domain')
            x[axis] = value
            row = self.evaluate(x)
            result = self.accept_trial(row, planned_eta=self.plan_eta, planned_model=self.plan_calibration)
        elif name in {'rc_step', 'coupled_step'}:
            plan = self.allocation(name == 'coupled_step', args['radius'])
            self.event('bounded_allocation', plan)
            if name == 'rc_step' and (not plan['rc_only_feasible']):
                raise ToolUnavailable('RC-only feasibility precondition failed')
            x = dict(self.current['parameters'])
            if name == 'coupled_step':
                x = self.size_devices(plan)
            x.update(plan['passives'])
            if not self.room(2, 10):
                raise ToolUnavailable('Preserve independent endpoint; no full measurement allowance')
            row = self.evaluate(x)
            result = self.accept_trial(row, planned_eta=OTA(**plan['ota_target']), planned_model=self.plan_calibration)
            result['allocation'] = plan
        else:
            raise ValueError('Action dispatch denied')
        self.last_result = result
        self.event(name, result)
        return result

    def finish(self):
        result = super().finish()
        result.update(version=VERSION, method_id='M4', method_name='KIRA', native_optimizer=self.state, cost=self.costs(), llm_enabled=bool(self.state.get('llm_enabled')), complete_kira_implementation=True, complete_kira_empirically_validated=False, supported_types=['T1'], knowledge_version=self.kb.manifest['knowledge_version'])
        if result['status'] != 'success':
            result['time_to_success'] = None
        save(self.out / 'result.json', result)
        return result
