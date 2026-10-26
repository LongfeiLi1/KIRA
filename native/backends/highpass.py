"""M4 T2 bridge: public HighpassRun measurements, owned action controller."""
from dataclasses import asdict
from pathlib import Path
import math
import time
import numpy as np
from execution.highpass.adapter import ROOT, HighpassRun, binding, unpack, physical, physical_metrics, read, DEVICE_NAMES, PASSIVE_NAMES
from execution.highpass.execution import CommonExecution, ExecutionPaused, CandidateBudgetExhausted, save, value_hash
import backends.highpass_characterization as characterization
from execution.device_model.model import network_response
from numerics.metrics.response import ideal_response
from rc_agent.models import OTA
from rc_agent.native_powell import native_powell
from backends.lowpass import T1Backend
from backends.lowpass import ToolUnavailable
from backends.lowpass import SizingBlockComplete
from backends.highpass_knowledge import HighpassKnowledge
from backends.highpass_numerics import constraints
from backends.highpass_numerics import propose
from runtime.integrity import manifest as source_manifest
VERSION = 'm4_type_graph_development_v3'
DC_TOLERANCE = 1e-07

class HighpassBackend(T1Backend):
    """Only topology-neutral utility methods are inherited from frozen T1.

    The T1 constructor, evaluate, finish, SK formulas, and monkey-patched
    characterization are never called. Public HighpassRun owns circuit acceptance.
    """

    def __init__(self, case, out, protocol, ngspice, *, executor=None, launch_authorized=False):
        self.case, self.method = (case, 'M4')
        self.binding = binding(case)
        self.names, self.bounds = (self.binding['names'], self.binding['physical_bounds'])
        legacy = read(ROOT / 'resources/configs/sallen_key.json')
        self.common, self.optimizer_settings = (legacy['common_acceptance'], legacy['optimizer'])
        self.settings = self.optimizer_settings
        self.engine = CommonExecution(out, protocol, case['level'], ngspice, executor=executor, launch_authorized=launch_authorized)
        self.out = self.engine.directory
        self.rows, self.best, self.first, self.endpoint = ([], None, None, None)
        self.started, self.state, self.stop = (time.monotonic(), {}, 'not_started')
        self.current, self.descriptor_anchor = (None, None)
        self.descriptors, self.calibration_models = ({}, {})
        self.auxiliary, self.events = ([], [])
        self.plan_eta, self.plan_calibration, self.last_diagnostic = (None, None, None)
        self.calibrations, self.probes, self.sizing_active = (0, 0, False)
        self.last_result = {'status': 'not_started'}
        self.kb = HighpassKnowledge()
        save(self.out / 'source_manifest.json', source_manifest())
        save(self.out / 'input.json', {'case': case, 'method': self.method, 'binding': self.binding, 'protocol': protocol, 'version': VERSION, 'common_acceptance': self.common, 'initialization': self.binding['initial_policy'], 'classification': 'development; only public independent endpoint proves circuit acceptance', 'auxiliary_budget': 'all OTA attempts charge common candidate/process/analysis caps'})

    def evaluate(self, x, endpoint=False):
        return HighpassRun.evaluate(self, x, endpoint)

    def costs(self):
        result = HighpassRun.costs(self)
        result.update(full_filter_search_candidates=sum((not r['endpoint_check'] for r in self.rows)), ota_characterization_candidates=len(self.auxiliary), api_seconds=sum((r.get('wall_seconds') or 0 for r in self.engine.api_calls)), token_usage_complete=all((r.get('total_tokens') is not None for r in self.engine.api_calls)))
        return result

    def devices(self, x):
        return physical.Devices(**{n: float(x[n]) for n in DEVICE_NAMES}, **self.binding['fixed'])

    def descriptor_key(self, x):
        names = DEVICE_NAMES + ()
        return value_hash({n: float(x[n]) for n in names})

    def characterize(self, x, *, fresh=False):
        key = self.descriptor_key(x)
        if key in self.descriptors and (not fresh):
            return self.descriptors[key]
        if not (self.room(3, 13, 2) if self.sizing_active else self.room(1, 2)):
            raise ToolUnavailable('Shared OTA allowance exhausted; full-stage and endpoint reserves preserved')
        parameters = {'measurement_kind': 'ota_descriptor', 'devices': asdict(self.devices(x)), 'dc_condition_key': key}
        if fresh:
            parameters['independent_repeat'] = len(self.auxiliary)
        candidate, cached = self.engine.candidate(parameters)
        if cached:
            raise ToolUnavailable('Failed auxiliary candidate cannot be silently retried')
        record = {'candidate_id': candidate['id'], 'devices': parameters['devices'], 'status': 'started', 'fresh': fresh}
        self.auxiliary.append(record)
        measured = False
        try:
            outcome = self.engine.run_spice(candidate, 'ota_descriptor', characterization.deck(self.devices(x), self.case, x))
            if outcome['status'] != 'completed':
                raise RuntimeError('Common OTA execution ' + outcome['status'])
            folder = Path(outcome['directory'])
            eta, calibration, evidence = characterization.analyze(folder, self.devices(x), self.case['type_id'])
            save(folder / 'measurement.json', evidence)
            measured = True
            record.update(status='measured', eta=asdict(eta), evidence=evidence)
            self.descriptors[key] = (eta, evidence)
            self.calibration_models[key] = calibration
            if self.descriptor_anchor is None:
                self.descriptor_anchor = tuple(eta.values)
            return (eta, evidence)
        except (ValueError, RuntimeError, OSError) as exc:
            record.update(status='unknown', error_class=type(exc).__name__)
            if candidate['status'] == 'pending':
                candidate.update(status='execution_error', infrastructure_class='measurement_extraction_error')
            raise
        finally:
            self.engine.finish_candidate(candidate, measured=measured, accepted=False)
            save(self.out / 'auxiliary.json', self.auxiliary)
            self.event('independent_ota_characterization', record)
            if self.engine.status == 'paused':
                raise ExecutionPaused('Auxiliary infrastructure failure paused the common engine')

    def initialize(self):
        if not self.room(3, 13, 2):
            raise ToolUnavailable('Insufficient initialization allowance')
        self.current = self.evaluate(dict(self.binding['initial']))
        self.event('immutable_analytic_initial_full_measurement', self.current)
        if self.current.get('measurement_status') == 'valid':
            self.plan_eta, evidence = self.characterize(self.current['parameters'])
            self.event('initial_one_pole_diagnostic', self.diagnose())
            self.plan_calibration = self.calibration_models[self.descriptor_key(self.current['parameters'])]
            self.last_diagnostic = None
            self.event('common_model_bootstrap', {'model': 'registered_topology_independent_OTA_sampled_A_Z', 'descriptor_anchor': list(self.descriptor_anchor), 'scale_bounds': [0.5, 2.0], 'evidence': evidence['evidence'], 'extra_spice': 0, 'full_filter_fit': False})
        return self.current

    def wave(self, row):
        if row.get('measurement_status') != 'valid' or not row.get('evidence'):
            raise ToolUnavailable('Valid complete Type response missing')
        a = physical.table(ROOT / row['evidence']['performance'], 'filter_ac.dat', 3)
        return (a[:, 0], a[:, 1] + 1j * a[:, 2])

    def diagnose(self):
        row = self.current
        f, h = self.wave(row)
        eta, evidence = self.characterize(row['parameters'])
        p = {n: row['parameters'][n] for n in PASSIVE_NAMES[self.case['type_id']]}
        planned, model = (self.plan_eta or eta, self.plan_calibration)
        load = self.case['load_cap_pF'] * 1e-12

        def response(a):
            if model is not None:
                return model.response(p, a, f, load)
            gain = a.a0 / (1 + 2j * np.pi * f / a.wp_rad_s)
            return network_response(self.case['type_id'], p, f, gain, np.full(len(f), a.ro_ohm), load)
        ha, hp = (response(eta), response(planned))
        ht = ideal_response(f, physical.data.response_profile(self.case))
        parts = [h - ha, ha - hp, hp - ht]
        if not np.allclose(sum(parts), h - ht, rtol=1e-10, atol=1e-12):
            raise ValueError('Three-residual telescoping identity failed')
        den = np.maximum(np.abs(ht), 0.01)
        band = (f >= 0.1 * self.case['f0_hz']) & (f <= 10 * self.case['f0_hz'])
        if np.count_nonzero(band) < 20:
            raise ToolUnavailable('Insufficient model trust band')
        errors = [float(np.max(np.abs(v) / den)) for v in parts]
        band_errors = [float(np.max((np.abs(v) / den)[band])) for v in parts]
        dc_difference = float(evidence['dc_out'] - row['metrics']['dc_out'])
        dc_matched = abs(dc_difference) <= DC_TOLERANCE
        diagnostic = {'candidate_id': row['candidate_id'], 'type_id': self.case['type_id'], 'port_id': self.binding['topology_sha256'], 'conditions': 'TT; VDD=1.8V; Vin/Vref DC=1.2V; ideal source', 'load_f': load, 'frequency_grid_sha256': value_hash(f.tolist()), 'actual_passives': p, 'achieved_eta': asdict(eta), 'planned_tau': asdict(planned), 'model_id': 'one_pole_series_ro_registered_graph' if model is None else 'independent_ota_shape_registered_graph', 'model_error': errors[0], 'realization_error': errors[1], 'planning_error': errors[2], 'total_error': float(np.max(np.abs(h - ht) / den)), 'normalization_floor': 0.01, 'model_trust_threshold': 0.08, 'trust_band_hz': [0.1 * self.case['f0_hz'], 10 * self.case['f0_hz']], 'trust_band_component_errors': band_errors, 'nominal_dc_conditions_equal': True, 'dc_output_difference_v': dc_difference, 'matched_operating_point': dc_matched, 'dc_match_tolerance_v': DC_TOLERANCE, 'model_trusted': bool(dc_matched and band_errors[0] <= 0.08), 'qualification_blocker': None if dc_matched else 'Independent topology DC bench differs from actual circuit; matched-point qualification failed', 'descriptor_domain_anchor': list(self.descriptor_anchor), 'descriptor_domain_scale': [0.5, 2.0], 'actual_constraints': constraints(row['metrics'], self.case, self.common).tolist(), 'acceptance': 'only public HighpassRun complete measurement and fresh independent endpoint', 'cancellation_is_trust': False, 'full_grid_errors_remain_reported': True}
        raw = np.column_stack([f] + [c for v in (h, ha, hp, ht, *parts) for c in (v.real, v.imag)])
        np.savetxt(self.out / ('residual_%03d_%02d.csv' % (row['candidate_id'], self.calibrations)), raw, delimiter=',', header='f,h_re,h_im,ha_re,ha_im,hp_re,hp_im,target_re,target_im,rmod_re,rmod_im,rreal_re,rreal_im,rplan_re,rplan_im')
        self.last_diagnostic = diagnostic
        return diagnostic

    def allocation(self, coupled, radius='normal'):
        diagnostic = self.last_diagnostic or self.diagnose()
        f, _ = self.wave(self.current)
        eta, _ = self.characterize(self.current['parameters'])
        if self.plan_calibration is None:
            raise ToolUnavailable('Qualified independent A/Z model missing')
        return propose(self.current, eta, self.plan_calibration, self.descriptor_anchor, f, self.case, self.binding, self.common, coupled=coupled, radius=radius, trusted=diagnostic['model_trusted'])

    def snapshot(self, step):
        diagnostic, error, rc = (None, None, None)
        try:
            diagnostic = self.diagnose()
            if diagnostic['model_trusted']:
                rc = self.allocation(False)
        except (ValueError, RuntimeError, OSError) as exc:
            if self.engine.status == 'paused':
                raise
            error = type(exc).__name__ + ': ' + str(exc)
        allowed = ['finish']
        if self.current and self.current['status'] != 'success':
            if self.probes < 6 and self.room(3, 13, 2):
                allowed.append('probe')
            if diagnostic and self.calibrations < 2 and self.room(3, 13, 2):
                allowed.append('calibrate')
            if diagnostic and diagnostic['model_trusted'] and self.room(3, 13, 2):
                if rc and rc['rc_only_feasible']:
                    allowed.append('rc_step')
                if self.room(4, 15, 3):
                    allowed.append('coupled_step')
        cid = str(self.current['candidate_id']) if self.current else 'none'
        knowledge = self.kb.retrieve(self.case, cid, self.binding, bool(diagnostic and diagnostic['model_trusted']))
        evidence_ids = ['observation:' + cid, 'tool_result:' + str(step)]
        if diagnostic:
            evidence_ids.append('diagnostics:' + cid)
        payload = {'step': step, 'case': self.case, 'parameter_names': self.names, 'observation': self.current, 'diagnostics': diagnostic, 'diagnostic_error': error, 'rc_only_assessment': None if rc is None else {'feasible_with_margin': rc['rc_only_feasible'], 'predicted_constraints': rc['predicted_constraints_with_margin']}, 'allowed_actions': allowed, 'evidence_ids': evidence_ids, 'knowledge': knowledge, 'knowledge_ids': [c['id'] for c in knowledge['cards']], 'cost': self.costs(), 'limits': self.engine.limits, 'last_tool_result': self.last_result, 'tool_history': self.events[-6:], 'calibration_count': self.calibrations, 'probe_count': self.probes}
        payload['state_id'] = value_hash(payload)
        return payload

    def repeat_full(self, x):
        if not self.room(2, 11):
            raise ToolUnavailable('No same-point 11-analysis remeasurement allowance')
        values, p, d = unpack(x, self.binding, self.case)
        candidate, cached = self.engine.candidate({'measurement_kind': 'independent_full_repeat', 'parameters': values, 'repeat': self.calibrations})
        if cached:
            raise ToolUnavailable('Independent remeasurement already attempted')
        row = {'candidate_id': candidate['id'], 'parameters': values, 'endpoint_check': False, 'independent_calibration_check': True, 'measurement_status': 'error', 'status': 'unknown', 'failed_constraints': ['measurement_invalid'], 'loss': 1000000000.0, 'metrics': {}, 'errors': []}
        measured = False
        try:
            evidence = {}
            for name, deck in [('performance', physical.performance_deck(self.case, p, d)), ('matrix', physical.matrix_deck(self.case, p, d))]:
                outcome = self.engine.run_spice(candidate, name, deck)
                if outcome['status'] != 'completed':
                    raise RuntimeError('Common repeat execution ' + outcome['status'])
                evidence[name] = str(Path(outcome['directory']).relative_to(ROOT))
            performance = physical.analyze_performance(ROOT / evidence['performance'], self.case)
            linear = physical.analyze_matrix(ROOT / evidence['matrix'], self.case, p)
            metrics, status, failed, loss = physical_metrics(performance, linear, self.case)
            measured = True
            row.update(measurement_status='valid', metrics=metrics, status=status, failed_constraints=failed, loss=loss, evidence=evidence)
            save(self.out / ('remeasured_%03d.json' % candidate['id']), {'performance': performance, 'linear': linear, 'row': row})
        except (RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as exc:
            row['errors'].append({'classification': 'measurement_error', 'reason': str(exc)})
            if candidate['status'] == 'pending':
                candidate.update(status='execution_error', infrastructure_class='measurement_extraction_error')
        finally:
            self.engine.finish_candidate(candidate, measured=measured, accepted=row['status'] == 'success')
            row['cost_to_observation'] = self.costs()
            self.rows.append(row)
            save(self.out / 'trajectory.json', self.rows)
        if measured and (self.best is None or row['loss'] < self.best['loss']):
            self.best = row
        if row['status'] == 'success' and self.first is None:
            self.first = row
        if self.engine.status == 'paused':
            raise ExecutionPaused('Repeat measurement infrastructure pause')
        return row

    def calibrate(self):
        self.calibrations += 1
        eta, _ = self.characterize(self.current['parameters'], fresh=True)
        calibration = self.calibration_models[self.descriptor_key(self.current['parameters'])]
        row = self.repeat_full(self.current['parameters'])
        if row['measurement_status'] != 'valid':
            raise ToolUnavailable('Calibration full remeasurement invalid; original selection retained')
        self.current, self.plan_eta, self.plan_calibration = (row, eta, calibration)
        self.last_diagnostic = None
        diagnostic = self.diagnose()
        return {'status': 'calibrated' if diagnostic['model_trusted'] else 'still_untrusted', 'diagnostic': diagnostic, 'plan_rebased': True, 'total_error_unchanged_by_model_definition': True}

    def size_devices(self, plan, max_objectives=12):
        initial = {**self.current['parameters'], **plan['passives']}
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
                raise ToolUnavailable('Probe outside fixed registered domain')
            x[axis] = value
            result = self.accept_trial(self.evaluate(x), planned_eta=self.plan_eta, planned_model=self.plan_calibration)
        elif name in {'rc_step', 'coupled_step'}:
            plan = self.allocation(name == 'coupled_step', args['radius'])
            self.event('bounded_allocation', plan)
            if name == 'rc_step' and (not plan['rc_only_feasible']):
                raise ToolUnavailable('RC-only linearized feasibility with margin failed')
            x = self.size_devices(plan) if name == 'coupled_step' else dict(self.current['parameters'])
            x.update(plan['passives'])
            if not self.room(2, 11):
                raise ToolUnavailable('Full stage allowance exhausted; preserve endpoint')
            result = self.accept_trial(self.evaluate(x), planned_eta=OTA(**plan['ota_target']), planned_model=self.plan_calibration)
            result['allocation'] = plan
        else:
            raise ValueError('Action dispatch denied')
        self.last_result = result
        self.event(name, result)
        return result

    def finish(self):
        result = HighpassRun.finish(self)
        result.update(version=VERSION, method_id='M4', method_name='KIRA', llm_enabled=bool(self.state.get('llm_enabled')), complete_kira_implementation=True, complete_kira_empirically_validated=False, supported_types=['T2'], knowledge_version=self.kb.version, model_operating_point_qualified=bool(self.last_diagnostic and self.last_diagnostic['matched_operating_point']), qualification_blocker=(self.last_diagnostic or {}).get('qualification_blocker'))
        if result['status'] != 'success':
            result['time_to_success'] = None
        save(self.out / 'result.json', result)
        return result
