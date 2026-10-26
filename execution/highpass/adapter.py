from runtime.paths import BENCHMARK_DIR
'T2 parameter binding, physical metrics and native M0 common execution.\n\nUses immutable physical deck builders/parsers without monkey-patching globals.\nM1 and M2 may consume HighpassRun.evaluate; their method readiness is separate.\n'
from pathlib import Path
from dataclasses import asdict
import json, math, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'native'), str(BENCHMARK_DIR)]
from numerics.physical import runner as physical
from rc_agent.native_powell import native_powell
from .execution import CommonExecution, ExecutionPaused, CandidateBudgetExhausted, save, value_hash
VERSION = 'typed_runtime_20261010'
DEVICE_NAMES = ('current_scale', 'output_scale', 'cc_pf')
PASSIVE_NAMES = {'T2': ('c1', 'c2', 'rf', 'rg')}

def read(path):
    return json.loads(Path(path).read_text())

def binding(case):
    authoritative = physical.data.load_case(case['case_id'])
    if case != authoritative:
        raise ValueError('Case differs from immutable manifest')
    if case['type_id'] not in PASSIVE_NAMES:
        raise ValueError('T2 isolated binding only; T1 retains its original entry')
    report = physical.data.binding_report(case)
    names = PASSIVE_NAMES[case['type_id']] + DEVICE_NAMES
    from initialization import synthesize
    receipt = synthesize(dict(task_id=case['case_id'], topology={'T2': 't2_sk_hp'}[case['type_id']], frequency_hz=case['f0_hz'], q=case['q_target'], gain=case['target_polarity']), report['actual_registered_search_domain'])
    initial = receipt['parameters']
    return {'version': VERSION, 'case_id': case['case_id'], 'names': list(names), 'physical_bounds': report['actual_registered_search_domain'], 'initial': initial, 'initialization_receipt': receipt, 'fixed': {'rz_ohm': physical.Devices().rz_ohm}, 'equality_ties': [], 'coordinate_policy': 'all named passives and three OTA knobs independent; log SI except cc_pf in pF', 'initial_policy': 'unified target-specification analytic synthesis plus explicit template; no per-case seed lookup', 'freeze_status': 'runtime binding frozen by this development version under current task authorization', 'historical_status': 'first Type registration remained proposal_pending_type_level_review; no retroactive freeze claim', 'manifest_runnable': False, 'development_only': True, 'domain_evidence': 'numerics/physical/domain_provenance.json', 'topology_sha256': report['normalized_topology_sha256']}

def unpack(x, bound, case):
    if set(x) != set(bound['names']):
        raise ValueError('Exact topology-specific coordinate set required')
    values = {}
    for name in bound['names']:
        value = x[name]
        lo, hi = bound['physical_bounds'][name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or (not math.isfinite(value)) or (not lo <= value <= hi):
            raise ValueError('Outside unchanged registered domain: ' + name)
        values[name] = float(value)
    passives = {n: values[n] for n in PASSIVE_NAMES[case['type_id']]}
    devices = physical.Devices(**{n: values[n] for n in DEVICE_NAMES}, **bound['fixed'])
    physical.validate_candidate(case, passives, devices)
    return (values, passives, devices)

def physical_metrics(performance, linear, case):
    """Exact original gates; continuous objective guides search, never acceptance."""
    if abs(linear['op_dc_out_v'] - performance['operating_point']['dc_out_v']) > 1e-07:
        raise ValueError('Performance and closed-MNA operating points differ')
    if linear['status'] not in ('stable', 'unstable'):
        raise ValueError('Unresolved complete local mode classification')
    failed = list(performance['failed_constraints'])
    if linear['status'] == 'unstable':
        failed.append('full_local_stability')
    m = {**performance['response'], **performance['operating_point'], **performance['step'], **performance['port'], 'stable': linear['status'] == 'stable', 'full_local_stability': linear['status'], 'dc_out': performance['operating_point']['dc_out_v']}
    loss = (math.log(m['f0_hz'] / case['f0_hz']) / case['f0_relative_tolerance']) ** 2
    loss += (math.log(m['q'] / case['q_target']) / case['q_relative_tolerance']) ** 2
    loss += 100 * max(0.0, m['power_w'] / (case['power_max_uW'] * 1e-06) - 1) ** 2
    loss += max(0.0, 60 - m['phase_margin_deg']) ** 2
    loss += max(0.0, m['passband_max_loss_db'] - case['passband_droop_max_db']) ** 2
    loss += max(0.0, case['stopband_attenuation_min_db'] - m['stopband_min_attenuation_db']) ** 2
    smooth = {'f0_hz', 'q', 'power', 'phase_margin', 'passband_max_loss_db', 'stopband_min_attenuation_db', 'full_local_stability'}
    loss += 1000 * sum((k not in smooth for k in failed)) + (0 if m['stable'] else 10000)
    if not math.isfinite(loss):
        raise ValueError('Nonfinite physical objective')
    return (m, 'success' if not failed else 'circuit_fail', failed, float(loss))

class FirstFeasible(Exception):
    pass

class HighpassRun:

    def __init__(self, case, method, out, protocol, ngspice, *, executor=None, launch_authorized=False):
        if method not in ('M0', 'M1', 'M2'):
            raise ValueError('Unknown baseline method')
        self.case, self.method = (case, method)
        self.binding = binding(case)
        self.names = self.binding['names']
        self.engine = CommonExecution(out, protocol, case['level'], ngspice, executor=executor, launch_authorized=launch_authorized)
        self.out = self.engine.directory
        self.rows, self.best, self.first, self.endpoint = ([], None, None, None)
        self.state, self.stop, self.started = ({}, 'not_started', time.monotonic())
        self.settings = read(ROOT / 'resources/configs/sallen_key.json')['optimizer']
        save(self.out / 'input.json', {'version': VERSION, 'case': case, 'method': method, 'binding': self.binding, 'protocol': protocol, 'optimizer': self.settings, 'physical_source_sha256': physical.sha(Path(physical.__file__))})

    def costs(self):
        e = self.engine
        return {'new_candidates': len(e.cache), 'spice_processes': sum((x['actual_started'] for x in e.attempts)), 'launch_attempts': len(e.attempts), 'analysis_commands': sum((x['scheduled_analyses'] for x in e.attempts)), 'simulator_seconds': sum((x['wall_seconds'] for x in e.attempts)), 'total_wall_seconds': time.monotonic() - self.started, 'llm_calls': len(e.api_calls), 'llm_usage': {k: sum((r.get(k) or 0 for r in e.api_calls)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}, 'monetary_cost': 0 if not e.api_calls else None, 'monetary_cost_reason': 'No API requests' if not e.api_calls else 'Provider billing unavailable; token ledger retained'}

    def evaluate(self, x, endpoint=False):
        values, p, d = unpack(x, self.binding, self.case)
        candidate, cached = self.engine.candidate(values, endpoint=endpoint)
        if cached:
            return next((r for r in self.rows if r['candidate_id'] == candidate['id']))
        folder = self.out / ('measured_%03d' % candidate['id'])
        folder.mkdir(exist_ok=False)
        row = {'candidate_id': candidate['id'], 'parameters': values, 'endpoint_check': endpoint, 'measurement_status': 'error', 'status': 'unknown', 'failed_constraints': ['measurement_invalid'], 'loss': 1000000000.0, 'metrics': {}, 'errors': []}
        measured = False
        try:
            evidence = {}
            for name, deck in [('performance', physical.performance_deck(self.case, p, d)), ('matrix', physical.matrix_deck(self.case, p, d))]:
                outcome = self.engine.run_spice(candidate, name, deck)
                save(folder / (name + '_attempt.json'), outcome)
                if outcome['status'] != 'completed':
                    raise RuntimeError('Common execution ' + outcome['status'])
                evidence[name] = str(Path(outcome['directory']).relative_to(ROOT))
            performance = physical.analyze_performance(ROOT / evidence['performance'], self.case)
            linear = physical.analyze_matrix(ROOT / evidence['matrix'], self.case, p)
            metrics, status, failed, loss = physical_metrics(performance, linear, self.case)
            measured = True
            row.update(measurement_status='valid', metrics=metrics, status=status, failed_constraints=failed, loss=loss, evidence=evidence)
            save(folder / 'physical.json', {'performance': performance, 'linear': linear, 'evidence': evidence})
        except (RuntimeError, ValueError, OSError, np.linalg.LinAlgError) as exc:
            row['errors'].append({'classification': 'measurement_error', 'reason': str(exc)})
            if candidate['status'] == 'pending':
                candidate['status'] = 'execution_error'
                candidate['infrastructure_class'] = 'measurement_extraction_error'
        self.engine.finish_candidate(candidate, measured=measured, accepted=row['status'] == 'success', endpoint_evidence=str(folder.relative_to(ROOT)) if measured else None)
        row['fom'] = 1 / (1 + row['loss'])
        row['cost_to_observation'] = self.costs()
        self.rows.append(row)
        save(folder / 'result.json', row)
        save(self.out / 'trajectory.json', self.rows)
        if endpoint:
            self.endpoint = row
        else:
            if measured and (self.best is None or row['loss'] < self.best['loss']):
                self.best = row
            if row['status'] == 'success' and self.first is None:
                self.first = row
        print(json.dumps({'case_id': self.case['case_id'], 'method_id': self.method, 'candidate': candidate['id'], 'endpoint': endpoint, 'status': row['status'], 'cost': self.costs()}), flush=True)
        if self.engine.status == 'paused':
            raise ExecutionPaused('Common infrastructure policy paused this case')
        return row

    def finish(self):
        selected = self.first or self.best
        if selected and self.engine.status != 'paused' and (self.endpoint is None):
            try:
                self.evaluate(selected['parameters'], endpoint=True)
            except (RuntimeError, ValueError, OSError) as exc:
                self.stop += '; endpoint: ' + str(exc)
        checked = bool(self.endpoint and self.engine.candidates[self.engine.endpoint].get('measured'))
        status = 'paused' if self.engine.status == 'paused' else 'endpoint_unconfirmed' if not checked else 'success' if self.endpoint['status'] == 'success' else 'candidate_budget_exhausted' if len(self.engine.cache) >= self.engine.limits['max_new_candidates'] else 'search_stopped_without_feasible_endpoint'
        metrics = self.endpoint['metrics'] if self.endpoint else {}
        result = {'schema_version': 2, 'version': VERSION, 'case_id': self.case['case_id'], 'type_id': self.case['type_id'], 'level': self.case['level'], 'method_id': self.method, 'suite_method_id': self.method, 'status': status, 'reason': self.stop, 'cost': self.costs(), 'endpoint': self.endpoint, 'metrics': metrics, 'failed_constraints': self.endpoint['failed_constraints'] if self.endpoint else ['independent_endpoint_unconfirmed'], 'independent_endpoint_checked': checked, 'acceptance_checks': {'all_physical_gates': status == 'success', 'complete_mna': metrics.get('stable') is True}, 'time_to_success': self.costs()['total_wall_seconds'] if status == 'success' else None, 'native_optimizer': self.state, 'case_sha256': value_hash(self.case), 'evaluation_protocol_sha256': self.engine.limits['protocol_sha256'], 'binding_sha256': value_hash(self.binding), 'development_only': True, 'unknown_is_pass': False}
        save(self.out / 'result.json', result)
        return result

    def powell(self, smoke=False):
        if self.method != 'M0':
            raise ValueError('Native Powell entry belongs only to M0')
        initial = self.binding['initial']
        bounds = np.asarray([self.binding['physical_bounds'][n] for n in self.names])

        def objective(z):
            x = np.clip(np.exp(z), bounds[:, 0], bounds[:, 1])
            row = self.evaluate(dict(zip(self.names, map(float, x))))
            if row['status'] == 'success':
                raise FirstFeasible('First complete acceptance; independent endpoint required')
            return row['loss']
        try:
            if smoke:
                self.evaluate(initial)
                self.stop = 'bounded qualification: one fixed initial candidate plus independent endpoint'
            else:
                opt = native_powell(objective, np.log([initial[n] for n in self.names]), np.log(bounds), self.settings, self.state, lambda state, z: save(self.out / 'optimizer_progress.json', state))
                self.stop = str(opt.message)
        except (FirstFeasible, CandidateBudgetExhausted, ExecutionPaused) as exc:
            self.stop = str(exc)
        except (RuntimeError, ValueError, OSError) as exc:
            self.stop = 'implementation_error: ' + str(exc)
            self.engine.status = 'paused'
            self.engine.save()
        return self.finish()
