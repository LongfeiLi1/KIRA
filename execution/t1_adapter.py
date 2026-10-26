from runtime.paths import CODE_ROOT
'T1-only development adapter. Preserve native Powell and measured acceptance.'
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
import argparse, json, sys, time, shutil, hashlib
ROOT = CODE_ROOT
sys.path[:0] = [str(ROOT / 'native'), str(Path(__file__).resolve().parent)]
import numpy as np
from rc_agent import physical
from rc_agent.baseline import NAMES, task_metrics, FirstFeasible
from rc_agent.native_powell import native_powell
from common_execution import CommonExecution, ExecutionPaused, CandidateBudgetExhausted, save, value_hash
VERSION = 't1_common_powell_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def task_from_case(c, legacy):
    if (c['type_id'], c['nominal_filter_order'], c['filter_sections'], c['OTA_count'], c['outer_R_count'], c['outer_C_count']) != ('T1', 2, 1, 1, 2, 2):
        raise ValueError('Only verified single-OTA second-order T1 supported')
    expected = {'target_polarity': 1, 'gain_reference_kind': 'first_ac_sample_magnitude_DC_proxy', 'gain_reference_frequency_hz': 1.0, 'passband_low_hz': 1.0, 'stopband_high_hz': 100000000.0}
    if any((c[k] != v for k, v in expected.items())):
        raise ValueError('Unsupported T1 acceptance semantics')
    common = dict(legacy['common_acceptance'])
    common['dc_gain_range'] = [c['gain_magnitude_min'], c['gain_magnitude_max']]
    task = {'id': c['case_id'], 'f0_hz': c['f0_hz'], 'q': c['q_target'], 'f0_relative_tolerance': c['f0_relative_tolerance'], 'q_relative_tolerance': c['q_relative_tolerance'], 'load_f': c['load_cap_pF'] * 1e-12, 'power_limit_w': c['power_max_uW'] * 1e-06, 'passband_edge_hz': c['passband_high_hz'], 'stopband_edge_hz': c['stopband_low_hz'], 'passband_max_loss_db': c['passband_droop_max_db'], 'stopband_min_attenuation_db': c['stopband_attenuation_min_db'], 'phase_margin_min_deg': legacy['tasks'][0]['phase_margin_min_deg']}
    if any((t['phase_margin_min_deg'] != task['phase_margin_min_deg'] for t in legacy['tasks'])):
        raise ValueError('Inherited phase margin is ambiguous')
    return (task, common)

@contextmanager
def measured_runner(engine, candidate):
    original = physical.run_deck

    def run(deck, directory, budget, tag):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(engine.directory):
            raise ExecutionPaused('Measurement output escaped this run')
        outcome = engine.run_spice(candidate, tag, deck)
        directory.mkdir(parents=True, exist_ok=False)
        save(directory / 'common_attempt.json', outcome)
        if outcome['status'] != 'completed':
            raise RuntimeError('Common execution ' + outcome['status'] + ': ' + str(outcome.get('failure_class')))
        source = Path(outcome['directory'])
        for p in source.iterdir():
            if p.is_file():
                shutil.copy2(p, directory / p.name)
        return outcome['log']
    physical.run_deck = run
    try:
        yield
    finally:
        physical.run_deck = original

class T1Run:

    def __init__(self, case, method, out, protocol, ngspice, legacy=None):
        if method not in ['powell_g0', 'powell_g1', 'autosizer']:
            raise ValueError('Unsupported method')
        self.case = case
        self.method = method
        self.legacy = legacy or read(ROOT / 'resources/configs/sallen_key.json')
        self.task, self.common = task_from_case(case, self.legacy)
        self.engine = CommonExecution(out, protocol, case['level'], ngspice, launch_authorized=True)
        self.out = self.engine.directory
        self.rows = []
        self.best = None
        self.first = None
        self.endpoint = None
        self.started = time.monotonic()
        self.state = {}
        self.stop = 'not_started'
        save(self.out / 'input.json', {'case': case, 'method': method, 'task': self.task, 'common_acceptance': self.common, 'domain': self.legacy['physical_bounds'], 'initial': self.legacy['initial'], 'protocol': protocol, 'version': VERSION, 'classification': 'development; real feasibility unverified; not unseen evaluation'})

    def costs(self):
        e = self.engine
        return {'new_candidates': len(e.cache), 'spice_processes': sum((x['actual_started'] for x in e.attempts)), 'launch_attempts': len(e.attempts), 'analysis_commands': sum((x['scheduled_analyses'] for x in e.attempts)), 'simulator_seconds': sum((x['wall_seconds'] for x in e.attempts)), 'total_wall_seconds': time.monotonic() - self.started, 'llm_calls': len(e.api_calls)}

    def evaluate(self, x, endpoint=False):
        for n in NAMES:
            lo, hi = self.legacy['physical_bounds'][n]
            if not np.isfinite(x[n]) or not lo * (1 - 1e-14) <= x[n] <= hi * (1 + 1e-14):
                raise ValueError('Candidate outside unchanged domain: ' + n)
        c, cached = self.engine.candidate({n: float(x[n]) for n in NAMES}, endpoint=endpoint)
        if cached:
            return next((r for r in self.rows if r['candidate_id'] == c['id']))
        p = physical.Passives(*(x[n] for n in NAMES[:4]))
        d = physical.Devices(*(x[n] for n in NAMES[4:]), self.legacy['initial']['rz_ohm'])
        folder = self.out / ('measured_%03d' % c['id'])
        row = {'candidate_id': c['id'], 'parameters': dict(x), 'endpoint_check': endpoint, 'status': 'unknown', 'failed_constraints': ['measurement_invalid'], 'loss': 1000000000.0, 'metrics': {}}
        measured = False
        try:
            with measured_runner(self.engine, c):
                f, h, m = physical.measure_filter(p, d, self.task['f0_hz'], folder, None, load_f=self.task['load_f'])
            metrics, status, failed, loss = task_metrics(f, h, m['metrics'], self.task, self.common)
            measured = len(c['slot_ids']) == 2 and all((self.engine.slots[i]['status'] == 'completed' for i in c['slot_ids']))
            row.update(metrics=metrics, status=status, failed_constraints=failed, loss=loss, evidence=m['evidence'])
        except (RuntimeError, ValueError) as exc:
            row['error'] = str(exc)
        self.engine.finish_candidate(c, measured=measured, accepted=row['status'] == 'success', endpoint_evidence=str(folder.relative_to(ROOT)) if measured else None)
        row['cost_to_observation'] = self.costs()
        self.rows.append(row)
        save(self.out / 'trajectory.json', self.rows)
        print(json.dumps({'case': self.case['case_id'], 'method': self.method, 'candidate': c['id'], 'endpoint': endpoint, 'status': row['status'], 'failed': row['failed_constraints'], 'cost': self.costs()}), flush=True)
        if endpoint:
            self.endpoint = row
        else:
            if self.best is None or row['loss'] < self.best['loss']:
                self.best = row
            if row['status'] == 'success' and self.first is None:
                self.first = row
        if self.engine.status == 'paused':
            raise ExecutionPaused('Infrastructure policy paused run; see budget.json')
        return row

    def finish(self):
        selected = self.first or self.best
        if selected and self.engine.status != 'paused' and (self.endpoint is None):
            try:
                self.evaluate(selected['parameters'], True)
            except (ExecutionPaused, RuntimeError, ValueError) as exc:
                self.stop += '; endpoint: ' + str(exc)
        status = 'paused' if self.engine.status == 'paused' else 'endpoint_unconfirmed' if not self.endpoint or not self.engine.candidates[self.engine.endpoint].get('measured') else 'success' if self.endpoint['status'] == 'success' else 'candidate_budget_exhausted' if len(self.engine.cache) >= self.engine.limits['max_new_candidates'] else 'search_stopped_without_feasible_endpoint'
        checks = {} if not self.endpoint else {'all_physical_gates': self.endpoint['status'] == 'success', 'complete_mna': self.endpoint.get('metrics', {}).get('stable') is True}
        result = {'schema_version': 1, 'version': VERSION, 'case_id': self.case['case_id'], 'type_id': 'T1', 'level': self.case['level'], 'method_id': self.method, 'status': status, 'reason': self.stop, 'failed_constraints': self.endpoint.get('failed_constraints', []) if self.endpoint else ['independent_endpoint_unconfirmed'], 'metrics': self.endpoint.get('metrics', {}) if self.endpoint else {}, 'cost': self.costs(), 'independent_endpoint_checked': bool(self.endpoint and self.engine.candidates[self.engine.endpoint].get('measured')), 'acceptance_checks': checks, 'endpoint': self.endpoint, 'time_to_success': self.costs()['total_wall_seconds'] if status == 'success' else None, 'native_optimizer': self.state, 'case_sha256': value_hash(self.case), 'evaluation_protocol_sha256': self.engine.limits['protocol_sha256'], 'development_only': True, 'unknown_is_pass': False}
        save(self.out / 'result.json', result)
        return result

    def powell(self, smoke=False):
        initial = dict(self.legacy['initial'])
        active = list(NAMES[:4] if self.method == 'powell_g0' else NAMES)
        z0 = np.log([initial[n] for n in active])
        bounds = np.log([self.legacy['physical_bounds'][n] for n in active])
        objectives = []

        def objective(z):
            x = dict(initial)
            x.update({n: float(v) for n, v in zip(active, np.exp(z))})
            row = self.evaluate(x)
            objectives.append({'candidate_id': row['candidate_id'], 'parameters': x})
            save(self.out / 'objectives.json', objectives)
            if row['status'] == 'success':
                raise FirstFeasible('first full acceptance; independent endpoint required')
            return row['loss']
        try:
            if smoke:
                self.evaluate(initial)
                self.stop = 'bounded integration smoke: one initial candidate plus independent endpoint'
            else:
                result = native_powell(objective, z0, bounds, self.legacy['optimizer'], self.state, lambda state, z: save(self.out / 'optimizer_progress.json', state))
                self.stop = str(result.message)
        except (FirstFeasible, CandidateBudgetExhausted, ExecutionPaused) as exc:
            self.stop = str(exc)
        except (RuntimeError, ValueError) as exc:
            self.stop = 'execution_error: ' + str(exc)
            self.engine.status = 'paused'
            self.engine.save()
        return self.finish()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-file', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--method', choices=['powell_g0', 'powell_g1'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ngspice', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if not out.is_relative_to(ROOT / 'output') and (not out.is_relative_to(ROOT / 'receipts')):
        raise ValueError('Output outside authorized run roots')
    r = T1Run(read(a.case_file), a.method, out, read(a.protocol), a.ngspice).powell(a.smoke)
    print('RESULT ' + json.dumps({k: r[k] for k in ['case_id', 'method_id', 'status', 'cost']}), flush=True)
if __name__ == '__main__':
    main()
