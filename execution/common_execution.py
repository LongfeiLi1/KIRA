"""Shared candidate/process accounting and failure policy for ALL future methods.
No optimizer, topology, model API, automatic resume, or experiment launch here.
"""
from pathlib import Path
import errno, hashlib, json, math, re, subprocess, time

class ExecutionPaused(RuntimeError):
    pass

class CandidateBudgetExhausted(RuntimeError):
    pass

def value_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def save(path, x):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def text(x):
    return x.decode(errors='replace') if isinstance(x, bytes) else x or ''

def protocol_budget(protocol, level):
    if protocol.get('status') != 'frozen':
        raise ExecutionPaused('Evaluation protocol is not frozen')
    n = protocol['candidate_budget_by_level'].get(level)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ExecutionPaused('Level candidate budget awaits explicit confirmation')
    attempts = protocol['max_attempts_per_process']
    if attempts != 3:
        raise ExecutionPaused('Reviewed implementation expects first plus two retries; revise protocol/tests for another value')
    timeout = protocol['spice_timeout_seconds']
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ExecutionPaused('Finite positive protective timeout required')
    if protocol.get('timeout_calibration', {}).get('status') != 'qualified':
        raise ExecutionPaused('Protective timeout calibration is not frozen')
    slots = protocol['base_process_slots_per_candidate']
    analyses = protocol['base_analysis_allowance_per_candidate']
    if slots != 2 or analyses != 10:
        raise ExecutionPaused('Reviewed full-filter allocation is two process slots and ten analyses; auxiliary tools must use this shared pool')
    return {'protocol_version': protocol['version'], 'protocol_sha256': value_hash(protocol), 'max_new_candidates': n, 'independent_endpoint_candidates': 1, 'max_base_process_slots': slots * (n + 1), 'search_base_process_slots': slots * n, 'max_spice_attempts': attempts * slots * (n + 1), 'max_scheduled_analyses': attempts * analyses * (n + 1), 'search_base_analysis_allowance': analyses * n, 'max_attempts_per_process': attempts, 'spice_timeout_seconds': timeout, 'max_case_seconds': None, 'consecutive_infrastructure_candidates_before_pause': 3}

class CommonExecution:

    def __init__(self, directory, protocol, level, ngspice, *, executor=None, launch_authorized=False, clock=time.monotonic):
        self.limits = protocol_budget(protocol, level)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.protocol = protocol
        self.ngspice = str(ngspice)
        self.executor = executor
        self.launch_authorized = launch_authorized
        self.clock = clock
        self.started = clock()
        self.candidates = []
        self.attempts = []
        self.slots = []
        self.cache = {}
        self.endpoint = None
        self.status = 'ready'
        self.streak = 0
        self.last_infra = None
        self.api_calls = []
        self.level = level
        self.save()

    def save(self):
        d = {'schema_version': 1, 'common_runner': 'candidate_retry_v1', 'protocol_version': self.protocol['version'], 'protocol_sha256': self.limits['protocol_sha256'], 'level': self.level, 'limits': self.limits, 'status': self.status, 'new_unique_candidate_attempts': len(self.cache), 'independent_endpoint_started': self.endpoint is not None, 'attempted_calls': len(self.attempts), 'actual_started_spice_processes': sum((x['actual_started'] for x in self.attempts)), 'attempted_explicit_analysis_commands': sum((x['scheduled_analyses'] for x in self.attempts)), 'process_elapsed_seconds': sum((x['wall_seconds'] for x in self.attempts)), 'total_elapsed_seconds': self.clock() - self.started, 'consecutive_infrastructure_candidate_count': self.streak, 'last_infrastructure_class': self.last_infra, 'candidates': self.candidates, 'slots': self.slots, 'attempts': self.attempts, 'api_calls': self.api_calls, 'automatic_api_retries': 0, 'competitive_case_wall_limit': None}
        save(self.directory / 'budget.json', d)

    def candidate(self, parameters, *, endpoint=False):
        if self.status == 'paused':
            raise ExecutionPaused('Group paused; no automatic retry or budget reset')
        key = value_hash(parameters)
        if not endpoint and key in self.cache:
            return (self.candidates[self.cache[key]], True)
        if endpoint:
            if self.endpoint is not None:
                raise ExecutionPaused('Independent endpoint already attempted')
        elif len(self.cache) >= self.limits['max_new_candidates']:
            self.status = 'candidate_budget_exhausted'
            self.save()
            raise CandidateBudgetExhausted('Unique candidate cap reached')
        c = {'id': len(self.candidates), 'parameters': parameters, 'parameter_sha256': key, 'role': 'endpoint' if endpoint else 'search', 'status': 'pending', 'slot_ids': [], 'infrastructure_class': None}
        self.candidates.append(c)
        if endpoint:
            self.endpoint = c['id']
        else:
            self.cache[key] = c['id']
        self.save()
        return (c, False)

    def _pause(self, reason):
        self.status = 'paused'
        self.save()
        raise ExecutionPaused(reason)

    def run_spice(self, candidate, slot_name, deck):
        if self.status == 'paused':
            raise ExecutionPaused('Paused group; no more process attempts')
        if self.executor is None and (not self.launch_authorized):
            self._pause('No explicit launch authorization')
        if candidate['status'] not in ['pending']:
            raise ExecutionPaused('Candidate is finalized; retries cannot create a new budget')
        if any((x['candidate_id'] == candidate['id'] and x['name'] == slot_name for x in self.slots)):
            raise ExecutionPaused('Logical process slot already used; no duplicate execution')
        count = len(re.findall('(?im)^\\s*(op|ac|tran|pz|tf)\\b', '\n'.join(deck.splitlines()[1:])))
        if count < 1:
            raise ExecutionPaused('No explicit analysis command in deck')
        search = [x for x in self.slots if not x['endpoint']]
        if candidate['role'] == 'search' and (len(search) >= self.limits['search_base_process_slots'] or sum((x['analysis_count'] for x in search)) + count > self.limits['search_base_analysis_allowance']):
            self._pause('Shared base process/analysis pool exhausted; preserve endpoint reserve')
        if len(self.slots) >= self.limits['max_base_process_slots']:
            self._pause('Common process allowance exhausted')
        slot = {'id': len(self.slots), 'candidate_id': candidate['id'], 'name': slot_name, 'endpoint': candidate['role'] == 'endpoint', 'analysis_count': count, 'deck_sha256': hashlib.sha256(deck.encode()).hexdigest(), 'attempt_ids': [], 'status': 'pending'}
        self.slots.append(slot)
        candidate['slot_ids'].append(slot['id'])
        self.save()
        for number in range(1, self.limits['max_attempts_per_process'] + 1):
            if len(self.attempts) >= self.limits['max_spice_attempts'] or sum((x['scheduled_analyses'] for x in self.attempts)) + count > self.limits['max_scheduled_analyses']:
                self._pause('Total attempts/analyses cap reached')
            d = self.directory / ('candidate_%03d' % candidate['id']) / ('slot_%02d' % slot['id']) / ('attempt_%02d' % number)
            d.mkdir(parents=True, exist_ok=False)
            (d / 'circuit.cir').write_text(deck)
            row = {'attempt': len(self.attempts) + 1, 'candidate_id': candidate['id'], 'parameter_sha256': candidate['parameter_sha256'], 'slot_id': slot['id'], 'slot_attempt': number, 'deck_sha256': slot['deck_sha256'], 'evidence': str(d.relative_to(self.directory)), 'status': 'started', 'actual_started': False, 'scheduled_analyses': count, 'wall_seconds': 0.0}
            self.attempts.append(row)
            slot['attempt_ids'].append(row['attempt'])
            self.save()
            begin = self.clock()
            retry = False
            fatal = False
            try:
                if self.executor is None:
                    result = subprocess.run([self.ngspice, '-n', '-b', 'circuit.cir'], cwd=d, capture_output=True, text=True, timeout=self.limits['spice_timeout_seconds'])
                else:
                    result = self.executor([self.ngspice, '-n', '-b', 'circuit.cir'], cwd=d, timeout=self.limits['spice_timeout_seconds'])
                row['actual_started'] = True
                log = text(result.stdout) + '\n' + text(result.stderr)
                (d / 'ngspice.log').write_text(log)
                row['returncode'] = result.returncode
                bad = result.returncode != 0 or re.search('(?im)^\\s*(error|fatal)\\b', log)
                if not bad:
                    row.update(status='completed')
                    slot['status'] = 'completed'
                elif re.search('syntax|unknown (subckt|parameter|device)|no such file|cannot open|could not find|command not found', log, re.I):
                    row.update(status='deterministic_error', failure_class='syntax_or_missing_dependency')
                    fatal = True
                elif re.search('convergence|timestep too small|singular matrix', log, re.I):
                    row.update(status='candidate_invalid', failure_class='electrical_nonconvergence')
                else:
                    row.update(status='execution_error', failure_class='nonzero_or_invalid_execution')
            except subprocess.TimeoutExpired as e:
                row.update(status='timeout', actual_started=True, failure_class='timeout')
                (d / 'ngspice.partial.log').write_text(text(e.stdout) + '\n' + text(e.stderr))
                retry = True
            except OSError as e:
                transient = e.errno in {errno.EAGAIN, errno.EINTR, errno.ETXTBSY}
                row.update(status='transient_launch_error' if transient else 'deterministic_error', failure_class='transient_launch_error' if transient else 'missing_executable_or_dependency', errno=e.errno, actual_started=False)
                retry = transient
                fatal = not transient
            finally:
                row['wall_seconds'] = self.clock() - begin
                self.save()
            if row['status'] == 'completed':
                return {'status': 'completed', 'directory': str(d), 'attempts': number, 'log': (d / 'ngspice.log').read_text()}
            if retry and number < self.limits['max_attempts_per_process']:
                continue
            slot['status'] = row['status']
            candidate['infrastructure_class'] = None if row['status'] == 'candidate_invalid' else row['failure_class']
            candidate['status'] = 'endpoint_unconfirmed' if candidate['role'] == 'endpoint' else 'candidate_invalid' if row['status'] == 'candidate_invalid' else 'execution_error'
            self.save()
            if candidate['role'] == 'endpoint':
                self.status = 'endpoint_unconfirmed'
                self.save()
            elif fatal:
                self.status = 'paused'
                self.save()
            return {'status': candidate['status'], 'failure_class': row.get('failure_class'), 'attempts': number, 'retry_exhausted': retry, 'fatal': fatal}

    def finish_candidate(self, candidate, *, measured, accepted=False, endpoint_evidence=None):
        if candidate.get('finalized'):
            raise ExecutionPaused('Candidate already finalized')
        if measured and (not candidate['slot_ids'] or any((self.slots[i]['status'] != 'completed' for i in candidate['slot_ids']))):
            raise ExecutionPaused('No complete measured process evidence')
        if candidate['role'] == 'endpoint' and measured and (len(candidate['slot_ids']) < 2 or sum((self.slots[i]['analysis_count'] for i in candidate['slot_ids'])) < 10):
            raise ExecutionPaused('Incomplete independent full-filter endpoint')
        candidate['finalized'] = True
        candidate['measured'] = bool(measured)
        candidate['accepted'] = bool(accepted)
        if measured:
            if any((self.slots[i]['status'] != 'completed' for i in candidate['slot_ids'])):
                raise ExecutionPaused('Cannot certify measurements from a failed process')
            candidate['status'] = 'measured'
            self.last_infra = None
            self.streak = 0
        elif candidate.get('infrastructure_class'):
            cls = candidate['infrastructure_class']
            self.streak = self.streak + 1 if cls == self.last_infra else 1
            self.last_infra = cls
            if self.streak >= 3:
                self.status = 'paused'
        else:
            self.last_infra = None
            self.streak = 0
        if candidate['role'] == 'endpoint':
            candidate['endpoint_evidence'] = endpoint_evidence
            if measured and accepted and endpoint_evidence:
                self.status = 'success'
            elif not measured:
                self.status = 'endpoint_unconfirmed'
            else:
                self.status = 'candidate_budget_exhausted' if len(self.cache) >= self.limits['max_new_candidates'] else 'search_stopped_without_feasible_endpoint'
        self.save()
        return {'status': self.status, 'candidate_status': candidate['status'], 'new_candidates': len(self.cache), 'actual_spice_processes': sum((x['actual_started'] for x in self.attempts))}

    def reserve_api(self, model, max_completion_tokens):
        if model != 'qwen3.8-flash':
            raise ExecutionPaused('All LLM calls must use the approved model')
        cap = self.protocol.get('max_llm_calls_per_case')
        tokens = self.protocol.get('max_completion_tokens_per_call')
        if not isinstance(cap, int) or cap < 1 or len(self.api_calls) >= cap:
            raise ExecutionPaused('LLM call budget absent/exhausted before request')
        if not isinstance(tokens, int) or not isinstance(max_completion_tokens, int) or (not 0 < max_completion_tokens <= tokens):
            raise ExecutionPaused('Explicit frozen completion-token bound required')
        row = {'attempt': len(self.api_calls) + 1, 'model': model, 'max_completion_tokens': max_completion_tokens, 'status': 'started'}
        self.api_calls.append(row)
        self.save()
        return row

    def finish_api(self, row, record):
        if not any((row is x for x in self.api_calls)) or row['status'] != 'started':
            raise ExecutionPaused('Unknown or already finalized API reservation')
        row.update({k: record.get(k) for k in ['status', 'returned_model', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'wall_seconds', 'error_class']})
        self.save()
        if record.get('returned_model') not in [None, 'qwen3.8-flash']:
            self._pause('Provider returned a different model; no silent substitution')
