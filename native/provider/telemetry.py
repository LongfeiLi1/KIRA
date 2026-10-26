"""Nested wall-clock traces. No prompts, response bodies, headers or exceptions."""
import contextlib, json, time, types
from datetime import datetime, timezone
from pathlib import Path
from evidence import save
from evidence import clean
from provider.stream import injected

class Trace:

    def __init__(self, directory, *, enabled=True, clock=time.monotonic_ns):
        self.directory = Path(directory)
        self.enabled = enabled
        self.clock = clock
        self.stack = []
        self.spans = []
        self.points = []
        self.serial = 0
        self.buffer = []

    def write(self, event):
        if not self.enabled:
            return
        event = {**event, 'utc': datetime.now(timezone.utc).isoformat()}
        if not self.directory.is_dir():
            self.buffer.append(event)
            return
        with (self.directory / 'timing_events.jsonl').open('a') as f:
            for item in self.buffer + [event]:
                f.write(json.dumps(clean(item), allow_nan=False) + '\n')
        self.buffer.clear()

    def point(self, name, **tags):
        event = {'event': 'point', 'name': name, 'monotonic_ns': self.clock(), 'parent_id': self.stack[-1]['id'] if self.stack else None, 'tags': tags}
        self.points.append(event)
        self.write(event)
        return event

    def begin(self, name, **tags):
        self.serial += 1
        s = {'id': self.serial, 'parent_id': self.stack[-1]['id'] if self.stack else None, 'name': name, 'start_monotonic_ns': self.clock(), 'child_ns': 0, 'tags': tags}
        self.stack.append(s)
        self.write({'event': 'begin', **s})
        return s

    def end(self, s, exc=None):
        if not self.stack or self.stack[-1] is not s:
            raise ValueError('timing_span_nesting_invalid')
        end = self.clock()
        self.stack.pop()
        elapsed = end - s['start_monotonic_ns']
        if elapsed < 0 or s['child_ns'] > elapsed:
            raise ValueError('timing_clock_or_nesting_invalid')
        if self.stack:
            self.stack[-1]['child_ns'] += elapsed
        row = {**s, 'end_monotonic_ns': end, 'inclusive_wall_seconds': elapsed / 1000000000.0, 'exclusive_wall_seconds': (elapsed - s['child_ns']) / 1000000000.0, 'status': 'error' if exc is not None else 'completed', 'error_class': type(exc).__name__ if exc is not None else None}
        self.spans.append(row)
        self.write({'event': 'end', **row})
        return row

    @contextlib.contextmanager
    def span(self, name, **tags):
        s = self.begin(name, **tags)
        try:
            yield s
        except BaseException as exc:
            self.end(s, exc)
            raise
        else:
            self.end(s)

    def call(self, name, function, *a, **k):
        with self.span(name):
            return function(*a, **k)

    def summarize(self, queue):
        if self.stack:
            raise ValueError('timing_spans_still_open')
        totals = {}
        for s in self.spans:
            row = totals.setdefault(s['name'], {'count': 0, 'inclusive_wall_seconds': 0, 'exclusive_wall_seconds': 0})
            row['count'] += 1
            for k in ('inclusive_wall_seconds', 'exclusive_wall_seconds'):
                row[k] += s[k]
        save(self.directory / 'timing_summary.json', {'schema_version': 1, 'logging_enabled': self.enabled, 'stages': totals, 'queue': queue, 'spans': self.spans, 'exclusive_is_wall_remainder_not_CPU_time': True, 'do_not_sum_nested_inclusive_times': True, 'missing_measurements': {'SPICE_op_ac_tran_subcommand_wall': None, 'provider_internal_queue_prefill_generation_split': None, 'DNS_TCP_TLS_separate_wall': None, 'per_token_arrival_wall': None}, 'missing_reason': 'not directly observable without changing frozen decks/provider or adding other instrumentation', 'headers_recorded': False, 'private_configuration_recorded': False})

class Proxy:

    def __init__(self, original, overrides):
        self.original, self.overrides = (original, overrides)

    def __getattr__(self, name):
        return self.overrides.get(name, getattr(self.original, name))

def attach(backend, trace):
    """Instance-only wrappers; original functions/arguments/results stay intact."""
    e = backend.engine
    old_candidate = e.candidate

    def candidate(*a, **k):
        c, cached = old_candidate(*a, **k)
        trace.point('candidate.cache', candidate_id=c['id'], cache_hit=cached, role=c['role'])
        return (c, cached)
    e.candidate = candidate
    original = e.run_spice

    def attempt(function, *a, **k):
        row = e.attempts[-1] if e.attempts else {}
        slot = e.slots[row['slot_id']] if row else {}
        with trace.span('spice.attempt', attempt=row.get('attempt'), candidate_id=row.get('candidate_id'), slot_id=row.get('slot_id'), slot_name=slot.get('name'), endpoint=slot.get('endpoint'), scheduled_analyses=row.get('scheduled_analyses')):
            return function(*a, **k)
    function = original.__func__
    sp = function.__globals__.get('subprocess')
    if sp is not None:
        proxy = Proxy(sp, {'run': lambda *a, **k: attempt(sp.run, *a, **k)})
        original = injected(function, subprocess=proxy).__get__(e, type(e))
    if e.executor is not None:
        ex = e.executor
        e.executor = lambda *a, **k: attempt(ex, *a, **k)

    def run_spice(c, name, deck, _engine_run=original):
        with trace.span('spice.slot', candidate_id=c['id'], slot_name=name, endpoint=c['role'] == 'endpoint'):
            return _engine_run(c, name, deck)
    e.run_spice = run_spice
    labels = {'initialize': 'initialization', 'snapshot': 'state_and_retrieval', 'characterize': 'ota.characterization', 'diagnose': 'diagnosis', 'allocation': 'planning', 'size_devices': 'optimization', 'calibrate': 'calibration', 'repeat_full': 'independent_calibration_measurement', 'evaluate': 'full_measurement_and_acceptance', 'finish': 'endpoint_and_finalization'}

    def instrument(function):
        glob = function.__globals__
        changes = {}
        for name in ('physical', 'measured_runner', 'characterization'):
            module = glob.get(name)
            if module is None:
                continue
            overrides = {}
            for method in ('analyze_matrix', 'analyze_performance', 'extract_measurements', 'analyze'):
                f = getattr(module, method, None)
                if callable(f):

                    def measured(*a, _f=f, _label=name + '.' + method, **k):
                        return trace.call('acceptance.' + _label, _f, *a, **k)
                    overrides[method] = measured
            if overrides:
                changes[name] = Proxy(module, overrides)
        tr = glob.get('HighpassRun')
        if tr is not None and callable(getattr(tr, 'evaluate', None)) and (tr.evaluate is not function):
            changes['HighpassRun'] = Proxy(tr, {'evaluate': instrument(tr.evaluate)})
        powell = glob.get('native_powell')
        if callable(powell):
            changes['native_powell'] = lambda *a, **k: trace.call('optimizer.native_powell', powell, *a, **k)
        return injected(function, **changes) if changes else function
    for name, label in labels.items():
        method = getattr(backend, name, None)
        if method is None:
            continue
        original = instrument(method.__func__).__get__(backend, type(backend))

        def wrapped(*a, _original=original, _label=label, **k):
            before = len(backend.auxiliary)
            with trace.span(_label):
                result = _original(*a, **k)
            if _label == 'ota.characterization':
                trace.point('ota.descriptor_cache', reused=len(backend.auxiliary) == before)
            return result
        setattr(backend, name, wrapped)
    original_execute = backend.execute

    def execute(decision):
        with trace.span('action.' + decision['action']['name'], state_id=decision['state_id']):
            return original_execute(decision)
    backend.execute = execute

class TimedOpener:

    def __init__(self, original, engine, trace):
        self.original, self.engine, self.trace = (original, engine, trace)

    def open(self, request, timeout):
        trace = self.trace
        span = trace.begin('api.attempt', attempt=len(self.engine.api_calls), IO_timeout_seconds=timeout)
        try:
            with trace.span('api.open_headers'):
                response = self.original.open(request, timeout=timeout)
        except BaseException as exc:
            trace.end(span, exc)
            raise

        class Response:
            first = False

            def __enter__(self):
                try:
                    self.active = response.__enter__()
                    return self
                except BaseException as exc:
                    trace.end(span, exc)
                    raise

            @property
            def status(self):
                return getattr(self.active, 'status', None)

            def readline(self, n):
                line = self.active.readline(n)
                if not self.first and line.startswith(b'data:') and (line[5:].strip() not in (b'', b'[DONE]')):
                    self.first = True
                    trace.point('api.first_SSE_data_line', attempt=span['tags']['attempt'], not_equivalent_to_first_nonempty_answer_token=True)
                return line

            def read(self, n):
                return self.active.read(n)

            def __exit__(self, *args):
                exc = args[1]
                try:
                    return response.__exit__(*args)
                except BaseException as error:
                    exc = error
                    raise
                finally:
                    trace.end(span, exc)
        return Response()
