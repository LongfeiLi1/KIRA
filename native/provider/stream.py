"""SSE normalization with original provider validation and exact state payload."""
import hashlib, http.client, json, time, types
from pathlib import Path
import provider.service as frozen
from provider.recovery import ProviderFailure
from evidence import MAX_BYTES
from evidence import clean
from evidence import save
MAX_STREAM_BYTES = 64 * 1024 * 1024

def stream_payload(state, protocol, feedback=None):
    payload = frozen.build_payload(state, protocol, feedback)
    payload['stream'] = True
    payload['stream_options'] = {'include_usage': True}
    return payload

def injected(function, **overrides):
    copy = types.FunctionType(function.__code__, dict(function.__globals__, **overrides), function.__name__, function.__defaults__, function.__closure__)
    copy.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return copy

class StreamClient(frozen.BailianDecisionClient):
    decide = injected(frozen.BailianDecisionClient.decide, build_payload=stream_payload)

class Collector:

    def __init__(self, folder, secret):
        self.folder = Path(folder)
        self.secret = secret
        self.raw = bytearray()
        self.raw_hash = hashlib.sha256()
        self.frames = 0
        self.model = None
        self.identity = None
        self.role = None
        self.content = []
        self.reasoning = []
        self.content_seen = False
        self.reasoning_seen = False
        self.finish = None
        self.usage = None
        self.done = False
        self.eof = False
        self.started = time.monotonic()
        self.times = {}
        self.last_progress = 0

    def reject(self, code):
        raise ProviderFailure('stream_' + code)

    def retain(self, line):
        if not isinstance(line, bytes):
            self.reject('nonbyte_line')
        self.raw_hash.update(line)
        self.raw.extend(line)
        if len(self.raw) > MAX_STREAM_BYTES:
            self.reject('oversized_response')

    def event(self, text):
        if text == '[DONE]':
            self.done = True
            return
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            self.reject('invalid_json_frame')
        if not isinstance(obj, dict):
            self.reject('invalid_frame')
        if 'error' in obj:
            self.reject('provider_error_frame')
        identity, model = (obj.get('id'), obj.get('model'))
        if not isinstance(identity, str) or not identity:
            self.reject('missing_stream_id')
        if model != frozen.MODEL:
            self.reject('model_mismatch')
        if self.identity is None:
            self.identity, self.model = (identity, model)
        if identity != self.identity or model != self.model:
            self.reject('stream_identity_changed')
        choices = obj.get('choices')
        if not isinstance(choices, list):
            self.reject('invalid_choices')
        self.frames += 1
        if not choices:
            if self.finish is None or self.usage is not None or (not isinstance(obj.get('usage'), dict)):
                self.reject('invalid_usage_frame')
            self.usage = obj['usage']
            self.times['usage'] = time.monotonic() - self.started
            return
        if self.finish is not None or self.usage is not None:
            self.reject('content_after_finish')
        if len(choices) != 1 or not isinstance(choices[0], dict) or type(choices[0].get('index')) is not int or (choices[0]['index'] != 0):
            self.reject('multiple_or_invalid_choice')
        if obj.get('usage') is not None:
            self.reject('usage_not_last_empty_choice')
        choice = choices[0]
        delta = choice.get('delta')
        if not isinstance(delta, dict):
            self.reject('invalid_delta')
        role = delta.get('role')
        if role is not None:
            if role != 'assistant' or self.role is not None:
                self.reject('invalid_role')
            self.role = role
        for key, dest in (('content', self.content), ('reasoning_content', self.reasoning)):
            value = delta.get(key)
            if value is not None:
                if not isinstance(value, str):
                    self.reject('invalid_' + key)
                dest.append(value)
                if key == 'content':
                    self.content_seen = True
                else:
                    self.reasoning_seen = True
                if value and key not in self.times:
                    self.times[key] = time.monotonic() - self.started
        if any((delta.get(k) is not None for k in ('tool_calls', 'function_call', 'audio', 'refusal'))):
            self.reject('unsupported_delta')
        reason = choice.get('finish_reason')
        if reason is not None:
            if not isinstance(reason, str):
                self.reject('invalid_finish_reason')
            self.finish = reason
            self.times['finish'] = time.monotonic() - self.started

    def progress(self, status):
        text = bytes(self.raw).decode('utf-8', errors='replace')
        (self.folder / 'response_sse_raw.log').write_text(clean(text, self.secret))
        save(self.folder / 'stream_audit.json', {'status': status, 'frames': self.frames, 'raw_bytes': len(self.raw), 'wire_response_sha256_before_redaction': self.raw_hash.hexdigest(), 'model': self.model, 'stream_id': self.identity, 'finish_reason': self.finish, 'done_marker_received': self.done, 'normal_EOF_received': self.eof, 'usage': frozen.usage_view({'usage': self.usage}), 'phase_seconds': self.times, 'elapsed_seconds': time.monotonic() - self.started, 'response_raw_json_semantics': 'normalized envelope; actual SSE wire in response_sse_raw.log', 'provider_acceptance_still_requires_original_validator': True}, self.secret)
        self.last_progress = time.monotonic()

    def read(self, response):
        data = []
        try:
            while not self.done:
                line = response.readline(MAX_BYTES + 1)
                if not line:
                    self.eof = True
                    if data:
                        self.event('\n'.join(data))
                        data = []
                    break
                self.retain(line)
                try:
                    text = line.decode('utf-8').rstrip('\r\n')
                except UnicodeDecodeError:
                    self.reject('invalid_UTF8')
                if not text:
                    if data:
                        self.event('\n'.join(data))
                        data = []
                elif text.startswith(':'):
                    pass
                elif text.startswith('data:'):
                    data.append(text[5:].lstrip(' '))
                elif text == 'event: message':
                    pass
                else:
                    self.reject('unsupported_SSE_field')
                if time.monotonic() - self.last_progress >= 1:
                    self.progress('receiving')
            if self.role != 'assistant' or self.finish is None or self.usage is None:
                self.reject('incomplete_response')
            message = {'role': 'assistant', 'content': ''.join(self.content) if self.content_seen else None}
            if self.reasoning_seen:
                message['reasoning_content'] = ''.join(self.reasoning)
            obj = {'id': self.identity, 'model': self.model, 'choices': [{'index': 0, 'finish_reason': self.finish, 'message': message}], 'usage': self.usage}
            save(self.folder / 'stream_normalized_envelope.json', obj, self.secret)
            self.progress('normalized_pending_original_validation')
            return json.dumps(obj).encode()
        except Exception as exc:
            if isinstance(exc, http.client.IncompleteRead) and isinstance(exc.partial, bytes):
                self.raw_hash.update(exc.partial)
                self.raw.extend(exc.partial)
            self.progress('failed_incomplete_or_invalid')
            raise

class StreamOpener:

    def __init__(self, original, ledger, secret):
        self.original, self.ledger, self.secret = (original, ledger, secret)
        self.last = None

    def open(self, request, timeout):
        self.last = None
        response = self.original.open(request, timeout=timeout)
        folder = Path(self.ledger.directory) / ('api_%03d' % len(self.ledger.engine.api_calls))
        collector = Collector(folder, self.secret)
        self.last = collector
        owner = self

        class StreamResponse:

            def __enter__(self):
                self.active = response.__enter__()
                return self

            @property
            def status(self):
                return getattr(self.active, 'status', None)

            def read(self, limit):
                return collector.read(self.active)

            def __exit__(self, *args):
                return response.__exit__(*args)
        return StreamResponse()
