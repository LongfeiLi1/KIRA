"""Correct cap semantics and retain evidence before validation; zero retries."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request
from urllib.error import HTTPError, URLError
from evidence import MAX_BYTES
from evidence import clean
from evidence import retain_body
from evidence import save
from evidence import sha
from control.decisions import contract
MODEL = 'qwen3.8-flash'
TOKEN_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
from provider.recovery import ProviderFailure

def integer(value):
    return type(value) is int and value >= 0

def usage_view(data):
    raw = data.get('usage') if isinstance(data, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    top = {k: raw.get(k) if integer(raw.get(k)) else None for k in TOKEN_KEYS}
    details = raw.get('completion_tokens_details')
    reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
    reasoning = reasoning if integer(reasoning) else None
    completion = top['completion_tokens']
    answer = completion - reasoning if completion is not None and reasoning is not None and (reasoning <= completion) else None
    return {**top, 'reasoning_tokens': reasoning, 'answer_tokens': answer, 'reasoning_usage_known': reasoning is not None, 'raw_usage': raw, 'usage_complete': all((v is not None for v in top.values()))}

def audit_cap(payload, usage):
    """Historical answer caps cannot be checked using aggregate usage alone."""
    total_cap, answer_cap = (payload.get('max_completion_tokens'), payload.get('max_tokens'))
    if total_cap is None and answer_cap is None:
        return {'parameter': None, 'scope': 'service_default_reasoning_plus_answer', 'cap': None, 'tolerance_tokens': None, 'observed_tokens': usage['completion_tokens'], 'status': 'not_checked_no_explicit_cap', 'platform_limits_still_apply': True}
    if total_cap is not None and answer_cap is not None:
        raise ProviderFailure('ambiguous_output_caps')
    cap = total_cap if total_cap is not None else answer_cap
    if type(cap) is not int or cap <= 0:
        raise ProviderFailure('invalid_output_cap')
    count = usage['completion_tokens'] if total_cap is not None else usage['answer_tokens']
    tolerance = 10 if total_cap is not None else 0
    return {'parameter': 'max_completion_tokens' if total_cap is not None else 'max_tokens', 'scope': 'reasoning_plus_answer' if total_cap is not None else 'answer_only', 'cap': cap, 'tolerance_tokens': tolerance, 'observed_tokens': count, 'status': 'unknown' if count is None else 'exceeded' if count > cap + tolerance else 'within_cap'}

def validate_provider(data, payload):
    if not isinstance(data, dict):
        raise ProviderFailure('invalid_envelope')
    usage = usage_view(data)
    if not usage['usage_complete']:
        raise ProviderFailure('missing_or_invalid_usage')
    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
        raise ProviderFailure('inconsistent_usage')
    if usage['reasoning_tokens'] is not None and usage['reasoning_tokens'] > usage['completion_tokens']:
        raise ProviderFailure('inconsistent_reasoning_usage')
    cap = audit_cap(payload, usage)
    if cap['status'] == 'exceeded':
        raise ProviderFailure('completion_cap_mismatch')
    if data.get('model') != MODEL:
        raise ProviderFailure('provider_model_mismatch')
    choices = data.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or (not isinstance(choices[0], dict)):
        raise ProviderFailure('invalid_choice')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise ProviderFailure('incomplete_model_response')
    message = choice.get('message')
    if not isinstance(message, dict) or not isinstance(message.get('content'), str):
        raise ProviderFailure('missing_decision_content')
    return message['content']

def build_payload(state, protocol, feedback=None):
    tokens = protocol['max_completion_tokens_per_call']
    if tokens is not None or protocol.get('output_token_policy') != 'omit_use_service_default':
        raise ProviderFailure('service_default_output_parameters_required')
    payload = {'model': MODEL, 'messages': [{'role': 'system', 'content': contract(state['case']['type_id']).SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps(state, allow_nan=False)}], 'temperature': 0, 'stream': False, 'response_format': {'type': 'json_object'}, 'enable_thinking': True}
    if feedback is not None:
        payload['messages'].append({'role': 'user', 'content': json.dumps({'decision_repair_feedback': feedback}, allow_nan=False)})
    return payload

class BailianDecisionClient:
    is_mock = False

    def __init__(self, engine, *, transfer_authorized=False, adapter=None):
        if transfer_authorized is not True:
            raise ProviderFailure('transfer_authorization_absent')
        if adapter is None:
            raise ProviderFailure('offline_release_live_provider_disabled')
        self.engine, self.adapter = (engine, adapter)
        self.adapter._settings.validate()

    def decide(self, state, *, feedback=None, request_kind='original'):
        payload = build_payload(state, self.engine.protocol, feedback)
        wire = json.dumps(payload).encode()
        row = self.engine.reserve_api(MODEL, payload.get('max_completion_tokens'))
        folder = Path(self.engine.directory) / ('api_%03d' % row['attempt'])
        folder.mkdir(exist_ok=False)
        secret = self.adapter._settings.api_key
        save(folder / 'call_context.json', {'request_kind': request_kind, 'repair_feedback': feedback, 'state_id': state['state_id']}, secret)
        save(folder / 'request.json', {'payload': payload, 'wire_sha256': sha(wire)}, secret)
        save(folder / 'state.json', state, secret)
        record = {'status': 'failed', 'returned_model': None, **{k: None for k in TOKEN_KEYS}, 'reasoning_tokens': None, 'answer_tokens': None}
        started = time.monotonic()
        obs = {'schema_version': 1, 'requests_started': 0, 'automatic_retries': 0, 'http_status': None, 'finish_reason': None, 'phase_times': [], 'configured_timeout_seconds': self.engine.protocol['api_timeout_seconds'], 'timeout_semantics': 'blocking I/O timeout, not a total wall deadline', 'failure_class': None, 'usage': usage_view({})}

        def phase(name):
            obs['phase'] = name
            obs['phase_times'].append({'phase': name, 'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started})
            save(folder / 'transport.json', obs, secret)
        try:
            settings = self.adapter._settings
            request = Request(settings.base_url.rstrip('/') + '/chat/completions', data=wire, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret}, method='POST')
            obs['requests_started'] = 1
            phase('open_response')
            with self.adapter._opener.open(request, timeout=self.engine.protocol['api_timeout_seconds']) as response:
                obs['http_status'] = getattr(response, 'status', None)
                phase('read_body')
                raw = response.read(MAX_BYTES + 1)
            obs['response_bytes'] = len(raw)
            retain_body(folder, raw, secret)
            if len(raw) > MAX_BYTES:
                raise ProviderFailure('oversized_response')
            phase('decode_json')
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ProviderFailure('invalid_envelope')
            save(folder / 'response.json', data, secret)
            usage = usage_view(data)
            obs['usage'] = clean(usage, secret)
            for key in (*TOKEN_KEYS, 'reasoning_tokens', 'answer_tokens'):
                record[key] = usage[key]
            record['usage_details'] = clean(usage['raw_usage'], secret)
            record['returned_model'] = MODEL if data.get('model') == MODEL else None
            choices = data.get('choices')
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                reason = choices[0].get('finish_reason')
                obs['finish_reason'] = reason if reason in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call') else 'unrecognized_or_missing'
                message = choices[0].get('message')
                if isinstance(message, dict) and isinstance(message.get('content'), str):
                    save(folder / 'decision_raw.json', {'content': message['content']}, secret)
            obs['cap_audit'] = audit_cap(payload, usage)
            phase('validate_provider')
            content = validate_provider(data, payload)
            record['status'] = 'completed'
            phase('complete')
            return clean(content, secret)
        except Exception as exc:
            reason = exc.reason if isinstance(exc, URLError) else None
            if isinstance(exc, ProviderFailure):
                code = exc.code
            elif isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                code = 'transport_timeout'
            elif isinstance(exc, HTTPError):
                code = 'http_error'
                obs['http_status'] = int(exc.code)
            elif isinstance(exc, json.JSONDecodeError):
                code = 'invalid_json_envelope'
            else:
                code = 'transport_or_implementation_error'
            obs['failure_class'] = code
            record.update(error_class=type(exc).__name__, failure_code=code)
            raise ProviderFailure(code) from None
        finally:
            obs['wall_seconds'] = record['wall_seconds'] = time.monotonic() - started
            save(folder / 'transport.json', obs, secret)
            save(folder / 'usage_audit.json', {'attempt': row['attempt'], 'request_kind': request_kind, **record}, secret)
            self.engine.finish_api(row, record)
