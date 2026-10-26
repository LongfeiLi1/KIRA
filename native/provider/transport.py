"""Bounded transport engineering over the unchanged service-default provider."""
import errno
import copy
import http.client
import json
import re
import socket
import ssl
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from provider.service import BailianDecisionClient
from provider.service import MODEL
from provider.service import build_payload
from provider.service import usage_view
from provider.recovery import ProviderFailure
from evidence import MAX_BYTES
from evidence import retain_body
from evidence import save
from evidence import sha
HTTP_TRANSIENT = {502, 503, 504}
SERVICE_TRANSIENT = {'ServiceUnavailable', 'ModelUnavailable', 'SystemError', 'ModelServiceFailed', 'InternalError.Algo', 'RequestTimeOut', 'Throttling.RateQuota', 'Throttling.BurstRate', 'Throttling.ServiceOverloaded', 'Throttling.ResourceExhausted'}
SERVICE_PERMANENT = {'InvalidApiKey', 'AccessDenied', 'InvalidParameter', 'InsufficientBalance', 'PrepaidBillOverdue', 'PostpaidBillOverdue', 'BudgetLimitExceeded', 'CommodityNotPurchased', 'AllocationQuota.FreeTierOnly'}
RETRY_ERRNO = {errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE, errno.ETIMEDOUT, errno.EAGAIN, errno.EINTR}

def classify(exc, service_code=None):
    """Explicit transport allowlist. Local/SSL/schema errors fail closed."""
    if isinstance(exc, HTTPError):
        retry = service_code not in SERVICE_PERMANENT and (exc.code in HTTP_TRANSIENT or (exc.code in {429, 500} and service_code in SERVICE_TRANSIENT))
        return {'category': 'http_error', 'http_status': exc.code, 'service_code': service_code, 'retryable': retry}
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLError):
        return {'category': 'tls_error', 'retryable': False}
    if isinstance(reason, TimeoutError):
        return {'category': 'io_timeout', 'retryable': True}
    if isinstance(reason, http.client.IncompleteRead):
        return {'category': 'incomplete_read', 'retryable': True}
    if isinstance(reason, http.client.RemoteDisconnected):
        return {'category': 'remote_disconnected', 'retryable': True}
    if isinstance(reason, socket.gaierror):
        return {'category': 'dns_error', 'retryable': reason.errno == socket.EAI_AGAIN}
    if isinstance(reason, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return {'category': 'connection_error', 'retryable': True}
    if isinstance(reason, OSError) and reason.errno in RETRY_ERRNO:
        return {'category': 'connection_error', 'retryable': True}
    return {'category': 'unclassified_nonretryable', 'retryable': False}

class AuditedOpener:

    def __init__(self, original, ledger, secret):
        self.original, self.ledger, self.secret = (original, ledger, secret)
        self.failure = None
        self.expected_wire = None
        self.network_dispatches = 0
        self.retry_dispatches = 0
        self.is_retry = False
        self.last_network_dispatch = False
        self.error_usage = None

    def folder(self):
        return Path(self.ledger.directory) / ('api_%03d' % len(self.ledger.engine.api_calls))

    def failed(self, exc, raw=None):
        code = None
        if isinstance(exc, HTTPError):
            try:
                raw = exc.read(MAX_BYTES + 1)
            except http.client.IncompleteRead as part:
                raw = part.partial
            except Exception:
                raw = None
            if raw is not None:
                try:
                    body = json.loads(raw)
                    error = body.get('error', body) if isinstance(body, dict) else {}
                    candidate = error.get('code') if isinstance(error, dict) else None
                    if isinstance(candidate, str) and re.fullmatch('[A-Za-z0-9_.-]{1,128}', candidate):
                        code = candidate
                except (ValueError, TypeError):
                    pass
        elif isinstance(exc, http.client.IncompleteRead):
            raw = exc.partial
        if raw is not None:
            retain_body(self.folder(), raw, self.secret)
            try:
                usage = usage_view(json.loads(raw))
                save(self.folder() / 'transport_error_usage.json', usage, self.secret)
                if usage['usage_complete'] and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']:
                    self.error_usage = usage
            except (ValueError, TypeError):
                pass
        detail = classify(exc, code)
        try:
            save(self.folder() / 'transport_failure_detail.json', {**detail, 'error_class': type(exc).__name__, 'raw_bytes_retained': len(raw) if raw is not None else None, 'missing_usage_is_unknown': True}, self.secret)
        except Exception:
            self.failure = {'category': 'evidence_persistence_error', 'retryable': False}
            raise
        self.failure = detail

    def open(self, request, timeout):
        if sha(request.data) != self.expected_wire:
            raise ValueError('transport_retry_wire_mismatch')
        self.last_network_dispatch = True
        self.network_dispatches += 1
        self.retry_dispatches += int(self.is_retry)
        try:
            response = self.original.open(request, timeout=timeout)
        except Exception as exc:
            self.failed(exc)
            raise
        owner = self

        class Response:

            def __enter__(self):
                try:
                    self.active = response.__enter__()
                except Exception as exc:
                    owner.failed(exc)
                    raise
                return self

            @property
            def status(self):
                return getattr(self.active, 'status', None)

            def read(self, limit):
                try:
                    return self.active.read(limit)
                except Exception as exc:
                    owner.failed(exc)
                    raise

            def __exit__(self, *args):
                try:
                    return response.__exit__(*args)
                except Exception as exc:
                    owner.failed(exc)
                    raise
        return Response()

class RetryingClient:
    is_mock = False
    is_live_transport = True

    def __init__(self, engine, prior_calls, *, adapter_factory, max_case_retries=1, delay_seconds=2, sleeper=time.sleep):
        from provider.service_live import ServiceDefaultBudget
        if type(max_case_retries) is not int or max_case_retries != 1:
            raise ValueError('reviewed_case_retry_limit_is_one')
        self.engine, self.ledger = (engine, ServiceDefaultBudget(engine, prior_calls))
        self.factory, self.client, self.opener = (adapter_factory, None, None)
        self.retry_used, self.max_case_retries = (0, max_case_retries)
        self.delay_seconds, self.sleeper = (delay_seconds, sleeper)
        self.events = []
        original_save = engine.save

        def budget_save():
            original_save()
            path = Path(engine.directory) / 'budget.json'
            data = json.loads(path.read_text())
            data.update(common_executor_automatic_api_retries=0, provider_transport_retries=self.actual_retries, automatic_api_retries=self.actual_retries)
            save(path, data)
        engine.save = budget_save

    @property
    def actual_retries(self):
        return self.opener.retry_dispatches if self.opener is not None else 0

    def audit(self):
        save(Path(self.engine.directory) / 'transport_retry_control.json', {'max_case_transport_retries': self.max_case_retries, 'actual_transport_retries': self.actual_retries, 'retry_quota_consumed': self.retry_used, 'events': self.events, 'all_attempts_count_toward_API32': True, 'unknown_usage_not_zero': True, 'timeout_seconds': 300, 'timeout_semantics': 'blocking I/O, not whole-request deadline'})
        self.engine.save()

    def decide(self, state, *, feedback=None, request_kind='original'):
        binding = sha(json.dumps(state, sort_keys=True, allow_nan=False).encode())
        stable_state, stable_feedback = (copy.deepcopy(state), copy.deepcopy(feedback))
        expected_wire = sha(json.dumps(build_payload(stable_state, self.engine.protocol, stable_feedback)).encode())
        if self.client is None:
            if self.ledger.prior_calls + len(self.engine.api_calls) >= 32:
                raise ProviderFailure('cumulative_case_api_cap_reached')
            adapter = self.factory()
            self.opener = AuditedOpener(adapter._opener, self.ledger, adapter._settings.api_key)
            self.client = BailianDecisionClient(self.ledger, transfer_authorized=True, adapter=SimpleNamespace(_settings=adapter._settings, _opener=self.opener))
        first_attempt = len(self.engine.api_calls) + 1
        retrying = False
        while True:
            if binding != sha(json.dumps(state, sort_keys=True, allow_nan=False).encode()):
                raise ProviderFailure('transport_retry_state_mutated')
            self.opener.failure = None
            self.opener.error_usage = None
            self.opener.expected_wire = expected_wire
            self.opener.is_retry = retrying
            self.opener.last_network_dispatch = False
            before = len(self.engine.api_calls)
            attempt_is_retry = retrying
            try:
                content = self.client.decide(stable_state, feedback=stable_feedback, request_kind=request_kind if not retrying else 'transport_retry:' + request_kind)
                return content
            except ProviderFailure as exc:
                failure = self.opener.failure
                permitted = bool(failure and failure['retryable'] and (self.retry_used < self.max_case_retries) and (self.ledger.prior_calls + len(self.engine.api_calls) < 32))
                self.events.append({'original_request_kind': request_kind, 'state_sha256': binding, 'first_attempt': first_attempt, 'failed_attempt': len(self.engine.api_calls), 'provider_failure': exc.code, 'transport_failure': failure, 'wire_sha256': expected_wire, 'will_retry': permitted})
                if not permitted:
                    raise
                self.retry_used += 1
                retrying = True
                self.audit()
                self.sleeper(self.delay_seconds)
            finally:
                if len(self.engine.api_calls) > before:
                    row = self.engine.api_calls[-1]
                    row.update(transport_retry=attempt_is_retry, network_dispatch_attempted=self.opener.last_network_dispatch, original_request_kind=request_kind, retry_wire_sha256=expected_wire)
                    if self.opener.error_usage is not None:
                        usage = self.opener.error_usage
                        row.update({k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'answer_tokens')})
                        row.update(usage_details=usage['raw_usage'], usage_source='retained_transport_error_envelope')
                self.audit()
