"""Same per-Case retry allowance, including already-spent historical extras."""
import provider.transport as frozen
from provider.service import usage_view
from evidence import save
from provider.stream import StreamClient
from provider.stream import StreamOpener
from provider.stream import stream_payload
from provider.stream import injected

class StreamAuditedOpener(frozen.AuditedOpener):

    def __init__(self, original, ledger, secret):
        self.stream = StreamOpener(original, ledger, secret)
        super().__init__(self.stream, ledger, secret)

    def failed(self, exc, raw=None):
        super().failed(exc, raw)
        c = self.stream.last
        if c is not None and c.usage is not None:
            u = usage_view({'usage': c.usage})
            if u['usage_complete'] and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']:
                self.error_usage = u

class StreamingClient(frozen.RetryingClient):

    def __init__(self, engine, prior_calls, *, prior_transport_extras, adapter_factory, **kwargs):
        if type(prior_transport_extras) is not int or not 0 <= prior_transport_extras <= 1:
            raise ValueError('historical_transport_extra_allowance_invalid')
        super().__init__(engine, prior_calls, adapter_factory=adapter_factory, **kwargs)
        self.prior_transport_extras = prior_transport_extras
        self.retry_used = prior_transport_extras
    decide = injected(frozen.RetryingClient.decide, build_payload=stream_payload, BailianDecisionClient=StreamClient, AuditedOpener=StreamAuditedOpener)

    def audit(self):
        super().audit()
        from pathlib import Path
        from evidence import clean
        import json
        path = Path(self.engine.directory) / 'transport_retry_control.json'
        d = json.loads(path.read_text())
        d.update(prior_transport_extra_requests=self.prior_transport_extras, cumulative_transport_extra_requests=self.prior_transport_extras + self.actual_retries, streaming_enabled=True, stream_include_usage=True, max_cumulative_transport_extra_requests_per_case=1, requests_may_exceed300s_total_wall_while_each_blocking_IO_stays300=True)
        save(path, d)
