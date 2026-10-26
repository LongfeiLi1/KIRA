"""R6 request/response policy, unchanged, with observable timing only."""
import time
from types import SimpleNamespace
from provider.measurement import StreamingClient
from provider.telemetry import TimedOpener

class Client(StreamingClient):

    def __init__(self, engine, *, trace, adapter_factory, runtime_authority, sleeper=time.sleep):
        self.trace = trace

        def factory():
            with trace.span('provider.configuration_load'):
                adapter = adapter_factory()
            return SimpleNamespace(_settings=adapter._settings, _opener=TimedOpener(adapter._opener, engine, trace))
        super().__init__(engine, 0, prior_transport_extras=0, adapter_factory=factory, runtime_authority=runtime_authority, sleeper=lambda seconds: trace.call('transport.retry_wait', sleeper, seconds))
        original = self.decide

        def decide(state, *, feedback=None, request_kind='original'):
            with trace.span('decision.correction' if feedback is not None else 'decision.original', state_id=state['state_id'], request_kind=request_kind):
                return original(state, feedback=feedback, request_kind=request_kind)
        self.decide = decide
