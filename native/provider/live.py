"""Lazy existing-config adapter; no probe, retry, or credential logging."""
from provider.recovery import BailianDecisionClient
from provider.recovery import ProviderFailure
from rc_agent.qwen_adapter import QwenConnectivityAdapter

class HistoricalAPIBudget:
    """The original Case's 32-call limit includes earlier pilot/diagnostic calls."""

    def __init__(self, engine, prior_calls):
        self.engine, self.prior_calls = (engine, prior_calls)
        self.directory, self.protocol = (engine.directory, engine.protocol)

    def reserve_api(self, model, tokens):
        if self.prior_calls + len(self.engine.api_calls) >= self.protocol['max_llm_calls_per_case']:
            raise ProviderFailure('cumulative_case_api_cap_reached')
        row = self.engine.reserve_api(model, tokens)
        row['combined_attempt'] = self.prior_calls + row['attempt']
        return row

    def finish_api(self, row, record):
        return self.engine.finish_api(row, record)

class AuthorizedLazyClient:
    is_mock = False
    is_live_transport = True

    def __init__(self, engine, prior_calls, *, adapter_factory=None):
        self.ledger = HistoricalAPIBudget(engine, prior_calls)
        self.factory = adapter_factory or QwenConnectivityAdapter.from_existing_config
        self.client = None

    def decide(self, state, *, feedback=None, request_kind='original'):
        if self.ledger.prior_calls + len(self.ledger.engine.api_calls) >= self.ledger.protocol['max_llm_calls_per_case']:
            raise ProviderFailure('cumulative_case_api_cap_reached')
        if self.client is None:
            try:
                adapter = self.factory()
                self.client = BailianDecisionClient(self.ledger, transfer_authorized=True, adapter=adapter)
            except Exception:
                raise ProviderFailure('existing_provider_configuration_unavailable') from None
        return self.client.decide(state, feedback=feedback, request_kind=request_kind)
