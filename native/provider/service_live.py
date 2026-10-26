"""Omitted output cap with the original cumulative 32-call ledger."""
from provider.recovery import ProviderFailure
from provider.live import AuthorizedLazyClient as OldLazyClient
from rc_agent.qwen_adapter import QwenConnectivityAdapter
from provider.service import BailianDecisionClient
from provider.service import MODEL

class ServiceDefaultBudget:

    def __init__(self, engine, prior_calls):
        self.engine, self.prior_calls = (engine, prior_calls)
        self.directory, self.protocol = (engine.directory, engine.protocol)

    def reserve_api(self, model, tokens):
        if model != MODEL or tokens is not None:
            raise ProviderFailure('service_default_output_parameters_required')
        if self.protocol.get('max_completion_tokens_per_call') is not None:
            raise ProviderFailure('artificial_output_cap_present')
        if self.protocol.get('output_token_policy') != 'omit_use_service_default':
            raise ProviderFailure('service_default_policy_missing')
        if self.protocol.get('max_llm_calls_per_case') != 32:
            raise ProviderFailure('case_api_cap_changed')
        if self.prior_calls + len(self.engine.api_calls) >= 32:
            raise ProviderFailure('cumulative_case_api_cap_reached')
        row = {'attempt': len(self.engine.api_calls) + 1, 'model': model, 'max_completion_tokens': None, 'output_cap_parameter_sent': False, 'output_token_policy': 'omit_use_service_default', 'status': 'started', 'combined_attempt': self.prior_calls + len(self.engine.api_calls) + 1}
        self.engine.api_calls.append(row)
        self.engine.save()
        return row

    def finish_api(self, row, record):
        return self.engine.finish_api(row, record)

class AuthorizedLazyClient(OldLazyClient):

    def __init__(self, engine, prior_calls, *, adapter_factory=None):
        self.ledger = ServiceDefaultBudget(engine, prior_calls)
        self.factory = adapter_factory or QwenConnectivityAdapter.from_existing_config
        self.client = None

    def decide(self, state, *, feedback=None, request_kind='original'):
        if self.ledger.prior_calls + len(self.ledger.engine.api_calls) >= 32:
            raise ProviderFailure('cumulative_case_api_cap_reached')
        if self.client is None:
            try:
                self.client = BailianDecisionClient(self.ledger, transfer_authorized=True, adapter=self.factory())
            except Exception:
                raise ProviderFailure('existing_provider_configuration_unavailable') from None
        return self.client.decide(state, feedback=feedback, request_kind=request_kind)
