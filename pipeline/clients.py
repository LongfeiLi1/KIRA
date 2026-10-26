"""Provider policy adapter; no credentials, endpoint loader or raw opener."""
import copy
import json
from .contracts import canonical
from .protocol import enrich_state, PROMPT_SUPPLEMENT
SYSTEM = 'Control the supplied circuit tool interface. Return exactly schema_version=1,state_id,diagnosis,action,knowledge_ids. diagnosis contains category (model_mismatch/realization_gap/planning_gap/constraints/success/unknown), summary, evidence_ids. action contains name and args matching one allowed tool schema. Cite only supplied evidence and knowledge IDs. Choose one action, never a script or action batch. Numerical tools compute parameters. Proposals and model predictions are not measurements. Do not invent unavailable or withheld residuals; Read qualification applicability and scope for this family: model_trusted is not a universal prerequisite for every action. Null/not_applicable is neither a failed qualification nor permission to skip applicable physical checks. A qualified local repair model is not endpoint acceptance. Use the offered tool interface and family-specific conditions; an offered action is not proof of success. Where stop is offered it remains legal without claiming success; never force finish. Success requires current public receipts and the unchanged independent family verifier. Return JSON only.'
SYSTEM += PROMPT_SUPPLEMENT

class ControlledQwenClient:
    """Host injects a reviewed, globally metered transport; no alternative pool.

    transport.complete(payload) must reserve/settle EACH actual attempt through
    the public control and return content + an authoritative accounting receipt.
    No calls are possible with absent/disabled authority; no output cap invented.
    """
    is_mock = False

    def __init__(self, transport, authority):
        if authority is not None and hasattr(transport, 'job_id') and (transport.job_id != authority.approved_job):
            raise ValueError('provider transport cannot use another job budget')
        self.transport = transport
        self.authority = authority
        self.calls = 0
        self.receipts = []

    def decide(self, state):
        auth = self.authority
        if not auth or not auth.paid_enabled or (not auth.approved_job) or auth.unknown_usage_blocked:
            raise RuntimeError('core paid authority absent or unknown usage holds block API')
        if auth.model != 'qwen3.8-flash':
            raise ValueError('frozen experiment model changed')
        payload = dict(model=auth.model, messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': canonical(enrich_state(state))}], temperature=0, enable_thinking=True, stream=auth.stream, response_format={'type': 'json_object'})
        if auth.stream:
            payload['stream_options'] = {'include_usage': True}
        self.calls += 1
        response = self.transport.complete(payload)
        receipt = response['accounting']
        if receipt.get('model') != auth.model or receipt.get('job_id') != auth.approved_job or receipt.get('settlement') != 'settled' or (receipt.get('reservation_id') is None) or (type(receipt.get('actual_attempts')) is not int) or (receipt['actual_attempts'] != 1):
            raise RuntimeError('current globally accounted provider attempt required; no fallback/retry')
        self.receipts.append(copy.deepcopy(receipt))
        return response['content']

    @property
    def all_attempts(self):
        return copy.deepcopy(getattr(self.transport, 'audit', self.receipts))

class PublicLedgerTransport:
    """One existing-ledger reservation per dispatch; failed costs remain visible."""

    def __init__(self, ledger, job_id, dispatch, *, v103_upper=False, validator=None):
        self.ledger = ledger
        self.job_id = job_id
        self.dispatch = dispatch
        self.v103_upper = v103_upper
        self.validator = validator
        self.audit = []

    def complete(self, payload):
        snapshot = self.ledger.snapshot()
        if not snapshot['paid_enabled'] or snapshot.get('pause') or (not snapshot.get('tariff')):
            raise RuntimeError('public paid gate closed')
        tariff = json.loads(snapshot['tariff']) if isinstance(snapshot['tariff'], str) else snapshot['tariff']
        if tariff['model'] != payload['model'] or payload['model'] != 'qwen3.8-flash':
            raise ValueError('tariff/model mismatch')
        upper = tariff['verified_input_max_tokens']
        output = tariff['verified_output_max_tokens']
        if any((type(x) is not int or x < 1 for x in (upper, output))):
            raise ValueError('verified service ceilings required')
        reservation = self.ledger.reserve(self.job_id, 'new_agent_decision', prompt_upper=upper, output_upper=output)
        record = dict(model=payload['model'], returned_model=None, job_id=self.job_id, reservation_id=reservation, settlement='reserved', actual_attempts=0, charged_upper_nano_rmb=None, actual_invoice_asserted=False, provider_envelope_valid=False, usage=None, accounting_basis=tariff.get('accounting_basis'))
        self.audit.append(record)
        self.ledger.mark_sent(reservation)
        record.update(settlement='sent', actual_attempts=1, transport_status='invoked_or_uncertain')
        try:
            data = self.dispatch(copy.deepcopy(payload))
        except BaseException as exc:
            record.update(settlement='unknown', hold_retained=True, error_class=type(exc).__name__)
            if self.v103_upper:
                self.ledger.settle(reservation, {})
            else:
                self.ledger.uncertain(reservation, 'new_agent_transport_exception')
            raise
        data = data if isinstance(data, dict) else {}
        raw = data.get('usage')
        raw = raw if isinstance(raw, dict) else {}
        details = raw.get('prompt_tokens_details')
        details = details if isinstance(details, dict) else {}
        usage = {key: raw.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        usage.update(cached_tokens=details.get('cached_tokens'), cache_creation_input_tokens=details.get('cache_creation_input_tokens'), cache_mode=tariff['cache_mode'], model=data.get('model'), endpoint_host=tariff['endpoint_host'])
        record.update(usage=copy.deepcopy(usage), raw_usage=copy.deepcopy(raw), returned_model=data.get('model'), transport_status='response_received')
        amount = self.ledger.settle(reservation, usage)
        if amount is None:
            record.update(settlement='unknown', hold_retained=True)
            raise RuntimeError('usage incomplete/unknown; public hold retained')
        record.update(settlement='settled', charged_upper_nano_rmb=amount)
        choices = data.get('choices') or []
        if data.get('model') != payload['model'] or len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
            raise ValueError('incomplete provider decision; charged attempt retained')
        content = self.validator(data, payload) if self.validator else choices[0]['message']['content']
        if type(content) is not str:
            raise ValueError('provider content required')
        record['provider_envelope_valid'] = True
        return dict(content=content, accounting=copy.deepcopy(record))

class ExistingAliyunDispatch:
    """Reuse existing Aliyun settings/opener and sealed SSE Collector. No loader.

    Construct only after core review; no private config is read by this project.
    PublicLedgerTransport must surround every invocation.
    """

    def __init__(self, existing_adapter, collector_type, directory, *, authorized=False):
        if authorized is not True:
            raise RuntimeError('core transport authorization absent')
        from urllib.request import OpenerDirector
        if type(getattr(existing_adapter, '_opener', None)) is not OpenerDirector:
            raise ValueError('plain existing urllib opener required; no nested metering or opaque retry wrapper')
        self.adapter = existing_adapter
        self.collector_type = collector_type
        self.directory = directory
        self.attempts = 0

    def __call__(self, payload):
        from pathlib import Path
        from urllib.request import Request
        settings = self.adapter._settings
        settings.validate()
        if settings.model != 'qwen3.8-flash' or payload['model'] != settings.model:
            raise ValueError('experiment model changed')
        self.attempts += 1
        folder = Path(self.directory) / ('api_%03d' % self.attempts)
        folder.mkdir(parents=True, exist_ok=False)
        wire = json.dumps(payload).encode()
        request = Request(settings.base_url.rstrip('/') + '/chat/completions', data=wire, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + settings.api_key}, method='POST')
        with self.adapter._opener.open(request, timeout=300) as response:
            if payload['stream']:
                raw = self.collector_type(folder, settings.api_key).read(response)
            else:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError('oversized provider response')
        return json.loads(raw)
