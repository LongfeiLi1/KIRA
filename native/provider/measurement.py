"""Frozen streamed provider with state-bound truthful measurement guidance."""
import json
from pathlib import Path
from provider.client import StreamingClient as BaseClient
from provider.client import StreamAuditedOpener
from provider.stream import StreamClient
from provider.stream import injected
from provider.transport import RetryingClient
from provider.guidance import build_payload
from provider.guidance import failure_context
from evidence import save

class StreamingClient(BaseClient):

    def __init__(self, engine, prior_calls, *, runtime_authority, **kwargs):
        super().__init__(engine, prior_calls, **kwargs)
        self.runtime_authority = dict(runtime_authority)

        def payload(state, protocol, feedback=None):
            context = None
            observation = state['observation']
            if observation.get('measurement_status') != 'valid':
                cid = observation['candidate_id']
                if type(cid) is not int or cid < 0:
                    raise ValueError('invalid_measurement_candidate_identity')
                folder = Path(engine.directory) / ('measured_%03d' % cid)
                result, attempt = (folder / 'result.json', folder / 'matrix_attempt.json')
                if result.exists():
                    saved = json.loads(result.read_text())
                    if saved != observation:
                        raise ValueError('saved_measurement_state_mismatch')
                    if attempt.exists():
                        context = failure_context(observation, json.loads(attempt.read_text()))
            p = build_payload(state, protocol, feedback, runtime_authority=self.runtime_authority, context=context)
            audit = {'state_id': state['state_id'], 'candidate_id': observation['candidate_id'], 'runtime_authority': self.runtime_authority, 'failure_context': context, 'guidance_version': 'measurement_guidance_v1', 'original_state_unchanged': True, 'legal_finish_preserved': True, 'physical_gates_unchanged': True}
            n = len(engine.api_calls) + 1
            save(Path(engine.directory) / ('guidance_request_%03d.json' % n), audit)
            return p

        class GuidedStreamClient(StreamClient):
            decide = injected(StreamClient.decide, build_payload=payload)
        self.decide = injected(RetryingClient.decide, build_payload=payload, BailianDecisionClient=GuidedStreamClient, AuditedOpener=StreamAuditedOpener).__get__(self, type(self))

    def audit(self):
        super().audit()
        path = Path(self.engine.directory) / 'transport_retry_control.json'
        audit = json.loads(path.read_text())
        audit.update(measurement_guidance_enabled=True, legal_finish_preserved=True, state_validator_changed=False)
        save(path, audit)
