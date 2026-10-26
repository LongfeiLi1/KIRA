"""Run the SK decision loop using a caller-owned, already prepared backend."""
import copy
from .contracts import Limits, digest, validate_decision
from .loop import run_loop
from .session import NativeSession

class _NoModel:
    calls = 0
    is_mock = False
    receipts = []

    def decide(self, state):
        raise RuntimeError('A model client is required in full mode')

def run(backend, client=None, *, task, run_id, receipt_verifier, endpoint_verifier, arm='full', limits=Limits(), source_check=None):
    if backend.case.get('type_id') not in {'T1', 'T2'} or task != backend.case:
        raise ValueError('Exact Sallen-Key backend task required')
    if arm not in {'full', 'deterministic'}:
        raise ValueError('Choose full or deterministic')
    if not callable(receipt_verifier) or not callable(endpoint_verifier):
        raise TypeError('Independent receipt and endpoint verifiers are required')
    if arm == 'full' and client is None:
        raise ValueError('Supply an authorized model client')
    from control.decisions import validate_decision as native_validator
    session = NativeSession(backend, install=None, native_validator=native_validator, receipt_verifier=receipt_verifier, family_verifier=endpoint_verifier, run_id=run_id, installed_arm=arm, source_check=source_check)
    expected = copy.deepcopy(task)

    class Router:

        def route(self, value):
            return {'status': 'supported' if value == expected else 'unsupported', 'package_id': 'sallen_key'}
    controller = None
    if arm == 'deterministic':
        from control.controller import FixedRuleController
        policy = FixedRuleController(True)

        class Controller:

            def choose(self, state, wrapped):
                self.last_record = {}
                return policy.choose(state, backend, None, self.last_record, lambda: None)
        controller = Controller()
        client = _NoModel()

    def decode(raw, state):
        return (validate_decision(raw, state), {})
    return run_loop(task, Router(), lambda *a: session, client, arm=arm, limits=limits, digest=digest, decode=decode, controller=controller, stop_output=False)
