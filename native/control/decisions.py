"""Keep the frozen T1/Type schema and expose stable audit codes."""
from provider.diagnostics import contract
from provider.diagnostics import REASONS as FROZEN_REASONS
REASONS = {**FROZEN_REASONS, 'Nonfinite JSON': 'nonfinite_json'}

class DecisionError(ValueError):

    def __init__(self, code, nested=None):
        self.code, self.nested = (code, nested)
        super().__init__(code)

def validate_decision(raw, state):
    module = contract(state['case']['type_id'])
    try:
        return module.validate_decision(raw, state)
    except module.DecisionError as exc:
        raise DecisionError(REASONS.get(str(exc), 'unclassified_contract_rejection'), REASONS.get(str(exc.__cause__)) if exc.__cause__ else None) from None
