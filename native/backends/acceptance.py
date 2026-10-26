"""Version-owned initialization; public physics, gates and endpoint are unchanged."""
import backends.lowpass as t1
import backends.highpass as typed
from evidence import save
ExecutionPaused = t1.ExecutionPaused
CandidateBudgetExhausted = t1.CandidateBudgetExhausted
ToolUnavailable = t1.ToolUnavailable

def initial_pass(backend):
    row = backend.current
    return bool(isinstance(row, dict) and row.get('status') == 'success' and row.get('evidence') and (row.get('failed_constraints') == []) and (row.get('measurement_status', 'valid') == 'valid'))
