"""No exception message, traceback, locals, credentials, or remote text."""
import re

def exception_view(exc, phase, record=None):
    name = type(exc).__name__
    name = name if re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,79}', name) else 'Exception'
    row = {'code': 'unexpected_exception', 'error_class': name, 'phase': phase, 'message_policy': 'omitted_no_exception_text_or_locals'}
    if record:
        row.update(step=record.get('step'), initial_recovery_trigger=record.get('initial_recovery_trigger'), latest_recovery_trigger=record.get('recovery_trigger'))
    return row
