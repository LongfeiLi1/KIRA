"""Bounded, credential-redacted evidence. Never serialize settings or headers."""
import hashlib
import json
import re
from pathlib import Path
MAX_BYTES = 2 * 1024 * 1024
SECRET_KEY = re.compile('(?i)^(?:api[_-]?key|authorization|access[_-]?token|secret|password)$')

def clean(value, secret=None):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if SECRET_KEY.match(k) else clean(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, secret) for v in value]
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, '[REDACTED]')
        value = re.sub('(?i)\\bsk-[a-z0-9_-]{8,}', '[REDACTED]', value)
        value = re.sub('(?i)bearer\\s+[^\\s\\"\'\\\\]+', 'Bearer [REDACTED]', value)
        return value
    return value

def save(path, value, secret=None):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value, secret), indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def retain_body(folder, raw, secret=None):
    """Persist the sanitized text before JSON parsing or any contract check."""
    saved = raw[:MAX_BYTES]
    text = clean(saved.decode('utf-8', errors='replace'), secret)
    try:
        text = json.dumps(clean(json.loads(text), secret), ensure_ascii=False, allow_nan=False)
    except (ValueError, RecursionError):
        text = re.sub('(?i)("(?:api[_-]?key|authorization|access[_-]?token|secret|password)"\\s*:\\s*")[^"]*', '\\1[REDACTED]', text)
    save(Path(folder) / 'response_raw.json', {'body_text': text, 'body_sha256': sha(raw), 'received_bytes': len(raw), 'saved_truncated': len(raw) > MAX_BYTES, 'sanitized': True}, secret)
