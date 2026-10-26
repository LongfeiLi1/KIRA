"""Controller resource paths and canonical serialization."""
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE_SHA = 'cbb3ba4347b76606e0fa65f93c0344a8f6ee3ecf24b0b023e4174957bd976d16'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def model_binding():
    import os
    filename=os.environ.get('KIRA_PDK_MANIFEST')
    if not filename:raise RuntimeError('Set KIRA_PDK_MANIFEST to the model/simulator binding JSON')
    value=read(Path(filename).expanduser())
    if not isinstance(value,dict) or not isinstance(value.get('files'),dict) or 'simulator' not in value:
        raise ValueError('Model binding requires files and simulator metadata')
    return value
