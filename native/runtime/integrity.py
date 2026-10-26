"""Source fingerprints for associating runtime outputs with their implementation."""
import hashlib, json
from runtime.paths import CODE_ROOT

def _files():
    return {p.relative_to(CODE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CODE_ROOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json'}
            and p.relative_to(CODE_ROOT).parts[0] in {'execution', 'initialization', 'native', 'numerics', 'pipeline', 'resources'}}

def verify():
    return hashlib.sha256(json.dumps(_files(), sort_keys=True).encode()).hexdigest()

def manifest():
    files = _files()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'files': files, 'source_manifest_sha256': digest, 'public_release': True}
