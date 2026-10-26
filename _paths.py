"""Locate the numerical backends in a source checkout or installation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent

def activate():
    for path in reversed([ROOT / 'native', ROOT, ROOT / 'execution']):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
