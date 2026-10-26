"""External OTA netlist renderer used by physical circuit backends."""
import importlib.util
import os
from pathlib import Path
import re

def render_ota(devices):
    filename = os.environ.get('KIRA_OTA_RENDERER')
    if not filename:
        raise RuntimeError('Set KIRA_OTA_RENDERER to a Python module providing ota_netlist(devices)')
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('KIRA_OTA_RENDERER must point to an existing Python file')
    spec = importlib.util.spec_from_file_location('_kira_external_ota', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load the external OTA renderer')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    renderer = getattr(module, 'ota_netlist', None)
    if not callable(renderer):
        raise ValueError('External module must provide ota_netlist(devices)')
    netlist = renderer(devices)
    if not isinstance(netlist, str) or any((re.search('(?im)^\\s*\\.subckt\\s+' + name + '\\b', netlist) is None for name in ('ota', 'bias'))):
        raise ValueError('Renderer must return SPICE ota and bias subcircuits')
    return netlist
