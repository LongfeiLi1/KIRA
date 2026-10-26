"""Explicit legacy SK LP assumptions, isolated from generic matrix machinery.

Read-only reference for existing A/B. No import, edit, or execution of its
evaluator. A caller must supply independently verified matrix index ordering.
"""
import math
from .core import ChargeAnchor, ContractError, GroundedSource, Layout, Ports
CORE_NODES = ('vbias', 'xu.s1o', 'vp', 'xu.tail', 'xu.s1x', 'out', 'xu.z', 'a')
BODY_NODES = tuple((f"m.{('xb' if i == 8 else 'xu')}.xm{i}.msky130_fd_pr__{('pfet' if i in (3, 4, 6) else 'nfet')}_01v8#{part}" for i in range(1, 9) for part in ('body', 'dbody', 'sbody')))
DYNAMIC_NODES = CORE_NODES + BODY_NODES
SOURCES = (GroundedSource('vin#branch', 'in'), GroundedSource('vdd#branch', 'vdd'))

def explicit_sk_contract(matrix_names_in_exporter_order, cf_f):
    layout = Layout(tuple(matrix_names_in_exporter_order), DYNAMIC_NODES, SOURCES)
    layout.validate()
    if not math.isfinite(cf_f) or cf_f <= 0:
        raise ContractError('positive explicit Cf required')
    ports = Ports({'vin#branch': 1.0}, {'out': 1.0})
    anchors = (ChargeAnchor('a', 'a', cf_f), ChargeAnchor('a', 'out', -cf_f))
    return (layout, ports, anchors)
