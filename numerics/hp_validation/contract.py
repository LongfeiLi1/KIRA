"""Explicit SK high-pass contract for the existing quasistatic real OTA.

An expected variable SET is not an exported matrix ORDER. Callers must obtain
that order independently before using this contract for numerical evidence.
The preparation deck has no launch entry point and does not promote any Case.
"""
import math
from numerics.topology_stability.core import ChargeAnchor, ContractError, GroundedSource, Layout, Ports
from numerics.topology_stability.decks import ExtractionPorts, build_matrix_deck
from numerics.topology_stability.sk_reference import BODY_NODES
CORE_NODES = ('vbias', 'xu.s1o', 'y', 'xu.tail', 'xu.s1x', 'out', 'xu.z', 'x')
DYNAMIC_NODES = CORE_NODES + BODY_NODES
SOURCES = (GroundedSource('vin#branch', 'in'), GroundedSource('vdd#branch', 'vdd'), GroundedSource('vref#branch', 'vref'))

def expected_hp_variables():
    """Expected coverage only; this return order MUST NOT label a real dump."""
    return DYNAMIC_NODES + tuple((n for s in SOURCES for n in (s.node, s.branch)))

def explicit_hp_contract(matrix_names_in_exporter_order, c1_f, c2_f):
    if any((not math.isfinite(c) or c <= 0 for c in (c1_f, c2_f))):
        raise ContractError('positive finite explicit C1 and C2 required')
    layout = Layout(tuple(matrix_names_in_exporter_order), DYNAMIC_NODES, SOURCES)
    layout.validate()
    anchors = (ChargeAnchor('x', 'x', c1_f + c2_f), ChargeAnchor('x', 'in', -c1_f), ChargeAnchor('x', 'y', -c2_f))
    return (layout, Ports({'vin#branch': 1.0}, {'out': 1.0}), anchors)

def build_hp_matrix_deck(closed_dc_prefix, c1_f, c2_f):
    """Render an extraction plan; no actual matrix order is inferred here."""
    layout, _, _ = explicit_hp_contract(expected_hp_variables(), c1_f, c2_f)
    required = ('XU y out out vdd 0 vbias ota',)
    lines = [line.strip() for line in closed_dc_prefix.splitlines()]
    if any((line not in lines for line in required)):
        raise ContractError('HP extraction requires explicitly closed XU y out out')
    if any((line.lower().startswith(('vloop ', 'iport ')) for line in lines)):
        raise ContractError('HP extraction prefix must not contain an output probe')
    expected = {'c1': ('in', 'x', c1_f), 'c2': ('x', 'y', c2_f), 'rf': ('x', 'out', None), 'rg': ('y', 'vref', None)}
    seen = set()
    required_elements = {'vdd', 'vin', 'vref', 'xb', 'xu', 'c1', 'c2', 'rf', 'rg', 'cl'}
    top_level_seen = set()
    depth = 0
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        key = fields[0].lower()
        if key == '.subckt':
            depth += 1
        if key == '.ends':
            depth -= 1
        if depth or key.startswith(('*', '.')):
            continue
        if key not in required_elements:
            raise ContractError('unreviewed top-level element: ' + key)
        if key in top_level_seen:
            raise ContractError('duplicate top-level element: ' + key)
        top_level_seen.add(key)
        if key == 'xb' and fields[1:] != ['vdd', '0', 'vbias', 'bias']:
            raise ContractError('bias instance differs from registered HP topology')
        if key == 'cl':
            if len(fields) != 4 or fields[1:3] != ['out', '0']:
                raise ContractError('output load wiring differs from registered HP topology')
            try:
                load_f = float(fields[3])
            except ValueError as exc:
                raise ContractError('explicit numeric output load required') from exc
            if not math.isfinite(load_f) or load_f <= 0:
                raise ContractError('positive finite output load required')
        if key not in expected:
            continue
        if key in seen or len(fields) != 4:
            raise ContractError('duplicate/malformed HP passive: ' + key)
        seen.add(key)
        a, b, value = expected[key]
        if fields[1:3] != [a, b]:
            raise ContractError('HP passive wiring differs: ' + key)
        try:
            actual = float(fields[3])
        except ValueError as exc:
            raise ContractError('explicit numeric HP passive required') from exc
        if not math.isfinite(actual) or actual <= 0:
            raise ContractError('positive finite HP passive required')
        if value is not None and (not math.isclose(actual, value, rel_tol=1e-12)):
            raise ContractError('HP charge anchor differs from deck')
    if seen != set(expected):
        raise ContractError('missing HP passive')
    if top_level_seen != required_elements:
        raise ContractError('missing required top-level element: ' + ', '.join(sorted(required_elements - top_level_seen)))
    deck = build_matrix_deck(closed_dc_prefix, layout, ExtractionPorts('Vin', 'out'))
    return {'deck': deck, 'runnable': False, 'simulation_authorized': False, 'matrix_mapping_status': 'unresolved', 'expected_variable_set': sorted(layout.names), 'expected_mna_dimension': 38, 'expected_independent_dynamic_states': 32, 'scope': 'prepared extraction instructions only; no real circuit measured'}
