"""Build a bounded extraction deck from an explicitly CLOSED, DC-only prefix.

No implicit probe removal, source termination, node renaming or topology change.
No simulator invocation is provided. Unsupported source syntax fails closed.
"""
import re
from dataclasses import dataclass
import math
from .core import ContractError, Layout

@dataclass(frozen=True)
class ExtractionPorts:
    input_source: str
    output_node: str

def build_matrix_deck(prefix: str, layout: Layout, ports: ExtractionPorts, times_s=(1e-09, 1e-12)):
    layout.validate()
    ident = re.compile('^[A-Za-z_][A-Za-z0-9_.#]*$')
    if not ident.fullmatch(ports.input_source) or not ident.fullmatch(ports.output_node):
        raise ContractError('unsafe/unsupported SPICE port identifier')
    if ports.output_node not in layout.names or ports.output_node in {s.branch for s in layout.sources}:
        raise ContractError('output node absent from voltage layout')
    if len(times_s) != 2 or len(set(times_s)) != 2 or any((not math.isfinite(t) or t <= 0 for t in times_s)):
        raise ContractError('two distinct positive extraction time steps required')
    lines = prefix.splitlines()
    if not lines or not lines[0].lstrip().startswith('*'):
        raise ContractError('explicit SPICE title comment required')
    sources, depth = ({}, 0)
    number = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'
    for line in lines[1:]:
        text = line.strip()
        if not text or text.startswith('*'):
            continue
        if re.match('(?i)^\\.(control|endc|end|ac|tran|op|pz|tf|save|print|measure)\\b', text):
            raise ContractError('prefix must contain circuit statements only')
        if re.match('(?i)^\\.subckt\\b', text):
            depth += 1
        elif re.match('(?i)^\\.ends\\b', text):
            depth -= 1
            if depth < 0:
                raise ContractError('unbalanced subcircuit')
        elif re.match('(?i)^[VI]\\S*\\s', text):
            match = re.fullmatch('([VI]\\S*)\\s+(\\S+)\\s+(\\S+)\\s+(?:DC\\s+)?(' + number + ')', text, re.I)
            if match is None:
                raise ContractError('DC-only numeric independent source required: ' + text.split()[0])
            name, pos, neg, value = match.groups()
            if not math.isfinite(float(value)):
                raise ContractError('nonfinite source')
            if depth == 0:
                if name.lower() in sources:
                    raise ContractError('duplicate independent source')
                sources[name.lower()] = (pos, neg)
        elif text.startswith('+'):
            raise ContractError('continuations require explicit prefix review')
    if depth:
        raise ContractError('unbalanced subcircuit')
    expected = {}
    for source in layout.sources:
        if not source.branch.endswith('#branch'):
            raise ContractError('SPICE source branch name must end #branch')
        name = source.branch[:-7].lower()
        expected[name] = (source.node, '0') if source.incidence == 1 else ('0', source.node)
    actual = {k: v for k, v in sources.items() if k.startswith('v')}
    if actual != expected:
        raise ContractError('explicit source incidence differs from closed DC prefix')
    if ports.input_source.lower() not in actual:
        raise ContractError('input must name a declared independent voltage source')
    controls = ['.control', 'option klu method=gear maxord=1 reltol=1e-9 abstol=1e-16 vntol=1e-12', 'set numdgt=15', 'set wr_singlescale', 'set wr_vecnames', 'op', 'print all > op.txt', 'mdump']
    for tag, t in zip(('medium', 'fast'), times_s):
        controls += [f'tran {t / 100:.17g} {t:.17g}', 'mdump', f'wrdata steady_{tag}.dat all']
    controls += [f'alter {ports.input_source} acmag=1', 'ac dec 20 1 1e14', f'let hr=real(v({ports.output_node}))', f'let hi=imag(v({ports.output_node}))', 'wrdata matrix_ac.dat hr hi', 'quit', '.endc', '.end']
    return prefix.rstrip() + '\n' + '\n'.join(controls) + '\n'
