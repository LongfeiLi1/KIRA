"""Sallen-Key initialization and analytical response checks."""
from _paths import activate
activate()
import math
from initialization import synthesize
TOPOLOGIES = {'lowpass': ('sallen_key_lowpass', 't1_sk_lp'), 'highpass': ('sallen_key_highpass', 't2_sk_hp')}

def demo_spec(topology='lowpass'):
    if topology not in TOPOLOGIES:
        raise ValueError('Choose lowpass or highpass')
    family, kind = TOPOLOGIES[topology]
    from numerics.topologies.registry import get_topology
    return {'family': family, 'spec': {'task_id': 'sk_' + topology + '_example', 'topology': kind, 'frequency_hz': 50000.0, 'q': 2 ** (-0.5), 'gain': 1.0}, 'bounds': get_topology(family).physical_bounds}

def initialize(row, arm='full'):
    if arm not in {'full', 'no_initialization'}:
        raise ValueError('Unknown initialization arm')
    if type(row) is not dict or not {'spec', 'bounds'} <= row.keys():
        raise ValueError('spec and bounds are required')
    spec = row['spec']
    kind = spec.get('topology')
    family = {'t1_sk_lp': 'sallen_key_lowpass', 't2_sk_hp': 'sallen_key_highpass'}.get(kind)
    if family is None or row.get('family', family) != family:
        raise ValueError('Only SK lowpass/highpass specifications are available')
    receipt = synthesize(spec, row['bounds'])
    if arm == 'no_initialization':
        receipt={'schema':'kira.neutral_initialization.v1','topology':kind,
            'parameters':{k:math.sqrt(v[0]*v[1]) for k,v in row['bounds'].items()},
            'domain':row['bounds'],'strategy':'specification-blind geometric domain center',
            'online_llm_calls':0,'online_spice_processes':0,'physical_acceptance':False}
        from initialization.synthesis import digest
        receipt['receipt_sha256']=digest(receipt)
    return receipt

def analytical_response(row, frequency_hz, parameters=None):
    from numerics.topologies.registry import ideal_response
    receipt = initialize(row)
    p = receipt['parameters'] if parameters is None else parameters
    names = ('r1', 'r2', 'cg', 'cf') if row['spec']['topology'] == 't1_sk_lp' else ('c1', 'c2', 'rf', 'rg')
    family = 'sallen_key_lowpass' if row['spec']['topology'] == 't1_sk_lp' else 'sallen_key_highpass'
    return ideal_response(family, {k: p[k] for k in names}, frequency_hz)
