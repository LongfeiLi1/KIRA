"""Sallen-Key initialization command line."""
import argparse, json
from pathlib import Path
from api import demo_spec, initialize

def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo', action='store_true', help='Calculate a feasible Sallen-Key example')
    g.add_argument('--spec', type=Path, help='Caller-owned SK specification JSON')
    p.add_argument('--topology', choices=['lowpass', 'highpass'], default='lowpass', help='Topology for --demo')
    p.add_argument('--arm', choices=['full', 'no_initialization'], default='full')
    a = p.parse_args()
    row = demo_spec(a.topology) if a.demo else json.loads(a.spec.read_text())
    print(json.dumps({'initialization': initialize(row, a.arm), 'API_started': 0, 'SPICE_started': 0, 'physical_acceptance': False}, indent=2, allow_nan=False))
if __name__ == '__main__':
    main()
