"""Data-side v3 adapters. No simulator, optimizer, scheduler, or release authority."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from runtime.paths import BENCHMARK_DIR as HERE
from runtime.paths import CODE_ROOT as PROJECT
from numerics.metrics.response import ResponseProfile
from numerics.topologies.registry import get_topology, validate_passives
MANIFEST_SHA = 'd8eda4b7922fdca1b92fb37ccc0a656f7a07fb731cac132bba8545167bba5c77'
TOPOLOGY_IDS = {'T1': 'sallen_key_lowpass', 'T2': 'sallen_key_highpass'}

def manifest():
    raw = (HERE / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('Authoritative manifest integrity mismatch')
    return json.loads(raw)

def load_case(case_id):
    if re.fullmatch('V3-T[12]-L[123]-C0[1-7]', case_id) is None:
        raise ValueError('Invalid v3 Case identifier')
    _, type_id, level, case = case_id.split('-')
    path = HERE / type_id / level / case / 'runtime.json'
    runtime = json.loads(path.read_bytes())
    authoritative = manifest()
    expected = next((c for c in authoritative['cases'] if c['case_id'] == case_id), None)
    if expected is None or runtime['case'] != expected:
        raise ValueError('Runtime Case differs from authoritative manifest')
    if runtime['topology'] != authoritative['topologies'][type_id]:
        raise ValueError('Runtime topology differs from authoritative manifest')
    if runtime['acceptance_profiles'] != authoritative['acceptance_profiles'] or runtime['conditions_and_domain_policy'] != authoritative['conditions_and_domain_policy']:
        raise ValueError('Runtime acceptance/condition policy differs')
    if runtime['runnable'] is not False or expected['runnable'] is not False:
        raise ValueError('Data adapter cannot promote physical admission')
    return expected

def response_profile(case):
    hp = case['type_id'] == 'T2'
    expected_reference = 'finite_frequency_magnitude' if hp else 'first_ac_sample_magnitude_DC_proxy'
    if case['gain_reference_kind'] != expected_reference:
        raise ValueError('Wrong response-reference policy')
    if not hp and case['gain_reference_frequency_hz'] != 1.0:
        raise ValueError('Legacy lowpass reference must stay at 1 Hz')
    return ResponseProfile(case['case_id'], 'highpass' if hp else 'lowpass', case['target_polarity'], case['f0_hz'], case['q_target'], case['gain_reference_frequency_hz'], (case['passband_low_hz'], case['passband_high_hz']), (case['stopband_low_hz'], case['stopband_high_hz']), case['f0_relative_tolerance'], case['q_relative_tolerance'], (case['gain_magnitude_min'], case['gain_magnitude_max']), case['passband_droop_max_db'], case['stopband_attenuation_min_db'])

def legacy_lowpass_task(case):
    """Exact v3 T1 task values; caller retains its existing shared evaluator."""
    if case['type_id'] != 'T1':
        raise ValueError('Legacy lowpass evaluator bridge supports T1 only')
    return {'id': case['case_id'], 'f0_hz': case['f0_hz'], 'q': case['q_target'], 'f0_relative_tolerance': case['f0_relative_tolerance'], 'q_relative_tolerance': case['q_relative_tolerance'], 'passband_edge_hz': case['passband_high_hz'], 'stopband_edge_hz': case['stopband_low_hz'], 'passband_max_loss_db': case['passband_droop_max_db'], 'stopband_min_attenuation_db': case['stopband_attenuation_min_db'], 'load_f': case['load_cap_pF'] * 1e-12, 'power_limit_w': case['power_max_uW'] * 1e-06}

def source_record():
    paths = ['numerics/topologies/registry.py', 'numerics/metrics/response.py', 'native/rc_agent/physical.py', 'resources/configs/sallen_key.json', 'numerics/hp_validation/contract.py', 'numerics/topology_stability/core.py']
    return {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in paths}

def binding_report(case):
    t = get_topology(TOPOLOGY_IDS[case['type_id']])
    declared = manifest()['topologies'][case['type_id']]
    frozen = json.loads((PROJECT / 'resources/configs/sallen_key.json').read_bytes())
    if any((list(frozen['physical_bounds'][old]) != t.physical_bounds[new] for new, old in t.bounds_mapping)):
        raise ValueError('Registered bound differs from frozen v1 source')
    node_alias = {'Vin': 'in', 'Vref': 'vref'}
    expected_elements = [tuple([line.split()[0]] + [node_alias.get(n, n) for n in line.split()[1:]]) for line in declared['elements']]
    if expected_elements != [tuple(row[:3]) for row in t.elements]:
        raise ValueError('Topology graph differs from authoritative manifest')
    ports = declared['ota_ports']
    if [node_alias.get(ports['plus'], ports['plus']), node_alias.get(ports['minus'], ports['minus'])] != [t.ota_plus, t.ota_minus]:
        raise ValueError('OTA ports differ from authoritative manifest')
    analytic = declared['domain_used_for_analytic_check']
    capacitor_domains = {'T1': {'cg': 'Cg_f', 'cf': 'Cf_f'}, 'T2': {'c1': 'C1_f', 'c2': 'C2_f'}}[case['type_id']]
    differences = [{'parameter': name, 'analytic_domain': analytic[key], 'frozen_adapter_domain': t.physical_bounds[name]} for name, key in capacitor_domains.items() if analytic[key] != t.physical_bounds[name]]
    return {'case_id': case['case_id'], 'topology_id': t.topology_id, 'normalized_topology_sha256': t.structure_hash, 'frozen_physical_bounds': t.physical_bounds, 'bounds_source_mapping': dict(t.bounds_mapping), 'analytic_vs_frozen_domain_differences': differences, 'domain_resolution': 'analytic check domain is a narrower subset, not a search-bound override; retain original registered mapping to frozen physical bounds', 'actual_registered_search_domain': t.physical_bounds, 'topology_binding_freeze_status': 'original T2 registration was proposal_pending_type_level_review' if case['type_id'] in ('T2',) else 'existing T1 binding', 'domain_provenance_receipt': 'numerics/physical/domain_provenance.json', 'parameter_independence': 'not inferred from bounds mapping; optimizer variable bindings require coordinator validation', 'runnable': False, 'physical_validation_status': 'unverified'}

def ideal_seed(case):
    qa = next((r for r in manifest()['ideal_analytic_checks'] if r['case_id'] == case['case_id']))
    mapping = {'R1_ohm': 'r1', 'R2_ohm': 'r2', 'Cg_f': 'cg', 'Cf_f': 'cf', 'C1_f': 'c1', 'C2_f': 'c2', 'Rf_ohm': 'rf', 'Rg_ohm': 'rg'}
    seed = {mapping[k]: v for k, v in qa['ideal_seed'].items()}
    return validate_passives(TOPOLOGY_IDS[case['type_id']], seed, enforce_bounds=True)

def render_real_ota_prefix(case, passives, devices, *, closed_dc_only=False):
    """Render explicit real SKY130 OTA wiring for inspection; never execute it.

    Renders only the two registered Sallen-Key node graphs.
    This function grants no physical admission or optimizer-variable bindings.
    """
    binding_report(case)
    t = get_topology(TOPOLOGY_IDS[case['type_id']])
    p = validate_passives(t.topology_id, passives, enforce_bounds=True)
    frozen = json.loads((PROJECT / 'resources/configs/sallen_key.json').read_bytes())
    for name, value in asdict(devices).items():
        if not math.isfinite(value):
            raise ValueError('Nonfinite device parameter')
        if name == 'rz_ohm':
            if value != frozen['initial'][name]:
                raise ValueError('Frozen Rz must not change')
        else:
            lo, hi = frozen['physical_bounds'][name]
            if not lo <= value <= hi:
                raise ValueError('Device parameter outside frozen domain')
    from rc_agent.physical import header
    output = 'out' if closed_dc_only else 'ampout'
    lines = [header(devices), 'Vdd vdd 0 1.8', 'Vref vref 0 1.2' if closed_dc_only else 'Vref vref 0 DC 1.2 AC 0', 'Vin in 0 1.2' if closed_dc_only else 'Vin in 0 DC 1.2 AC 1', 'XB vdd 0 vbias bias', f'XU {t.ota_plus} {t.ota_minus} {output} vdd 0 vbias ota']
    if not closed_dc_only:
        lines += ['Vloop ampout out DC 0 AC 0', 'Iport 0 out DC 0 AC 0']
    lines += [f'{name} {a} {b} {p[key]:.16g}' for name, a, b, key in t.elements]
    lines += [f"CL out 0 {case['load_cap_pF'] * 1e-12:.16g}"]
    return '\n'.join(lines) + '\n'
