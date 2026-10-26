from runtime.paths import BENCHMARK_DIR
'Bounded T2 fixed-candidate physical evaluator, independent of live baselines.\n\nManifest admission is never changed. Candidate validation retains the original\nregistered domain; the narrower analytic-check range is not a search bound.\n'
from dataclasses import asdict
from pathlib import Path
import argparse, fcntl, hashlib, json, os, re, subprocess, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'native', ROOT / 'numerics', BENCHMARK_DIR):
    sys.path.insert(0, str(p))
import task_data as data
from rc_agent.physical import Devices, PDK, op_commands, scalar, table, loop_metrics
from rc_agent.stability import port_commands, two_port_return
from rc_agent.linear_stability import parse_matrices, model_contract
from numerics.topology_stability.core import Layout, GroundedSource, ChargeAnchor, Ports, check_linear_evidence
from numerics.topology_stability.decks import build_matrix_deck, ExtractionPorts
from numerics.topology_stability.response_mapping import map_response_states
from numerics.topology_stability.sk_reference import BODY_NODES
from numerics.metrics.response import evaluate_response
from numerics.hp_validation.evidence import highpass_step_metrics
BASE = ROOT / 'numerics/physical'
SPICE = Path('ngspice')
SOURCES = (GroundedSource('vin#branch', 'in'), GroundedSource('vdd#branch', 'vdd'), GroundedSource('vref#branch', 'vref'))
SIMULATOR_FAILURE = re.compile('(?im)(^\\s*(warning:|error:|fatal|singular matrix|convergence failed)|timestep too small|simulation\\(s\\) aborted)')

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contract(case, p, names=None):
    middle = 'y'
    dynamic = ('vbias', 'xu.s1o', middle, 'xu.tail', 'xu.s1x', 'out', 'xu.z', 'x') + BODY_NODES
    expected = dynamic + tuple((n for s in SOURCES for n in (s.node, s.branch)))
    layout = Layout(tuple(names or expected), dynamic, SOURCES)
    layout.validate()
    anchors = (ChargeAnchor('x', 'x', p['c1'] + p['c2']), ChargeAnchor('x', 'in', -p['c1']), ChargeAnchor('x', 'y', -p['c2']))
    return (layout, Ports({'vin#branch': 1.0}, {'out': 1.0}), anchors)

def validate_candidate(case, p, devices):
    data.render_real_ota_prefix(case, p, devices, closed_dc_only=True)

def matrix_deck(case, p, devices):
    layout, _, _ = contract(case, p)
    prefix = data.render_real_ota_prefix(case, p, devices, closed_dc_only=True)
    times = (1e-09, 1e-12)
    deck = build_matrix_deck(prefix, layout, ExtractionPorts('Vin', 'out'), times_s=times).split('alter Vin acmag=1')[0]
    for source in SOURCES:
        for s in SOURCES:
            deck += f'alter {s.branch[:-7]} acmag={int(s == source)}\n'
        deck += 'ac dec 20 1 1e14\n'
        for j, name in enumerate(layout.names):
            expr = f'i({name[:-7]})' if name.endswith('#branch') else f'v({name})'
            deck += f'let mr{j}=real({expr})\nlet mi{j}=imag({expr})\n'
        deck += f'wrdata mapping_{source.branch[:-7]}.dat ' + ' '.join((f'mr{j} mi{j}' for j in range(len(layout.names)))) + '\n'
    return deck + 'quit\n.endc\n.end\n'

def performance_deck(case, p, devices):
    stop = 12 / case['f0_hz']
    delay = 1 / case['f0_hz']
    prefix = data.render_real_ota_prefix(case, p, devices)
    prefix = prefix.replace('Vin in 0 DC 1.2 AC 1', f'Vin in 0 DC 1.2 AC 1 PULSE(1.2 1.201 {delay:.17g} {delay / 100:.17g} {delay / 100:.17g} {stop * 2:.17g} {stop * 4:.17g})')
    return prefix + '.control\nset wr_singlescale\nset wr_vecnames\nset numdgt=15\n' + op_commands() + '\nlet vref_power_w=-i(Vref)*1.2\nlet vin_power_w=-i(Vin)*1.2\nprint vref_power_w vin_power_w\nac dec 80 1 1e8\nlet hr=real(v(out))\nlet hi=imag(v(out))\nwrdata filter_ac.dat hr hi\n' + port_commands() + f'\nalter Vloop acmag=0\nalter Iport acmag=0\ntran {stop / 3000:.17g} {stop:.17g}\nwrdata step.dat v(out)\nquit\n.endc\n.end\n'

def run_spice(deck, folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(BASE / 'evidence'):
        raise ValueError('exclusive evidence path required')
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'circuit.cir').write_text(deck)
    with (BASE / 'execution.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = []
        for receipt in (BASE / 'evidence').rglob('receipt.json'):
            row = json.loads(receipt.read_text())
            if row.get('spice_processes'):
                records.append(row)
        budget = json.loads((BASE / 'budget.json').read_text())
        if len(records) >= budget['maximum_spice_processes']:
            raise RuntimeError('bounded development simulation budget exhausted')
        receipt = {'status': 'reserved', 'spice_processes': 1, 'max_seconds': 60, 'netlist_sha256': sha(folder / 'circuit.cir'), 'simulator_sha256': sha(SPICE), 'invocation_number': len(records) + 1}
        save(folder / 'receipt.json', receipt)
        start = time.monotonic()
        try:
            proc = subprocess.run([str(SPICE), '-n', '-b', 'circuit.cir'], cwd=folder, capture_output=True, text=True, timeout=60, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
            out, err, code = (proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired as exc:
            decode = lambda value: value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            out, err, code = (decode(exc.stdout), decode(exc.stderr), 'timeout')
        except OSError as exc:
            out, err, code = ('', str(exc), 'launch_error')
        (folder / 'stdout.log').write_text(out)
        (folder / 'stderr.log').write_text(err)
        log = out + '\n' + err
        (folder / 'ngspice.log').write_text(log)
        bad = code != 0 or bool(SIMULATOR_FAILURE.search(log))
        receipt.update(exit_code=code, wall_seconds=time.monotonic() - start, status='failed' if bad else 'complete', simulator_diagnostic_failure=bool(SIMULATOR_FAILURE.search(log)))
        save(folder / 'receipt.json', receipt)
    if code != 0 or SIMULATOR_FAILURE.search(log):
        raise RuntimeError('simulator extraction failure: ' + str(folder))
    return log

def analyze_matrix(folder, case, p):
    log = (folder / 'ngspice.log').read_text()
    if SIMULATOR_FAILURE.search(log):
        raise ValueError('matrix simulator warning/error')
    g, *js = parse_matrices(log)
    layout, _, _ = contract(case, p)
    arrays = [table(folder, f'mapping_{s.branch[:-7]}.dat', 1 + 2 * len(layout.names)) for s in SOURCES]
    f = arrays[0][:, 0]
    if any((not np.array_equal(a[:, 0], f) for a in arrays)):
        raise ValueError('mapping frequency grids mismatch')
    observed = np.array([a[:, 1::2] + 1j * a[:, 2::2] for a in arrays])
    trajectories = []
    cs = []
    intervals = []
    requested = [float(x) for x in re.findall('(?im)^tran\\s+\\S+\\s+(\\S+)', (folder / 'circuit.cir').read_text())]
    if len(requested) != 2:
        raise ValueError('two explicit BE trajectories required')
    voltage_names = set(layout.names) - {s.branch for s in SOURCES}
    for tag, j in zip(('medium', 'fast'), js):
        path = folder / f'steady_{tag}.dat'
        columns = path.read_text().splitlines()[0].split()
        raw = table(folder, path.name, len(columns))
        if not np.isclose(raw[-1, 0], requested[len(intervals)], rtol=1e-12, atol=0):
            raise ValueError('incomplete BE trajectory')
        dt = raw[-1, 0] - raw[-2, 0]
        intervals.append(float(dt))
        cs.append((j - g) * dt)
        trajectories.append({name: raw[:, columns.index(name)] for name in voltage_names})
    if np.linalg.norm(cs[0] - cs[1]) / np.linalg.norm(cs[1]) > 1e-05:
        raise ValueError('bootstrap BE interval consistency failed')
    names, proof = map_response_states(g, cs[-1], layout.names, [s.branch for s in SOURCES], f, observed)
    proof.update(netlist_sha256=sha(folder / 'circuit.cir'), simulator_sha256=sha(SPICE), accepted_BE_intervals_s=intervals)
    save(folder / 'mapping_proof.json', proof)
    layout, ports, anchors = contract(case, p, names)
    optext = (folder / 'op.txt').read_text()
    op = {name: scalar(optext, name) for name in voltage_names}
    h = observed[0, :, contract(case, p)[0].names.index('out')]
    linear = check_linear_evidence(g, js, layout, names, anchors, ports, op, trajectories, f, h)
    linear.update(op_dc_out_v=op['out'], model_contract=model_contract(PDK), mapping=proof)
    save(folder / 'linear_evidence.json', linear)
    return linear

def analyze_performance(folder, case):
    common = json.loads((ROOT / 'resources/configs/sallen_key.json').read_text())['common_acceptance']
    log = (folder / 'ngspice.log').read_text()
    if SIMULATOR_FAILURE.search(log):
        raise ValueError('performance simulator warning/error')
    a = table(folder, 'filter_ac.dat', 3)
    f = a[:, 0]
    h = a[:, 1] + 1j * a[:, 2]
    response = evaluate_response(f, h, data.response_profile(case), fit_max_component_error=common['fit_max_component_error'])
    op = {'dc_out_v': scalar(log, 'dc_out'), 'power_w': scalar(log, 'power_w'), 'minimum_saturation_margin_v': min((scalar(log, 'margin' + str(i)) for i in range(1, 8))), 'vref_power_w': scalar(log, 'vref_power_w'), 'vin_power_w': scalar(log, 'vin_power_w')}
    op['power_method'] = 'unchanged v1 VDD quiescent supply including bias Iref; Vref and signal source recorded separately'
    step = table(folder, 'step.dat', 2)
    stop = 12 / case['f0_hz']
    delay = 1 / case['f0_hz']
    transient = highpass_step_metrics(step[:, 0], step[:, 1], dc_out_v=op['dc_out_v'], input_edge_time_s=delay, tail_start_s=float(step[-100, 0]), final_error_limit_v=common['step_final_error_max_v'], tail_p2p_limit_v=common['step_tail_peak_to_peak_max_v'])
    va = table(folder, 'port_voltage.dat', 7)
    ia = table(folder, 'port_current.dat', 7)
    if not np.array_equal(va[:, 0], ia[:, 0]):
        raise ValueError('port excitation grids mismatch')
    ret, condition = two_port_return(va[:, 0], va[:, 1:], ia[:, 1:])
    port = loop_metrics(va[:, 0], ret)
    port['excitation_condition'] = condition
    port['inherited_phase_margin_min_deg'] = 60.0
    failures = list(response['failed_constraints'])
    lo, hi = common['dc_out_range_v']
    if not lo <= op['dc_out_v'] <= hi:
        failures.append('dc_out')
    if op['power_w'] > case['power_max_uW'] * 1e-06:
        failures.append('power')
    if op['minimum_saturation_margin_v'] < common['minimum_saturation_margin_v']:
        failures.append('saturation_margin')
    if transient['step_status'] != 'pass':
        failures.append('step')
    if port['phase_margin_deg'] < 60:
        failures.append('phase_margin')
    if abs(port['sampled_nyquist_turns']) >= 0.1:
        failures.append('sampled_port_nyquist')
    return {'response': response, 'operating_point': op, 'step': transient, 'port': port, 'failed_constraints': failures}

def evaluate(case_id, *, passives=None, devices=None, output_name=None, matrix_evidence=None, performance_evidence=None):
    """One candidate; caller supplies values in original registered physical bounds.

    This is the shared T2 development execution interface. It is not an
    optimizer-domain admission and cannot set manifest runnable=true.
    """
    case = data.load_case(case_id)
    if case['type_id'] not in ('T2',):
        raise ValueError('exclusive T2 adapter')
    p = data.ideal_seed(case) if passives is None else dict(passives)
    d = Devices() if devices is None else devices
    validate_candidate(case, p, d)
    label = output_name or case_id
    if re.fullmatch('[A-Za-z0-9_.-]+', label) is None:
        raise ValueError('safe unique output label required')
    folder = BASE / 'evidence' / label
    folder.mkdir(exist_ok=False)
    result = {'case_id': case_id, 'passives': p, 'devices': asdict(d), 'runnable': False, 'optimizer_admission': False, 'source_hashes': data.source_record(), 'binding_audit': data.binding_report(case), 'measurement_status': 'unverified', 'candidate_status': 'unknown', 'errors': []}
    save(folder / 'candidate.json', result)
    try:
        performance = folder / 'performance' if performance_evidence is None else Path(performance_evidence)
        expected = performance_deck(case, p, d)
        if performance_evidence is None:
            run_spice(expected, performance)
        elif (performance / 'circuit.cir').read_text() != expected:
            raise ValueError('reused performance deck does not exactly match this candidate')
        result['performance'] = analyze_performance(performance, case)
        result['performance_evidence'] = str(performance.relative_to(ROOT))
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as exc:
        result['errors'].append({'stage': 'performance_measurement', 'classification': 'measurement_error', 'reason': str(exc)})
    try:
        matrix = folder / 'matrix' if matrix_evidence is None else Path(matrix_evidence)
        expected = matrix_deck(case, p, d)
        if matrix_evidence is None:
            run_spice(expected, matrix)
        elif (matrix / 'circuit.cir').read_text() != expected:
            raise ValueError('reused matrix deck does not exactly match this candidate')
        result['linear'] = analyze_matrix(matrix, case, p)
        result['matrix_evidence'] = str(matrix.relative_to(ROOT))
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as exc:
        result['errors'].append({'stage': 'complete_mna', 'classification': 'measurement_error', 'reason': str(exc)})
    if not result['errors']:
        failures = list(result['performance']['failed_constraints'])
        linear = result['linear']
        dc = result['performance']['operating_point']['dc_out_v']
        if abs(linear['op_dc_out_v'] - dc) > 1e-07:
            result['errors'].append({'stage': 'operating_point_consistency', 'classification': 'measurement_error', 'reason': 'performance and closed-MNA operating points differ'})
        elif linear['status'] == 'unknown':
            result['errors'].append({'stage': 'complete_mna', 'classification': 'measurement_error', 'reason': 'unresolved local mode classification'})
        else:
            if linear['status'] == 'unstable':
                failures.append('full_local_stability')
            result.update(measurement_status='valid', candidate_status='pass' if not failures else 'circuit_fail', failed_constraints=failures)
    if result['errors']:
        result.update(measurement_status='error', candidate_status='unclassified_due_to_measurement_error')
    result['manifest_sha256_after'] = sha(BENCHMARK_DIR / 'manifest.json')
    if result['manifest_sha256_after'] != data.MANIFEST_SHA:
        raise RuntimeError('manifest was changed externally')
    save(folder / 'result.json', result)
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case_id', choices=sorted(PILOTS))
    parser.add_argument('--matrix-evidence')
    parser.add_argument('--output-name')
    args = parser.parse_args()
    result = evaluate(args.case_id, output_name=args.output_name, matrix_evidence=args.matrix_evidence)
    print(json.dumps({k: result.get(k) for k in ('case_id', 'measurement_status', 'candidate_status', 'failed_constraints', 'errors')}))
