"""Complete local small-signal G+sC audit using ngspice's full MNA Jacobian.

For the reviewed quasistatic BSIM4.5 circuit, constant-bias backward-Euler
Jacobian is G+alpha*C. All internal voltage states are retained. Only exact
ideal independent-voltage constraints and their branch currents are eliminated.
No transfer-function fit, root deletion or PDK modification is used.
"""
from pathlib import Path
import hashlib, json, re
import numpy as np
from scipy.linalg import eig
from rc_agent.stability import DYNAMIC_NODES
from rc_agent.poles import dc_terminated_prefix
VERSION = 'full_mna_gc_v1'

def matrix_deck(deck):
    prefix = dc_terminated_prefix(deck)
    prefix = '\n'.join((x for x in prefix.splitlines() if not re.match('^(Vloop|Iport)\\b', x, re.I))) + '\n'
    prefix = re.sub('\\bampout\\b', 'out', prefix)
    control = '.control\noption klu method=gear maxord=1 reltol=1e-9 abstol=1e-16 vntol=1e-12\nset numdgt=15\nset wr_singlescale\nset wr_vecnames\nop\nprint all > op.txt\nmdump\n'
    for tag, t in [('medium', 1e-09), ('fast', 1e-12)]:
        control += f'tran {t / 100:.17g} {t:.17g}\nmdump\nwrdata steady_{tag}.dat all\n'
    return prefix + control + 'alter Vin acmag=1\nac dec 20 1 1e14\nlet hr=real(v(out))\nlet hi=imag(v(out))\nwrdata matrix_ac.dat hr hi\nquit\n.endc\n.end\n'

def parse_matrices(log, expected=3):
    triples = [(int(r), int(c), float(v)) for r, c, v in re.findall('Row:\\s*(\\d+)\\s+Col:\\s*(\\d+)\\s+Value:\\s*([-+0-9.eE]+)', log)]
    if not triples:
        raise ValueError('Full matrix dump absent')
    blocks = []
    block = []
    seen = set()
    for r, c, v in triples:
        if r < 1 or c < 1 or (not np.isfinite(v)):
            raise ValueError('Invalid matrix coordinate/value')
        if (r, c) in seen:
            blocks.append(block)
            block = []
            seen = set()
        block.append((r, c, v))
        seen.add((r, c))
    if block:
        blocks.append(block)
    if len(blocks) != expected:
        raise ValueError('Unexpected matrix dump count')
    coordinates = [{(r, c) for r, c, _ in b} for b in blocks]
    if any((c != coordinates[0] for c in coordinates)):
        raise ValueError('MNA structural coordinates changed')
    n = max((max(r, c) for r, c, _ in triples))
    mat = []
    if set(range(1, n + 1)) != {r for r, _, _ in triples}:
        raise ValueError('MNA rows missing')
    for block in blocks:
        a = np.zeros((n, n))
        for r, c, v in block:
            a[r - 1, c - 1] = v
        mat.append(a)
    return mat

def ideal_source_projection(g, c):
    n = len(g)
    branches = [i for i in range(n) if np.count_nonzero(g[i]) == 1 and np.count_nonzero(c[i]) == 0 and (np.max(abs(g[i])) == 1) and (g[i, i] == 0)]
    nodes = [int(np.flatnonzero(g[i])[0]) for i in branches]
    if not branches or len(set(nodes)) != len(nodes) or set(nodes) & set(branches):
        raise ValueError('Unsupported/ambiguous ideal-source constraints')
    for i, j in zip(branches, nodes):
        if np.count_nonzero(g[:, i]) != 1 or g[j, i] != g[i, j] or np.count_nonzero(c[:, i]) != 0:
            raise ValueError('Source branch is not a pure algebraic incidence')
    retained = [i for i in range(n) if i not in set(branches + nodes)]
    return (retained, branches, nodes)

def complete_modes(g, c):
    retained, branches, nodes = ideal_source_projection(g, c)
    gr = g[np.ix_(retained, retained)]
    cr = c[np.ix_(retained, retained)]
    sv = np.linalg.svd(cr, compute_uv=False)
    if sv[-1] <= sv[0] * 1e-12:
        raise ValueError('Independent dynamic charge matrix rank/conditioning unresolved')
    spectra = []
    residuals = []
    for sf in (10000.0, 100000000.0):
        row = np.maximum(np.max(abs(gr), axis=1), np.max(abs(cr) * sf, axis=1))
        w, v = eig(-gr / row[:, None], cr * sf / row[:, None], homogeneous_eigvals=True)
        if np.any(w[1] == 0):
            raise ValueError('Unexplained infinite mode after exact source-constraint elimination')
        p = w[0] / w[1] * sf
        if not np.all(np.isfinite(p)):
            raise ValueError('Nonfinite independent mode')
        residual = max((np.linalg.norm(gr @ vec + s * cr @ vec) / ((np.linalg.norm(gr) + abs(s) * np.linalg.norm(cr)) * np.linalg.norm(vec)) for s, vec in zip(p, v.T)))
        spectra.append(p)
        residuals.append(float(residual))
    distance = max((min(abs(s - spectra[1])) / max(abs(s), 1) for s in spectra[0]))
    if distance > 1e-05 or max(residuals) > 1e-09:
        raise ValueError('Generalized eigenpair residual/scale consistency failed')
    return (spectra[-1], {'mna_dimension': len(g), 'independent_dynamic_states': len(retained), 'algebraic_source_nodes_1based': [x + 1 for x in nodes], 'algebraic_source_branches_1based': [x + 1 for x in branches], 'charge_condition_number': float(sv[0] / sv[-1]), 'eigensolver_relative_match': float(distance), 'maximum_eigenpair_residual': max(residuals), 'discarded_dynamic_modes': 0})

def model_contract(pdk):
    folder = Path(pdk).parents[2] / 'libs.ref/sky130_fd_pr/spice'
    rows = []
    expected = {'level': 54.0, 'version': 4.5, 'rbodymod': 1.0, 'rgatemod': 0.0, 'rdsmod': 0.0, 'acnqsmod': 0.0, 'trnqsmod': 0.0}
    for kind in ('nfet', 'pfet'):
        p = folder / f'sky130_fd_pr__{kind}_01v8__tt.pm3.spice'
        raw = p.read_bytes()
        text = raw.decode()
        actual = {}
        for k, v in expected.items():
            values = {float(x) for x in re.findall('(?im)^\\s*\\+\\s*' + k + '\\s*=\\s*([-+0-9.eE]+)', text)}
            if values != {v}:
                raise ValueError('Installed model is outside reviewed quasistatic full-state scope: ' + k)
            actual[k] = v
        rows.append({'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'options': actual})
    return rows

def analyze_dut(directory, p, pdk, log=None):
    directory = Path(directory)
    log = (directory / 'ngspice.log').read_text() if log is None else log
    result = {'status': 'unknown', 'version': VERSION, 'scope': 'complete local small-signal numerical modes, pinned TT model and measured DC point', 'formal_global_certificate': 'not_claimed'}
    try:
        result['model_contract'] = model_contract(pdk)
        if re.search('(?i)(warning:|error:|singular matrix|convergence failed)', log):
            raise ValueError('Matrix extraction has simulator warning/error')
        g, *js = parse_matrices(log)
        op = {name: float(value) for name, value in re.findall('(?m)^([^\\s=]+)\\s*=\\s*([-+0-9.eE]+)\\s*$', (directory / 'op.txt').read_text())}
        expected = set(DYNAMIC_NODES) | {'vdd', 'in', 'vdd#branch', 'vin#branch'}
        if set(op) != expected or len(op) != len(g):
            raise ValueError('Independent/internal state coverage mismatch')
        a_candidates = np.flatnonzero(np.isclose(np.diag(g), 1 / p.r1 + 1 / p.r2, rtol=2e-07, atol=0))
        if len(a_candidates) != 1:
            raise ValueError('Cannot uniquely identify passive intermediate node')
        ai = int(a_candidates[0])
        cs = []
        spectra = []
        details = []
        drifts = []
        for j, tag in zip(js, ('medium', 'fast')):
            alpha = (j[ai, ai] - g[ai, ai]) / p.cf
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError('Invalid integration coefficient')
            c = (j - g) / alpha
            cs.append(c)
            poles, detail = complete_modes(g, c)
            spectra.append(poles)
            detail['integration_alpha_per_s'] = float(alpha)
            details.append(detail)
            path = directory / f'steady_{tag}.dat'
            names = path.read_text().splitlines()[0].split()
            data = np.loadtxt(path, skiprows=1)
            dr = []
            for name, value in op.items():
                if '#branch' in name:
                    continue
                if name not in names:
                    raise ValueError('Constant-bias trajectory lacks internal node ' + name)
                dr.append(np.max(abs(data[:, names.index(name)] - value)))
            drifts.append(float(max(dr)))
        if max(drifts) > 1e-07:
            raise ValueError('Jacobian recovery did not preserve the DC point')
        distance = max((min(abs(s - spectra[1])) / max(abs(s), 1) for s in spectra[0]))
        cdiff = np.linalg.norm(cs[0] - cs[1]) / np.linalg.norm(cs[1])
        if distance > 0.001 or cdiff > 1e-05:
            raise ValueError('Two-step matrix/mode consistency failed')
        c = cs[-1]
        poles = spectra[-1]
        outidx = int(np.argmin(c[ai]))
        retained, branches, nodes = ideal_source_projection(g, c)
        vin = [br for br, node in zip(branches, nodes) if g[ai, node] < 0]
        if len(vin) != 1 or c[ai, outidx] >= 0 or abs(c[ai, outidx] / p.cf + 1) > 1e-06:
            raise ValueError('Input/output mapping inconsistent with explicit Cf')
        rhs = np.zeros(len(g))
        rhs[vin[0]] = 1
        raw = np.loadtxt(directory / 'matrix_ac.dat', skiprows=1)
        f = raw[:, 0]
        h = raw[:, 1] + 1j * raw[:, 2]
        pred = np.array([np.linalg.solve(g + 2j * np.pi * ff * c, rhs)[outidx] for ff in f])
        error = float(np.max(abs(pred - h) / np.maximum(abs(h), 0.01)))
        if error > 0.0001:
            raise ValueError('Full matrix does not reproduce independent full-circuit AC sweep')
        result.update(op_dc_out_v=op['out'], matrix_details=details, op_vector_names=sorted(op), maximum_dc_drift_v=max(drifts), charge_matrix_relative_difference=float(cdiff), step_size_mode_relative_match=float(distance), ac_reconstruction_error=error, ac_grid_hz=[float(f[0]), float(f[-1])], poles_rad_s=[[float(s.real), float(s.imag)] for s in sorted(poles, key=abs)], max_pole_real_rad_s=float(max(poles.real)), finite_mode_count=len(poles), discarded_dynamic_modes=0)
        if any((s.real >= 0 for s in poles)):
            result.update(status='fail', reason='Resolved non-LHP complete local mode')
        elif any((-s.real / abs(s) < 1e-05 for s in poles)):
            result['reason'] = 'Numerically marginal damping needs review'
        else:
            result.update(status='pass', reason='All independent dynamic modes LHP; full-state/DC/matrix/eigenpair/AC controls passed')
        np.savez(directory / 'full_mna.npz', g=g, c_medium=cs[0], c_fast=cs[1], poles=poles)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        result['reason'] = str(exc)
    (directory / 'linear_stability.json').write_text(json.dumps(result, indent=2))
    return result
