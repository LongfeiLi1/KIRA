"""Full-circuit two-excitation port audit and measured nodal small-signal modes.

Two-port return ratio: Tian et al., IEEE C&D 2001, Eq.20,
https://kenkundert.com/docs/cd2001-01.pdf . Port margin is not a proof that
remaining internal feedback is stable. The nodal modal check is independent.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import eig
CORE_NODES = ('vbias', 'xu.s1o', 'vp', 'xu.tail', 'xu.s1x', 'out', 'xu.z', 'a')
BODY_NODES = tuple((f"m.{('xb' if i == 8 else 'xu')}.xm{i}.msky130_fd_pr__{('pfet' if i in (3, 4, 6) else 'nfet')}_01v8#{part}" for i in range(1, 9) for part in ('body', 'dbody', 'sbody')))
DYNAMIC_NODES = CORE_NODES + BODY_NODES

def verify_node_coverage(log):
    import re
    found = set(re.findall('(?m)^([^\\s=]+)\\s*=\\s*[-+0-9.eE]+\\s*$', log))
    found = {x for x in found if not x.endswith('#branch')} - {'vdd', 'in', 'ampout'}
    if found != set(DYNAMIC_NODES):
        raise ValueError('Unexpected/incomplete dynamic nodes: ' + str(sorted(found ^ set(DYNAMIC_NODES))))
    return sorted(found)

def two_port_return(f, voltage, current):
    """Columns: Va real/imag,Vb real/imag,I(Vloop) real/imag.

    Vloop directed a->b; current probe directed ground->b. Infer two-port Y
    from two independent excitations. All original device/load paths remain.
    """

    def c(x):
        return x[:, 0::2] + 1j * x[:, 1::2]
    v = c(np.asarray(voltage))
    i = c(np.asarray(current))
    if v.shape != i.shape or v.shape != (len(f), 3):
        raise ValueError('Port data shape mismatch')
    values = []
    conditions = []
    for vv, ii in zip(v, i):
        vm = np.array([[vv[0], ii[0]], [vv[1], ii[1]]])
        im = np.array([[-vv[2], -ii[2]], [vv[2], ii[2] + 1]])
        if not np.all(np.isfinite(vm)) or np.linalg.cond(vm) > 1000000000000000.0:
            raise ValueError('Ill-conditioned two-port excitation')
        y = np.linalg.solve(vm.T, im.T).T
        denom = y[0, 0] + y[1, 1]
        if abs(denom) < 1e-30:
            raise ValueError('Undefined port return ratio')
        values.append((y[0, 1] + y[1, 0]) / denom)
        conditions.append(float(np.linalg.cond(vm)))
    return (np.array(values), max(conditions))

def recover_nodal_modes(f, z):
    """Identify G+sC from every node's independent current response, then eig.

    Reject incomplete/non-affine/ill-conditioned identification. This checks all
    modes of the reviewed quasistatic small-signal nodal model, not a fitted
    input-output transfer function and not a nonlinear/PVT stability proof.
    """
    f = np.asarray(f, float)
    z = np.asarray(z, complex)
    if f.ndim != 1 or len(f) < 3 or z.shape[0] != len(f) or (z.ndim != 3) or (z.shape[1] != z.shape[2]) or (not np.all(np.isfinite(z))):
        raise ValueError('Incomplete nodal response matrix')
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError('Invalid identification frequencies')
    try:
        y = np.array([np.linalg.inv(v) for v in z])
    except np.linalg.LinAlgError as e:
        raise ValueError('Nodal probes do not span all states') from e
    g = np.median(y.real, axis=0)
    c = np.median(y.imag / (2 * np.pi * f[:, None, None]), axis=0)
    predicted = g[None] + 2j * np.pi * f[:, None, None] * c[None]
    errors = [float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)) for a, b in zip(y, predicted)]
    condition = max((float(np.linalg.cond(v)) for v in z))
    if condition > 1000000000000.0 or max(errors) > 1e-07:
        raise ValueError(f'Nodal G+sC identification failed: condition={condition:.4g},relative_error={max(errors):.4g}')
    scale = 2 * np.pi * float(np.exp(np.mean(np.log(f))))
    row_scale = np.maximum(np.linalg.norm(g, axis=1) + scale * np.linalg.norm(c, axis=1), 1e-30)
    w, vec = eig(-g / row_scale[:, None], c * scale / row_scale[:, None])
    poles = w * scale
    finite = np.isfinite(poles)
    if not np.any(finite):
        raise ValueError('No finite dynamic modes found')
    residuals = []
    for s, x in zip(poles[finite], vec.T[finite]):
        residuals.append(float(np.linalg.norm((g + s * c) @ x) / (max(np.linalg.norm(g) + abs(s) * np.linalg.norm(c), 1e-30) * np.linalg.norm(x))))
    if max(residuals) > 1e-07:
        raise ValueError('Modal eigenpair verification failed')
    return ({'method': 'all reviewed dynamic nodes; current-injection admittance G+sC; generalized eigenmodes', 'identified_nodes': len(g), 'finite_modes': int(sum(finite)), 'algebraic_infinite_modes': int(sum(~finite)), 'poles_rad_s': [[float(s.real), float(s.imag)] for s in poles[finite]], 'max_pole_real_rad_s': float(np.max(poles[finite].real)), 'all_finite_modes_lhp': bool(np.all(poles[finite].real < -1.0)), 'identification_relative_error': max(errors), 'max_eigenpair_relative_residual': max(residuals), 'max_response_matrix_condition': condition, 'frequency_range_hz': [float(f[0]), float(f[-1])], 'scope': 'local linearized quasistatic circuit at measured OP, all reviewed nodes; not global nonlinear or PVT'}, g, c)

def port_commands():
    commands = ['alter Vin acmag=0', 'alter Vloop acmag=1', 'alter Iport acmag=0', 'ac dec 60 1 1e10', 'let ar=real(v(ampout))', 'let ai=imag(v(ampout))', 'let br=real(v(out))', 'let bi=imag(v(out))', 'let ir=real(i(Vloop))', 'let ii=imag(i(Vloop))', 'wrdata port_voltage.dat ar ai br bi ir ii', 'alter Vloop acmag=0', 'alter Iport acmag=1', 'ac dec 60 1 1e10', 'let ar=real(v(ampout))', 'let ai=imag(v(ampout))', 'let br=real(v(out))', 'let bi=imag(v(out))', 'let ir=real(i(Vloop))', 'let ii=imag(i(Vloop))', 'wrdata port_current.dat ar ai br bi ir ii', 'alter Iport acmag=0']
    return '\n'.join(commands)

def audit_commands():
    commands = [port_commands()]
    for j in range(len(DYNAMIC_NODES)):
        commands.extend([f'alter Inode{j} acmag=1', 'ac dec 1 1e3 1e9'])
        names = []
        for k, node in enumerate(DYNAMIC_NODES):
            commands.extend([f'let r{k}=real(v({node}))', f'let i{k}=imag(v({node}))'])
            names.extend([f'r{k}', f'i{k}'])
        commands.extend([f'wrdata node_{j}.dat ' + ' '.join(names), f'alter Inode{j} acmag=0'])
    return '\n'.join(commands)

def augmented_deck(original):
    if original.count('Vloop ampout out DC 0 AC 0') != 1:
        raise ValueError('Unexpected topology/probe')
    sources = 'Iport 0 out DC 0 AC 0\n' + ''.join((f'Inode{j} 0 {n} DC 0 AC 0\n' for j, n in enumerate(DYNAMIC_NODES)))
    circuit = original.split('.control')[0]
    return circuit + sources + '.control\nset wr_singlescale\nset wr_vecnames\nset numdgt=15\nop\nprint all\n' + audit_commands() + '\nquit\n.endc\n.end\n'
