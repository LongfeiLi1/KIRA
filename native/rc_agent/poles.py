"""Scoped numerical stability: full closed-loop PZ with correct source termination.

ngspice-46 cur/pol requests the loaded circuit determinant, not a reduced
Vout/Vin transfer. This numerical test does not claim a formal/global proof.
"""
import re
import numpy as np

def dc_terminated_prefix(deck):
    """Keep source DC values; remove AC *fields*, including AC 0, and PULSE."""
    prefix = deck.split('.control')[0]
    lines = []
    for line in prefix.splitlines():
        if re.match('^[VI]\\w+', line, re.I):
            line = re.sub('\\s+AC\\s+\\S+(?:\\s+[-+]?\\d+(?:\\.\\d*)?)?', '', line, flags=re.I)
            line = re.sub('\\s+PULSE\\(.*\\)', '', line, flags=re.I)
            if re.search('\\b(AC|PULSE|SIN|PWL|EXP)\\b', line, re.I):
                raise ValueError('Unreviewed independent-source waveform for PZ')
        lines.append(line)
    if any((re.match('\\s*\\.options?\\b.*\\bklu\\b', x, re.I) for x in lines)):
        raise ValueError('Reviewed PZ protocol uses default Sparse, not KLU')
    return '\n'.join(lines) + '\n'

def pole_deck(deck):
    prefix = dc_terminated_prefix(deck)
    prefix = '\n'.join((x for x in prefix.splitlines() if not re.match('^(Vloop|Iport)\\b', x, re.I))) + '\n'
    prefix = re.sub('\\bampout\\b', 'out', prefix)
    return prefix + '.control\noption sparse\nset numdgt=15\nop\npz out 0 out 0 cur pol\necho PZ_FIRST_BEGIN\nprint all\necho PZ_FIRST_END\npz vp 0 vp 0 cur pol\necho PZ_SECOND_BEGIN\nprint all\necho PZ_SECOND_END\nquit\n.endc\n.end\n'

def parse_poles(log, marker=None):
    if marker:
        begin = f'PZ_{marker}_BEGIN'
        end = f'PZ_{marker}_END'
        if log.count(begin) != 1 or log.count(end) != 1:
            raise ValueError('Missing PZ output markers')
        log = log.split(begin)[1].split(end)[0]
    rows = re.findall('(?m)^pole\\((\\d+)\\)\\s*=\\s*([-+0-9.eE]+),\\s*([-+0-9.eE]+)\\s*$', log)
    if not rows or len({int(r[0]) for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate pole indices')
    poles = np.array([complex(float(r), float(i)) for _, r, i in rows])
    if not np.all(np.isfinite(poles)):
        raise ValueError('Nonfinite PZ result')
    return poles

def audit_poles(log, ac_f0_hz, ac_q, expected_count=23):
    """Pass/fail/unknown under this fixed topology and pinned model.

    Count23 is an empirical fingerprint established by terminated-source
    repeated nominal runs and sanity controls, NOT a symbolic rank proof.
    """
    result = {'status': 'unknown', 'scope': 'ngspice-46 SKY130 tt at measured DC point; local small-signal numerical test', 'formal_global_certificate': 'not_claimed', 'expected_pole_count': expected_count, 'count_basis': 'empirical fixed-topology fingerprint, not independent symbolic capacitance rank', 'solver': 'explicit option sparse with -n', 'source_termination': 'DC source values preserved; all AC fields and PULSE removed', 'rootfinder_limit': 'finite numerical root search may miss roots; repeat/count/conjugacy/AC and hidden-mode controls manage but do not eliminate this risk'}
    try:
        first = parse_poles(log, 'FIRST')
        second = parse_poles(log, 'SECOND')
        result['poles_rad_s'] = [[float(v.real), float(v.imag)] for v in first]
        result['pole_count'] = len(first)
        result['max_pole_real_rad_s'] = float(max(first.real))
        if np.any(first.real > 0) or np.any(second.real > 0):
            result.update(status='fail', reason='Detected right-half-plane closed-loop mode')
            return result
        if len(first) != expected_count or len(second) != expected_count:
            raise ValueError('Unexpected finite pole count')
        if np.any(first.real >= -0.001):
            raise ValueError('Marginal or zero pole requires review')
        scale = np.maximum(abs(first), 1.0)
        distances = abs(first[:, None] - first[None, :]) / scale[:, None]
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < 1e-08:
            raise ValueError('Duplicate/near-duplicate roots require multiplicity review')
        if max((min(abs(x - second)) / max(abs(x), 1) for x in first)) > 1e-06:
            raise ValueError('Port-invariant determinant repeat disagrees')
        if max((min(abs(x.conjugate() - first)) / max(abs(x), 1) for x in first)) > 1e-07:
            raise ValueError('Complex conjugacy failed')
        low = first[np.argmax(first.real)]
        if abs(low.imag) <= 1e-06:
            raise ValueError('Expected dominant lowpass pair missing')
        f0 = abs(low) / (2 * np.pi)
        q = abs(low) / (-2 * low.real)
        result.update(dominant_pole_f0_hz=float(f0), dominant_pole_q=float(q), ac_f0_relative_discrepancy=float(abs(f0 / ac_f0_hz - 1)), ac_q_relative_discrepancy=float(abs(q / ac_q - 1)))
        if abs(f0 / ac_f0_hz - 1) > 0.03 or abs(q / ac_q - 1) > 0.03:
            raise ValueError('Dominant PZ pair disagrees with measured AC fit')
        if re.search('(?i)(doanalyses:|failed|singular matrix|too many iterations|panic|pole.*not found|numerical aberrations|iteration limit|giving up)', log):
            raise ValueError('PZ convergence warning/error')
        result.update(status='pass', reason='LHP poles with fixed count, conjugacy, independent port repeat and AC-pair consistency')
    except ValueError as exc:
        result['reason'] = str(exc)
    return result
