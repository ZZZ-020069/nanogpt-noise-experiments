"""Deterministic sequential search with expansion-feedback priority."""
import math
from .protocol import INITIAL

TOL = 1e-6
TERMINAL = ('completed', 'numerical_failure')


def winner(records):
    valid = [r for r in records if r.get('status') == 'completed'
             and r.get('loss') is not None and math.isfinite(r['loss'])]
    if not valid:
        return None
    best = min(r['loss'] for r in valid)
    return min((r for r in valid if r['loss'] <= best + TOL), key=lambda r: r['x'])


def propose(records):
    if any(r.get('status') not in TERMINAL for r in records):
        raise ValueError('resume unfinished candidate before proposing a new LR')
    if len(records) >= 8:
        return None
    if len(records) < 5:
        return dict(x=INITIAL[len(records)], reason='initial')
    w = winner(records)
    xs = sorted(r['x'] for r in records)
    last = records[-1]
    if last['reason'] == 'extend':
        old = next(r for r in records if r['x'] == last['parent_x'])
        if not (last['status'] == 'completed' and last['loss'] < old['loss']-TOL):
            return dict(x=(old['x']+last['x'])/2, reason='extension_feedback_midpoint')
    if w is None:
        return dict(x=xs[0]-.5, reason='all_failed_extend_lower')
    x = w['x']
    if x in (xs[0], xs[-1]):
        return dict(x=x+(-.5 if x == xs[0] else .5), reason='extend', parent_x=x)
    i = xs.index(x)
    y = xs[i-1] if x-xs[i-1] >= xs[i+1]-x else xs[i+1]
    return dict(x=(x+y)/2, reason='interior_midpoint')


def summarize(records):
    w = winner(records)
    xs = sorted(r['x'] for r in records)
    adjacent = []
    if w:
        i = xs.index(w['x'])
        neighbors = xs[max(0,i-1):i] + xs[i+1:i+2]
        adjacent = [r['x'] for r in records if r['x'] in neighbors
                    and r['status'] == 'numerical_failure']
    return dict(candidates=records, winner=w,
                valid_count=sum(r.get('status') == 'completed' for r in records),
                boundary_unclosed=bool(w and w['x'] in (xs[0], xs[-1])),
                adjacent_failed_exponents=adjacent,
                complete=len(records) == 8)
