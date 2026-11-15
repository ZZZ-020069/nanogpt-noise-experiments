"""Versioned Head-only identity, retaining v1.8 budgets and search."""
from pathlib import Path
from .. import protocol as baseline
from ..artifacts import digest, file_hash, code_identity as base_code

VERSION = 'LH-QP-20261114-draft2'
SCOPE = 'lm_head_only'
OBJECTIVES = ('G', 'T', 'K')
DEFAULT_MICRO = 16
CENTERS = baseline.CENTERS
Budget = baseline.Budget
require_current_input = baseline.require_current_input


def bound_micro(identity):
    value = identity.get('micro')
    if type(value) is not int or value < 1:
        raise ValueError('run identity requires a valid micro; use a fresh root, never migrate old outputs')
    return value


def require_micro(micro, test=False):
    bound_micro(dict(micro=micro))
    if not test and micro != DEFAULT_MICRO:
        raise ValueError('LM Head execution requires micro=16')


def configuration(objective, optimizer, node, exponent):
    if objective not in OBJECTIVES:
        raise ValueError('objective must be G, T or K')
    value = baseline.configuration('last1', optimizer, node, exponent)
    value.update(design=VERSION, scope=SCOPE, objective=objective,
                 weight_tying='clone_after_load', evaluation='real_validation_ce')
    return value


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = dict(base_code()['files'])
    for path in sorted((root/'lm_head').glob('*.py')):
        files['nanogpt_noise/'+path.relative_to(root).as_posix()] = file_hash(path)
    return dict(sha256=digest(files), files=files)


def candidate_identity(config):
    expected = configuration(config['objective'], config['optimizer'],
                             config['node'], config['exponent'])
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('candidate does not match the Head-only protocol')
