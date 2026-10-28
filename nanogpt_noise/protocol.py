"""Immutable research constants; runtime layout is separate from protocol."""
import math
from dataclasses import asdict, dataclass

SIGMA1 = (0., .03, .1, .3, .9)
SIGMA2 = (0., .0001, .001, .01, .1, 1., 10.)
SCOPES = ('full', 'last1', 'last2', 'last3', 'last6', 'last12')
CENTERS = dict(adamw=.0006, adam=.0006, sgd=.01, muon=.02)
INITIAL = (-1., -.5, 0., .5, 1.)
VERSION = 'v1.8'
INPUT_POLICY = 'server-fresh-v1'
SEEDS = dict(initialization=1337, data=1337, noise=20260902,
             eval_train=20260829, eval_val=20260830, preparation=2357)


@dataclass(frozen=True)
class Budget:
    sequence: int = 1024
    batch: int = 512
    steps: int = 3815
    warmup: int = 100_000_000
    eval_every: int = 100_000_000
    eval_sequences: int = 200

    @property
    def step_tokens(self):
        return self.sequence * self.batch

    @property
    def tokens(self):
        return self.step_tokens * self.steps

    def accumulations(self, world, micro):
        if world < 1 or micro < 1 or self.batch % (world * micro):
            raise ValueError('global batch must be divisible by world_size * micro_batch')
        return self.batch // (world * micro)

    def evaluate_at(self, step):
        return step in (0, self.steps) or (step * self.step_tokens // self.eval_every
                   > (step - 1) * self.step_tokens // self.eval_every)

    def multiplier(self, tokens):
        if tokens < self.warmup:
            return tokens / self.warmup
        if tokens >= self.tokens:
            return .1
        return .1 + .45 * (1 + math.cos(math.pi * (tokens-self.warmup)
                                       / (self.tokens-self.warmup)))


def grid():
    return [dict(scope=s, optimizer=o, node=i*7+j, sigma1=a, sigma2=b)
            for s in SCOPES for o in CENTERS
            for i, a in enumerate(SIGMA1) for j, b in enumerate(SIGMA2)]


def configuration(scope, optimizer, node, exponent):
    if scope not in SCOPES or optimizer not in CENTERS or not 0 <= node < 35:
        raise ValueError('invalid scope, optimizer or noise node')
    if not math.isfinite(exponent):
        raise ValueError('nonfinite LR exponent')
    return dict(design=VERSION, scope=scope, optimizer=optimizer, node=node,
                sigma1=SIGMA1[node//7], sigma2=SIGMA2[node%7], exponent=exponent,
                peak_lr=CENTERS[optimizer]*10**exponent, grad_clip=0,
                budget=asdict(Budget()), seeds=SEEDS)


def require_current_input(identity):
    """Require this release's freshly prepared, registered input realization."""
    from .model import Shape
    if (identity.get('tiny') or identity.get('design') != VERSION
            or identity.get('budget') != asdict(Budget())
            or identity.get('shape') != asdict(Shape())
            or identity.get('meta', {}).get('input_policy') != INPUT_POLICY
            or identity.get('meta', {}).get('preparation_design') != VERSION
            or 'migration' in identity):
        raise ValueError('Frozen inputs do not match the current v1.8/2B protocol; '
                         'run bash scripts/run_acp.sh inputs in a fresh INPUT_ROOT')
