"""Explicit fixed optimizer recipes; no gradient clipping."""
import math
import torch


def orthogonalize(u):
    x = u.to(torch.bfloat16)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        a = x @ x.T
        x = 3.4445*x + (-4.7750*a + 2.0315*(a@a))@x
    return x.T if transpose else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr, weight_decay=.01, lr_ratio=1.))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    raise RuntimeError('Muon parameter missing gradient')
                state = self.state[p]
                if not state:
                    state['momentum'] = torch.zeros_like(p)
                m = state['momentum']
                m.mul_(.95).add_(p.grad,alpha=.05)
                u = p.grad.mul(.05).add(m,alpha=.95)
                q = orthogonalize(u).to(p.dtype)
                p.mul_(1-lr*group['weight_decay'])
                p.add_(q,alpha=-lr*math.sqrt(max(1.,p.shape[0]/p.shape[1])))


class Bundle:
    def __init__(self, optimizers, groups):
        self.optimizers, self.groups = optimizers, groups

    def set_lr(self, lr):
        for opt in self.optimizers:
            for group in opt.param_groups:
                group['lr'] = lr * group['lr_ratio']

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, state):
        if len(state) != len(self.optimizers):
            raise ValueError('optimizer bundle mismatch')
        for o,s in zip(self.optimizers,state):
            o.load_state_dict(s)

    def state_tensors(self):
        for opt in self.optimizers:
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        yield value


def build(named, kind, lr):
    named = list(named)
    groups_meta = []
    def groups(items, decay, ratio):
        result = []
        for matrix in (True,False):
            selected = [(n,p) for n,p in items if (p.ndim>=2) == matrix]
            if selected:
                result.append(dict(params=[p for _,p in selected],weight_decay=decay if matrix else 0.,lr_ratio=ratio))
                groups_meta.append(dict(names=[n for n,_ in selected],weight_decay=decay if matrix else 0.,lr_ratio=ratio))
        return result
    if kind == 'muon':
        main = [(n,p) for n,p in named if n.startswith('transformer.h.') and p.ndim==2]
        main_names = {n for n,_ in main}
        aux = [(n,p) for n,p in named if n not in main_names]
        groups_meta.append(dict(names=sorted(main_names),weight_decay=.01,lr_ratio=1.))
        opts = [Muon([p for _,p in main],lr),
                torch.optim.AdamW(groups(aux,.1,.03),lr=lr*.03,betas=(.9,.95),eps=1e-8,foreach=False)]
    elif kind in ('adam','adamw'):
        cls = torch.optim.Adam if kind == 'adam' else torch.optim.AdamW
        opts = [cls(groups(named,.1,1.),lr=lr,betas=(.9,.95),eps=1e-8,foreach=False)]
    elif kind == 'sgd':
        opts = [torch.optim.SGD(groups(named,1e-4,1.),lr=lr,momentum=.9,dampening=0,nesterov=False,foreach=False)]
    else:
        raise ValueError(kind)
    return Bundle(opts,groups_meta)
