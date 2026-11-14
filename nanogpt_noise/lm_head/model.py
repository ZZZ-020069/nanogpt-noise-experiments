"""Fixed backbone and complete Head objectives, without changing GPT."""
import torch
import time
from torch.nn import functional as F
from ..model import GPT
from ..optim import Bundle, Muon, build as base_build


class HeadGPT(GPT):
    def activate_head(self, objective='G', statistics=None):
        if objective not in ('G', 'T', 'K'):
            raise ValueError('unknown objective')
        # Call only after loading the original tied initialization.
        for p in self.parameters():
            p.requires_grad_(False)
        self.lm_head.weight = torch.nn.Parameter(self.lm_head.weight.detach().clone())
        self.register_buffer('reference_head', self.lm_head.weight.detach().clone())
        self.objective = objective
        if objective == 'K':
            if statistics is None:
                raise ValueError('K requires a verified fitted statistics artifact')
            for key in ('B', 'pbar', 'alpha'):
                self.register_buffer('stat_'+key, statistics[key].float().clone())
        if self.lm_head.weight.data_ptr() == self.transformer.wte.weight.data_ptr():
            raise RuntimeError('Head and embedding still share storage')
        return self

    @torch.no_grad()
    def features(self, ids):
        profiling = getattr(self,'profile',None)
        if profiling is not None:
            if ids.device.type=='cuda':torch.cuda.synchronize(ids.device)
            started=time.perf_counter()
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.transformer.wte(ids) + self.transformer.wpe(positions)
        for block in self.transformer.h:
            x = block(x)
        h=self.transformer.ln_f(x).detach()
        if profiling is not None:
            if ids.device.type=='cuda':torch.cuda.synchronize(ids.device)
            profiling['features_seconds']+=time.perf_counter()-started
        return h

    def loss_from_features(self, h, targets, objective=None):
        objective = objective or self.objective
        if objective == 'G':
            logits = self.lm_head(h)
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        # Reference probabilities are constants; only the complete Head has grads.
        with torch.no_grad():
            logits0 = F.linear(h, self.reference_head).float()
            p0 = logits0.softmax(-1)
            constant = F.cross_entropy(logits0.flatten(0, 1), targets.flatten())
        delta = self.lm_head.weight - self.reference_head
        z = F.linear(h, delta).float()
        linear = (p0*z).sum(-1) - z.gather(-1, targets[..., None]).squeeze(-1)
        if objective == 'T':
            quadratic = .5*((p0*z.square()).sum(-1) - (p0*z).sum(-1).square())
            return constant + (linear+quadratic).mean()
        if objective != 'K':
            raise ValueError('unknown objective')
        # No V x V matrix. FP32 reduction/curvature math avoids BF16 cancellation.
        with torch.autocast(device_type=delta.device.type, enabled=False):
            db = delta @ self.stat_B
            mean = self.stat_pbar @ delta
            quadratic = .5*self.stat_alpha*(
                (self.stat_pbar[:, None]*delta*db).sum()
                - (mean @ self.stat_B * mean).sum())
        return constant + linear.mean() + quadratic

    def forward(self, ids, targets):
        return self.loss_from_features(self.features(ids), targets)

    def real_loss(self, ids, targets):
        return self.loss_from_features(self.features(ids), targets, 'G')


def build_optimizer(named, kind, lr):
    named = list(named)
    if len(named) != 1 or named[0][0] != 'lm_head.weight':
        raise ValueError('optimizer must receive the entire single Head matrix')
    if kind != 'muon':
        return base_build(named, kind, lr)
    name, p = named[0]
    return Bundle([Muon([p], lr)],
                  [dict(names=[name], weight_decay=.01, lr_ratio=1.,
                        algorithm='Muon-head', auxiliary=False)])
