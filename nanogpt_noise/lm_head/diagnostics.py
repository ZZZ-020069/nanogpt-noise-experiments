"""Fixed-feature FP64 diagnostics, kept outside the training path."""
import numpy as np
import torch
from torch.nn import functional as F
from ..data import batch
from .curvature import probabilities, gradient


def singular_values(g):
    return torch.linalg.eigvalsh(g.double().T @ g.double()).clamp_min(0).sqrt().flip(0).tolist()


def compatible(stats, g):
    values, vectors = torch.linalg.eigh(stats['B'].double())
    cutoff = max(float(values[-1]), 0.)*1e-12
    null = vectors[:, values.abs() <= cutoff]
    denominator = float(torch.linalg.vector_norm(g))
    residual = float(torch.linalg.vector_norm(g @ null))/denominator if denominator else 0.
    # The common-logit translation is an exact left null direction as well.
    translation = float(torch.linalg.vector_norm(g.sum(0))) / (
        denominator*len(g)**.5) if denominator else 0.
    return dict(numerical_nullity=null.shape[1], threshold=cutoff,
                B_minimum=float(values[0]), B_maximum=float(values[-1]),
                gradient_null_fraction=residual, translation_fraction=translation,
                passed=residual <= 1e-8 and translation <= 1e-8,
                caveat='Numerical diagnostic only; no mode is deleted or regularized.')


def taylor_chunk(h, y, w, w0):
    """Return losses and gradients on precisely the same fixed features."""
    h, w, w0 = h.double(), w.double(), w0.double()
    z0, z = h @ w0.T, h @ w.T
    p0, p = z0.softmax(-1), z.softmax(-1)
    delta = h @ (w-w0).T
    mean = (p0*delta).sum(-1)
    initial = F.cross_entropy(z0, y)
    linear = mean-delta.gather(1,y[:,None]).squeeze(1)
    quadratic = .5*((p0*delta.square()).sum(-1)-mean.square())
    tg = gradient(h,p0,y)+(p0*(delta-mean[:,None])).T @ h/len(h)
    return dict(real=F.cross_entropy(z,y), initial=initial,
                taylor=initial+(linear+quadratic).mean(),
                gradient=gradient(h,p,y), taylor_gradient=tg)


@torch.no_grad()
def fixed_gradient_probe(model, data, indices, budget, rt, micro, stats, mode, test=False,
                         chunk_tokens=64):
    """Fit: first 16 FULL batches. Check: the existing fixed evaluation pools."""
    w = model.lm_head.weight.detach().double()
    if mode == 'fit':
        groups = [(f'train-step-{i}', 'train', indices['train'][i])
                  for i in range(min(16,budget.steps) if test else 16)]
    else:
        groups = [(f'eval-{split}',split,indices['eval_'+split]) for split in ('train','val')]
    direction = torch.randn(w.shape,generator=torch.Generator().manual_seed(2026111421),
                            dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    mean = torch.zeros(w.shape,dtype=torch.float64)
    scatter = 0.
    records = []
    for index, (label,split,positions) in enumerate(groups):
        array = np.memmap(data/f'{split}.bin',dtype='<u2',mode='r')
        gs = torch.zeros_like(w)
        try:
            for offset in range(0,len(positions),micro):
                x,y = batch(array,positions[offset:offset+micro],budget.sequence,rt.device)
                with rt.autocast():
                    h = model.features(x).flatten(0,1)
                y = y.flatten()
                for start in range(0,len(h),chunk_tokens):
                    hf, yf = h[start:start+chunk_tokens].double(),y[start:start+chunk_tokens]
                    gs += gradient(hf,probabilities(hf,w),yf)*len(hf)
        finally:
            array._mmap.close()
        g = (gs/(len(positions)*budget.sequence)).cpu()
        change = g-mean
        mean += change/(index+1)
        scatter += float((change*(g-mean)).sum())
        records.append(dict(label=label, sequences=len(positions),
            gradient_norm=float(torch.linalg.vector_norm(g)),
            random_direction_projection=float((g*direction).sum()),
            singular_values=singular_values(g), compatibility=compatible(stats,g)))
        print(f'fixed gradients {mode}: {index+1}/{len(groups)}',flush=True)
    return dict(mode=mode, count=len(groups), test=test, batches=records,
                mean_norm=float(torch.linalg.vector_norm(mean)),
                sample_total_variance=scatter/(len(groups)-1) if mode=='fit' else None,
                mean_singular_values=singular_values(mean),
                mean_compatibility=compatible(stats,mean),
                passed=(compatible(stats,mean)['passed'] if mode=='fit'
                        else all(r['compatibility']['passed'] for r in records)),
                direction_seed=2026111421)
