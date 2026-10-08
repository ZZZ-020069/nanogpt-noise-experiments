import math
import torch


def norm(tensors):
    # Double precision reduction avoids classifying a finite FP32 vector as
    # nonfinite merely because its squared norm overflows FP32.
    sums=[torch.sum(t.detach().double().square()) for t in tensors]
    return math.sqrt(float(torch.stack(sums).sum())) if sums else 0.


def ratio(a, b):
    return a/b if b else None


@torch.no_grad()
def inject(named, sigma1, sigma2, step, diagnostic=True):
    named = list(named)
    if any(p.grad is None for _,p in named):
        raise RuntimeError('trainable parameter has grad=None')
    gs = [p.grad for _,p in named]
    if any(g.dtype != torch.float32 for g in gs):
        raise RuntimeError('noise requires FP32 accumulated gradients')
    d = sum(g.numel() for g in gs)
    gnorm = norm(gs)
    rms = math.hypot(sigma1*gnorm,sigma2)
    std = rms / math.sqrt(d)
    gen = torch.Generator(device=gs[0].device)
    gen.manual_seed((20260902+1000003*step) % (2**63-1))
    noise_sq = dot = 0.
    for g in gs:
        if std:
            xi = torch.randn(g.shape,device=g.device,dtype=g.dtype,generator=gen).mul_(std)
            if diagnostic:
                noise_sq += float(xi.double().square().sum())
                dot += float((g.double()*(g.double()+xi.double())).sum())
            g.add_(xi)
        elif diagnostic:
            dot += float(g.double().square().sum())
    result = dict(gradient_norm=gnorm)
    if diagnostic:
        actual, noisy = math.sqrt(noise_sq), norm(gs)
        result.update(D=d,noise_rms=rms,noise_norm=actual,noisy_gradient_norm=noisy,
                      noise_std=std,theoretical_ratio=ratio(rms,gnorm),
                      actual_ratio=ratio(actual,gnorm),cosine=ratio(dot,gnorm*noisy),
                      undefined_reason='zero denominator' if not gnorm or not noisy else None)
    return result
