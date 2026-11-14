"""FP64 curvature primitives; parameters are sampled only for measurement."""
import torch


def probabilities(h, w):
    return (h.double() @ w.double().T).softmax(-1)


def gradient(h, p, y):
    residual = p.double().clone()
    residual[torch.arange(len(y), device=y.device), y] -= 1
    return residual.T @ h.double() / len(h)


def hessian_block(h, p, rows, cols):
    """Exact principal block, given COMPLETE-vocabulary probabilities."""
    x = h.double()[:, cols]
    q = p.double()[:, rows]
    r, c = len(rows), len(cols)
    # Diagonal class term minus a Gram matrix; never allocate N x m x m.
    z = (q[:, :, None]*x[:, None, :]).reshape(len(x), r*c)
    out = -(z.T @ z)
    diagonal = torch.einsum('na,nj,nk->ajk', q, x, x)
    for a in range(r):
        out[a*c:(a+1)*c, a*c:(a+1)*c] += diagonal[a]
    return out / len(x)


def trace(h, p):
    return ((1-p.double().square().sum(-1))*h.double().square().sum(-1)).mean()


def fitted_statistics(h, p):
    h, p = h.double(), p.double()
    b = h.T @ h / len(h)
    mean = p.mean(0)
    t = trace(h, p)
    denom = (1-mean.square().sum())*torch.trace(b)
    if not bool(torch.isfinite(denom)) or denom <= 0:
        raise ValueError('degenerate K curvature')
    return dict(B=b, pbar=mean, alpha=t/denom, trace=t)


def statistics_block(stats, rows, cols):
    q = stats['pbar'][rows].double()
    b = stats['B'].double()[cols][:, cols]
    return stats['alpha']*torch.kron(torch.diag(q)-q[:, None]*q[None, :], b)


def inclusion_moments(block, rows, cols, vocab, width):
    """Horvitz-Thompson moments conditional on a fixed data-probe Hessian."""
    r, c = len(rows), len(cols)
    if r < 2 or c < 2 or r > vocab or c > width:
        raise ValueError('both rectangular sample dimensions must be >=2')
    vr = torch.arange(r, device=block.device).repeat_interleave(c)
    fc = torch.arange(c, device=block.device).repeat(r)
    scalar = lambda x: torch.tensor(x, dtype=torch.float64, device=block.device)
    pv = torch.where(vr[:, None] == vr[None, :], scalar(r/vocab),
                     scalar(r*(r-1)/(vocab*(vocab-1))))
    pf = torch.where(fc[:, None] == fc[None, :], scalar(c/width),
                     scalar(c*(c-1)/(width*(width-1))))
    t = block.diag().sum()/(r/vocab*c/width)
    s = (block.double().square()/(pv*pf)).sum()
    return dict(trace=float(t), second_moment=float(s),
                effective_rank_plugin=float(t*t/s) if s else None)


def block_metrics(block, g=None):
    h = block.double()
    size = torch.linalg.vector_norm(h)
    sym = float(torch.linalg.vector_norm(h-h.T)/size) if size else 0.
    values, vectors = torch.linalg.eigh((h+h.T)/2)
    top = float(values[-1])
    psd = float(values[0]) >= -1e-8*max(top, 0.)
    result = dict(symmetry_error=sym, psd=psd, maximum=top,
                  minimum=float(values[0]), trace=float(h.trace()),
                  second_moment=float(h.square().sum()),
                  near_zero_fraction=float((values.abs() <= 1e-8*max(top, 0.)).double().mean()),
                  quantiles=torch.quantile(values, torch.tensor([0., .05, .5, .95, 1.], dtype=torch.float64, device=h.device)).tolist())
    if g is not None:
        result['gradient_eigen_energy'] = (vectors.T @ g.double().flatten()).square().tolist()
    return result, values


def sample_blocks(vocab, width, *, tiny=False, count=5):
    layouts = [(min(4, vocab), min(4, width)), (min(8, vocab), min(2, width))] if tiny else [(128, 32), (512, 8)]
    result = []
    for pool, seed in [('fit', 2026111411), ('check', 2026111412)]:
        gen = torch.Generator().manual_seed(seed)
        for layout, (r, c) in enumerate(layouts):
            # Expanding the first layout must not change the second layout's prefix.
            gen.manual_seed(seed+100*layout)
            for index in range(count):
                rows = torch.randperm(vocab, generator=gen)[:r].sort().values
                cols = torch.randperm(width, generator=gen)[:c].sort().values
                result.append(dict(pool=pool, layout=layout, index=index,
                                   rows=rows.tolist(), cols=cols.tolist()))
    return result
