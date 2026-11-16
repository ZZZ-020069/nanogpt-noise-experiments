"""Fixed backbone and complete Head objectives, without changing GPT."""
import torch
import time
from torch.nn import functional as F
from ..model import GPT
from ..optim import Bundle, Muon, build as base_build
from . import kernels


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

    @torch.no_grad()
    def forward_fast_chunk(self, h, targets, accum, objective=None, *, row_slice=1024):
        """Compute chunk loss and pre-GEMM dz buffer for K/T without an autograd graph."""
        objective = objective or self.objective
        if objective not in ('K', 'T'):
            raise ValueError('fast chunk path is only for K and T')
        h2d = h.reshape(-1, h.shape[-1])
        y_flat = targets.reshape(-1)
        n_tok = y_flat.numel()
        v_size = self.reference_head.shape[0]
        dev_type = h2d.device.type
        mm_dtype = torch.get_autocast_dtype(dev_type) if torch.is_autocast_enabled(dev_type) else h2d.dtype
        h_mm = h2d.to(mm_dtype)
        w0_mm = self.reference_head.to(mm_dtype)
        delta_mm = (self.lm_head.weight - self.reference_head).detach().to(mm_dtype) if objective == 'T' else None
        inv_scale = 1.0 / (n_tok * accum)

        if mm_dtype == torch.bfloat16 and getattr(self, 'use_triton_kernels', True) and kernels.triton_ready(h2d.device):
            l0 = F.linear(h_mm, w0_mm)
            if objective == 'K':
                return kernels.triton_k_fused(l0, y_flat, inv_scale)
            zs = F.linear(h_mm, delta_mm)
            return kernels.triton_t_fused(l0, zs, y_flat, inv_scale)

        dz_buf = torch.empty((n_tok, v_size), device=h2d.device, dtype=mm_dtype)
        loss_sum = torch.zeros((), device=h2d.device, dtype=torch.float64)

        for start in range(0, n_tok, row_slice):
            end = min(n_tok, start + row_slice)
            hs = h_mm[start:end]
            ys = y_flat[start:end]
            l0 = F.linear(hs, w0_mm).float()
            lse = l0.logsumexp(-1, keepdim=True)
            p0 = (l0 - lse).exp()
            const_rows = lse.squeeze(-1) - l0.gather(-1, ys[:, None]).squeeze(-1)
            if objective == 'K':
                loss_sum += const_rows.double().sum()
                dz = p0
            else:
                zs = F.linear(hs, delta_mm).float()
                p0z = p0 * zs
                m1 = p0z.sum(-1, keepdim=True)
                m2 = (p0z * zs).sum(-1, keepdim=True)
                lin_rows = m1.squeeze(-1) - zs.gather(-1, ys[:, None]).squeeze(-1)
                quad_rows = 0.5 * (m2.squeeze(-1) - m1.squeeze(-1).square())
                loss_sum += (const_rows + lin_rows + quad_rows).double().sum()
                dz = p0 * (1.0 + zs - m1)
            dz[torch.arange(end - start, device=dz.device), ys] -= 1.0
            dz.mul_(inv_scale)
            dz_buf[start:end] = dz.to(mm_dtype)

        return (loss_sum / n_tok).to(torch.float32), dz_buf

    @torch.no_grad()
    def backward_fast_chunk(self, h, dz_buf, grad_accum):
        """Accumulate dz^T @ h in the exact forward GEMM dtype."""
        h2d = h.reshape(-1, h.shape[-1]).to(dz_buf.dtype)
        grad_accum.add_((dz_buf.T @ h2d).float())

    @torch.no_grad()
    def finalize_K_step(self, grad_b0_mean, const_mean):
        """Add the step-constant K quadratic loss and gradient once per optimizer step."""
        with torch.autocast(device_type=grad_b0_mean.device.type, enabled=False):
            delta = (self.lm_head.weight - self.reference_head).float()
            linear_val = (grad_b0_mean * delta).sum().double()
            db = delta @ self.stat_B
            mean_b = self.stat_pbar @ db
            grad_quad = (self.stat_alpha * self.stat_pbar)[:, None] * (db - mean_b[None, :])
            quad_val = 0.5 * (delta * grad_quad).sum().double()
        return const_mean + linear_val + quad_val, grad_b0_mean + grad_quad


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
