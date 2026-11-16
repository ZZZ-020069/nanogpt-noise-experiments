"""Optional CUDA Triton fused online-softmax kernels for LM Head T and K objectives."""
import os
from pathlib import Path
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

_TRITON_READY = None


def _ensure_cuda_link_path():
    standard = ('/usr/lib/x86_64-linux-gnu/libcuda.so', '/lib/x86_64-linux-gnu/libcuda.so')
    if any(Path(p).exists() for p in standard):
        return
    candidates = [Path('/usr/local/cuda/lib64/stubs')]
    usr_local = Path('/usr/local')
    if usr_local.is_dir():
        candidates.extend(sorted(usr_local.glob('cuda-*/lib64/stubs')))
    stubs = next((p for p in candidates if (p / 'libcuda.so').exists()), None)
    if stubs is None:
        return
    current = os.environ.get('LIBRARY_PATH', '')
    parts = [p for p in current.split(':') if p]
    if str(stubs) not in parts:
        os.environ['LIBRARY_PATH'] = ':'.join(parts + [str(stubs)])


if HAS_TRITON:
    @triton.jit
    def _t_online_fwd_bwd_kernel(
        l0_ptr,
        z_ptr,
        y_ptr,
        dz_ptr,
        loss_ptr,
        stride_l0,
        stride_z,
        stride_dz,
        n_cols,
        inv_scale,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        l0_row = l0_ptr + row * stride_l0
        z_row = z_ptr + row * stride_z
        dz_row = dz_ptr + row * stride_dz

        m_i = -float('inf')
        d_i = 0.0
        ez_i = 0.0
        ez2_i = 0.0

        for v0 in range(0, n_cols, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            l0 = tl.load(l0_row + offs, mask=mask, other=-float('inf')).to(tl.float32)
            zv = tl.load(z_row + offs, mask=mask, other=0.0).to(tl.float32)
            t_max = tl.max(l0, axis=0)
            m_new = tl.maximum(m_i, t_max)
            s_old = tl.where(m_i == -float('inf'), 0.0, tl.exp(m_i - m_new))
            e = tl.where(mask, tl.exp(l0 - m_new), 0.0)
            d_i = d_i * s_old + tl.sum(e, axis=0)
            ez_i = ez_i * s_old + tl.sum(e * zv, axis=0)
            ez2_i = ez2_i * s_old + tl.sum(e * zv * zv, axis=0)
            m_i = m_new

        lse = m_i + tl.log(d_i)
        m1 = ez_i / d_i
        m2 = ez2_i / d_i
        y_idx = tl.load(y_ptr + row).to(tl.int64)
        l0_y = tl.load(l0_row + y_idx).to(tl.float32)
        z_y = tl.load(z_row + y_idx).to(tl.float32)
        row_loss = (lse - l0_y) + (m1 - z_y) + 0.5 * (m2 - m1 * m1)
        tl.store(loss_ptr + row, row_loss)

        for v0 in range(0, n_cols, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            l0 = tl.load(l0_row + offs, mask=mask, other=-float('inf')).to(tl.float32)
            zv = tl.load(z_row + offs, mask=mask, other=0.0).to(tl.float32)
            p0 = tl.exp(l0 - lse)
            dz = p0 * (1.0 + zv - m1) - tl.where(offs.to(tl.int64) == y_idx, 1.0, 0.0)
            dz = (dz * inv_scale).to(tl.bfloat16)
            tl.store(dz_row + offs, dz, mask=mask)

    @triton.jit
    def _k_online_fwd_bwd_kernel(
        l0_ptr,
        y_ptr,
        dz_ptr,
        loss_ptr,
        stride_l0,
        stride_dz,
        n_cols,
        inv_scale,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        l0_row = l0_ptr + row * stride_l0
        dz_row = dz_ptr + row * stride_dz

        m_i = -float('inf')
        d_i = 0.0
        for v0 in range(0, n_cols, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            l0 = tl.load(l0_row + offs, mask=mask, other=-float('inf')).to(tl.float32)
            t_max = tl.max(l0, axis=0)
            m_new = tl.maximum(m_i, t_max)
            s_old = tl.where(m_i == -float('inf'), 0.0, tl.exp(m_i - m_new))
            e = tl.where(mask, tl.exp(l0 - m_new), 0.0)
            d_i = d_i * s_old + tl.sum(e, axis=0)
            m_i = m_new

        lse = m_i + tl.log(d_i)
        y_idx = tl.load(y_ptr + row).to(tl.int64)
        l0_y = tl.load(l0_row + y_idx).to(tl.float32)
        tl.store(loss_ptr + row, lse - l0_y)

        for v0 in range(0, n_cols, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            mask = offs < n_cols
            l0 = tl.load(l0_row + offs, mask=mask, other=-float('inf')).to(tl.float32)
            p0 = tl.exp(l0 - lse)
            dz = p0 - tl.where(offs.to(tl.int64) == y_idx, 1.0, 0.0)
            dz = (dz * inv_scale).to(tl.bfloat16)
            tl.store(dz_row + offs, dz, mask=mask)


def triton_t_fused(l0, z, y, inv_scale):
    """Fused online-softmax loss and dz computation for T in BF16/FP32 registers."""
    n_rows, n_cols = l0.shape
    l0 = l0.contiguous()
    z = z.contiguous()
    y = y.contiguous()
    dz = torch.empty_like(l0)
    losses = torch.empty((n_rows,), device=l0.device, dtype=torch.float32)
    _t_online_fwd_bwd_kernel[(n_rows,)](
        l0, z, y, dz, losses,
        l0.stride(0), z.stride(0), dz.stride(0),
        n_cols, inv_scale, BLOCK_V=2048, num_warps=8,
    )
    return losses.double().mean().to(torch.float32), dz


def triton_k_fused(l0, y, inv_scale):
    """Fused online-softmax constant loss and G_b^0 pre-GEMM dz buffer for K."""
    n_rows, n_cols = l0.shape
    l0 = l0.contiguous()
    y = y.contiguous()
    dz = torch.empty_like(l0)
    losses = torch.empty((n_rows,), device=l0.device, dtype=torch.float32)
    _k_online_fwd_bwd_kernel[(n_rows,)](
        l0, y, dz, losses,
        l0.stride(0), dz.stride(0),
        n_cols, inv_scale, BLOCK_V=2048, num_warps=8,
    )
    return losses.double().mean().to(torch.float32), dz


def triton_ready(device):
    """Verify Triton kernel JIT compilation once on CUDA, falling back safely if unavailable."""
    global _TRITON_READY
    if not HAS_TRITON or device.type != 'cuda':
        return False
    if _TRITON_READY is not None:
        return _TRITON_READY
    try:
        _ensure_cuda_link_path()
        probe_l0 = torch.zeros((1, 16), device=device, dtype=torch.bfloat16)
        probe_z = torch.zeros((1, 16), device=device, dtype=torch.bfloat16)
        probe_y = torch.zeros((1,), device=device, dtype=torch.int64)
        triton_k_fused(probe_l0, probe_y, 1.0)
        triton_t_fused(probe_l0, probe_z, probe_y, 1.0)
        _TRITON_READY = True
    except Exception as exc:
        print(f'[lm_head] Triton unavailable ({exc!r}); using PyTorch row-slice fallback', flush=True)
        _TRITON_READY = False
    return _TRITON_READY
