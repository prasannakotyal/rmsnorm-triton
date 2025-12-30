"""
RMSNorm Implementations

Provides 3 implementations of RMSNorm for benchmarking:
1. PyTorch native (torch.nn.functional.rms_norm)
2. PyTorch compiled (torch.compile wrapped native)
3. Triton optimized kernel
"""

import torch
import triton
import triton.language as tl


def pytorch_native_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """PyTorch native RMSNorm using torch.nn.functional.rms_norm."""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


# Compiled version - uses dynamic=True to handle varying tensor sizes
# Falls back to native if torch.compile is unavailable (e.g., Python 3.14+)
try:
    pytorch_compiled_rmsnorm = torch.compile(pytorch_native_rmsnorm, dynamic=True)
    TORCH_COMPILE_AVAILABLE = True
except RuntimeError:
    pytorch_compiled_rmsnorm = pytorch_native_rmsnorm  # fallback
    TORCH_COMPILE_AVAILABLE = False


@triton.jit
def rms_norm_triton_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMSNorm.
    Each block processes one row.
    """
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x_vals = tl.load(x_ptr + row_idx * N + col_offsets, mask=mask)
    w_vals = tl.load(weight_ptr + col_offsets, mask=mask)

    sum_sq = tl.sum(x_vals * x_vals)
    rms = tl.rsqrt(sum_sq / N + eps)

    y_vals = x_vals * rms * w_vals
    tl.store(y_ptr + row_idx * N + col_offsets, y_vals, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Triton optimized RMSNorm kernel."""
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)

    BLOCK_SIZE = 1024
    rms_norm_triton_kernel[grid](x, y, weight, M, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y


__all__ = [
    "pytorch_native_rmsnorm",
    "pytorch_compiled_rmsnorm",
    "triton_rmsnorm",
    "TORCH_COMPILE_AVAILABLE",
]
