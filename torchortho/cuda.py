"""
torchortho - CUDA-Accelerated Hermite Polynomial Computation

Author: Ismail Khalfaoui-Hassani
Date: February 2025
License: GPL-3.0
Description: Implements CUDA kernels for computing Hermite polynomials used in learnable activations.

For reproducing experiments:
- Vision: https://github.com/K-H-Ismail/ConvNeXt-ortho
- Language: https://github.com/K-H-Ismail/pytorch-language-models
"""

import torch
from torch import Tensor
from torch.autograd import Function

try:
    from numba import cuda

    @cuda.jit("void(float32[:],int64,float32[:])")
    def hermite_forward_cuda_kernel(x, n, out):
        """CUDA kernel to compute Hermite polynomials up to degree n."""
        idx = cuda.grid(1)
        if idx < x.size:
            base_idx = idx * n
            value = x[idx]
            out[base_idx] = 1.0  # He_0(x) = 1
            if n > 1:
                out[base_idx + 1] = value  # He_1(x) = x
            for k in range(2, n):
                out[base_idx + k] = (
                    value * out[base_idx + k - 1] - (k - 1) * out[base_idx + k - 2]
                )

    @cuda.jit("void(float32[:], int64, float32[:], float32[:])")
    def hermite_backward_cuda_kernel(x, n, out, grad_out):
        """CUDA kernel to compute Hermite polynomial gradients."""
        idx = cuda.grid(1)
        size = grad_out.size
        if idx < size:
            base_idx = idx * n
            grad = 0.0
            for k in range(1, n):
                grad += x[base_idx + k] * k * out[base_idx + k - 1]
            grad_out[idx] = grad

except Exception as _:
    print("No numba installed or no CUDA GPU found, running on cpu!")


class HermitePolynomialsNumba(Function):
    @staticmethod
    def forward(ctx, input: Tensor, degree: int) -> Tensor:
        """Computes Hermite polynomials via CUDA or CPU."""
        ctx.degree = degree
        ctx.input_size = input.size()
        ctx.numel = input.numel()
        output = torch.zeros(*input.size(), degree + 1).to(input)

        threads_per_block = 2**16
        blocks_per_grid = (ctx.numel + threads_per_block - 1) // threads_per_block

        hermite_forward_cuda_kernel[threads_per_block, blocks_per_grid](
            input.detach().flatten(),
            degree + 1,
            output.detach().flatten(),
        )

        ctx.output = output
        return output.view(*input.size(), -1)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple:
        """Computes gradient of Hermite polynomials."""
        threads_per_block = 2**16
        blocks_per_grid = (ctx.numel + threads_per_block - 1) // threads_per_block

        grad_output = torch.zeros(*ctx.input_size).to(grad)
        hermite_backward_cuda_kernel[threads_per_block, blocks_per_grid](
            grad.detach().flatten(),
            ctx.degree + 1,
            ctx.output.detach().flatten(),
            grad_output.detach().flatten(),
        )
        return (grad_output, None)


hermite_polynomials_numba = HermitePolynomialsNumba.apply
