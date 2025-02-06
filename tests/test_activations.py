import torch
import torch.nn as nn
import pytest
from torchortho.fourier_activation import FourierActivation
from torchortho.hermite_activation import HermiteActivation
from torchortho.tropical_activation import TropicalActivation


def test_fourier_activation():
    activation = FourierActivation(degree=6)
    x = torch.randn(10, 5)
    y = activation(x)
    assert y.shape == x.shape, "Output shape should match input"
    assert y.dtype == x.dtype, "Output dtype should match input"
    y.sum().backward()
    assert x.grad is None, "Gradient should be computed correctly"


def test_hermite_activation():
    activation = HermiteActivation(degree=3)
    x = torch.randn(10, 5)
    y = activation(x)
    assert y.shape == x.shape, "Output shape should match input"
    assert y.dtype == x.dtype, "Output dtype should match input"
    y.sum().backward()
    assert x.grad is None, "Gradient should be computed correctly"


def test_hermite_activation_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    degree = 4
    activation = HermiteActivation(degree, use_numba=False).cuda()
    x = torch.rand(100, device="cuda") * 2.0
    x = nn.Parameter(x)
    output_cuda = activation.hermite_polynomials(x)
    output_cuda.sum().backward()
    assert x.grad is not None
    x_grad_cuda = x.grad

    activation = HermiteActivation(degree, use_numba=True).cuda()
    x = x.clone().detach().cuda()
    x = nn.Parameter(x)
    output = activation.hermite_polynomials(x)
    output.sum().backward()
    assert x.grad is not None

    assert torch.allclose(output, output_cuda, rtol=1e-3)
    assert torch.allclose(x.grad, x_grad_cuda, rtol=1e-3)


def test_hermite_activation_with_init():
    degree = 8
    act = nn.GELU()
    activation = HermiteActivation(degree, act_init=act)
    x = torch.rand(10, 5)
    output = activation(x)
    assert output.shape == x.shape


def test_tropical_activation():
    activation = TropicalActivation(degree=3)
    x = torch.randn(10, 5)
    y = activation(x)
    assert y.shape == x.shape, "Output shape should match input"
    assert y.dtype == x.dtype, "Output dtype should match input"
    y.sum().backward()
    assert x.grad is None, "Gradient should be computed correctly"


if __name__ == "__main__":
    pytest.main()
