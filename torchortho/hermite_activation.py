"""
torchortho - Learnable Hermite Activation Function

Author: Ismail Khalfaoui-Hassani
Date: February 2025
License: GPL-3.0
Description: Implements a learnable Hermite polynomial activation function.

For reproducing experiments:
- Vision: https://github.com/K-H-Ismail/ConvNeXt-ortho
- Language: https://github.com/K-H-Ismail/pytorch-language-models
"""

import torch
import torch.nn as nn
import math
from torchortho.cuda import hermite_polynomials_numba
import os
import pickle


class HermiteActivation(nn.Module):
    def __init__(
        self,
        degree,
        use_numba=False,
        clamp=False,
        act_init=None,
        requires_grad=True,
        load_cached=True,
    ):
        """
        Initializes the Hermite activation function.

        Parameters:
        degree (int): Degree of the Hermite polynomial.
        """
        super(HermiteActivation, self).__init__()
        self.degree = degree
        self.requires_grad = requires_grad
        self.cache_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"cache_hermite_{degree}.pkl",
        )
        self.load_cached = load_cached
        # Initialize coefficients as learnable parameters
        coefficients_std = torch.zeros(degree + 1)
        coefficients_std[0] = 0.66549  # sqrt((pi^2 /6) - zeta(3))
        coefficients_std[1:] = 1.0 / (torch.arange(1, degree + 1) ** (3 / 2))
        self.coefficients = nn.Parameter(
            coefficients_std * math.sqrt(6.0) / math.pi, requires_grad=requires_grad
        )
        self.use_numba = use_numba
        if self.use_numba:
            self.register_buffer(
                "normalization_term",
                torch.arange(1, self.degree + 2).lgamma().exp().sqrt(),
            )
        coeffs, grid = self.init_grid(torch.tensor(self.degree))
        self.register_buffer("coeffs", coeffs)
        self.register_buffer("grid", grid)
        if clamp:
            self.clamp = LearnableClamp(math.sqrt(3))
        else:
            self.clamp = None
        if act_init:
            self.act_init = act_init
            self.init_coeffs()

    def init_coeffs(self):
        # Check for cached results
        if os.path.exists(self.cache_file) and self.load_cached:
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                print("Loaded cached coefficients")
                self.coefficients = nn.Parameter(
                    cached_data["coefficients"], requires_grad=self.requires_grad
                )
                return
        # Compute the coefficients and frequencies at init with a Hermite interpolation
        lim = math.sqrt(self.degree)  # self.degree
        act = self.act_init
        coefficients = nn.Parameter(
            self.coefficients.clone() + torch.randn_like(self.coefficients)
        )

        def model():
            # x = torch.linspace(torch.tensor(-lim), torch.tensor(lim), 2**9)
            x = (torch.rand(2**9) - 0.5) * 2 * lim
            x.requires_grad = True
            y = act(x)
            # Compute the derivative of the activation with respect to x
            y_deriv = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(x),
                create_graph=True,
            )[0]
            x.requires_grad = False
            x_orig = x

            if self.clamp:
                x = self.clamp(x)
                x_orig = self.clamp(x_orig)

            x = x.unsqueeze(-1).unsqueeze(-1)
            x = torch.pow(x.abs(), self.grid) * torch.sign(x).pow(self.grid)
            x = self.coeffs * x
            x = x.sum(-1)
            x_deriv = x[..., :-1]
            x = x @ coefficients

            # might need an extra term for clamp derivative
            # not using clamp and init_act together!
            k = torch.arange(1, self.degree + 1).sqrt().to(x)
            x_deriv = k * x_deriv
            x_deriv = x_deriv @ coefficients[1:]
            return x, x_deriv, y, y_deriv

        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = torch.optim.Adam([coefficients], lr=0.01)
        loss_min = 1e8
        coefficients_min = None

        # Training loop
        num_epochs = 40000
        print("Init activation coefficients")
        for _ in range(num_epochs):
            # Forward pass
            y_pred, y_pred_deriv, y, y_deriv = model()

            # Compute loss
            loss = loss_fn(y_pred, y) + loss_fn(y_pred_deriv, y_deriv)

            if loss < loss_min:
                loss_min = loss
                coefficients_min = coefficients.clone().detach()

            if loss < 1e-4:
                coefficients_min = coefficients.clone().detach()
                break
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("loss_min: ", loss_min)
        # Cache the results
        with open(self.cache_file, "wb") as f:
            pickle.dump(
                {
                    "coefficients": coefficients_min,
                },
                f,
            )
            print("Cached coefficients to", self.cache_file)
        self.coefficients = nn.Parameter(
            coefficients_min, requires_grad=self.requires_grad
        )

    def init_grid(self, n):
        # I couldn't vectorize this loop because the torch.lgamma function,
        # which calculates the factorial in a vectorized way, returns
        # wrong values when float(“inf”) is present. However, the loop
        # is just in the initialization, so nothing critical
        coeffs = [[0.0] * (n // 2 + 1) for _ in range(n + 1)]
        grid = [[0.0] * (n // 2 + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n // 2 + 1):
                if j <= i // 2:
                    coeffs[i][j] = math.exp(
                        0.5 * math.log(math.factorial(i))
                        - math.log(math.factorial(j))
                        - math.log(math.factorial(i - 2 * j))
                        - (j * math.log(2))
                    ) * math.pow(-1, j)
                    grid[i][j] = i - 2 * j
                else:
                    coeffs[i][j] = 0
                    grid[i][j] = 0

        return torch.tensor(coeffs), torch.tensor(grid)

    def hermite_polynomials_pytorch(self, x):
        # power
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.pow(x.abs(), self.grid) * torch.sign(x).pow(self.grid)

        # Apply Hermite combination
        x = self.coeffs * x
        x = x.sum(-1)
        return x

    def hermite_polynomials(self, x):
        """
        Compute the Hermite polynomial using the explicit formula.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Evaluated Hermite polynomial.
        """
        if self.clamp:
            x = self.clamp(x)
        # Determine device
        device = x.device.type
        if device == "cuda" and self.use_numba:
            return hermite_polynomials_numba(x, self.degree) / self.normalization_term
        else:
            return self.hermite_polynomials_pytorch(x)

    def forward(self, x):
        """
        Forward pass for the Hermite activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying Hermite activation function.
        """
        # Calculate the Hermite polynomial
        # exp_x = (-x * x * 0.5).exp()
        # norm_x = x.norm()
        x = self.hermite_polynomials(x)
        # Multiply by learnable coefficients and sum
        x = x @ self.coefficients
        # x = x * exp_x
        return x

    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class LearnableClamp(nn.Module):
    def __init__(self, init_lim=1.0, scale=1.0):
        super(LearnableClamp, self).__init__()
        # Initialize learnable parameters for min and max
        self.lim_val = nn.Parameter(torch.tensor(init_lim))
        self.scale = scale  # Controls the steepness of the clamping effect

    def forward(self, x):
        # Use a smooth clamping function, e.g., sigmoid-based clamp
        return ((x * self.scale).sigmoid() - 0.5) * self.lim_val * 2.0
