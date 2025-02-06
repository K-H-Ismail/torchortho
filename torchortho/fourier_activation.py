"""
torchortho - Learnable Fourier Activation Function

Author: Ismail Khalfaoui-Hassani
Date: February 2025
License: GPL-3.0
Description: Implements a learnable Fourier-based activation function.

For reproducing experiments:
- Vision: https://github.com/K-H-Ismail/ConvNeXt-ortho
- Language: https://github.com/K-H-Ismail/pytorch-language-models
"""

import math
import torch
import torch.nn as nn
from torch.nn import init
import os
import pickle


class FourierActivation(nn.Module):
    def __init__(self, degree, act_init=None, requires_grad=True, load_cached=True):
        """
        Initializes the Fourier activation function.

        Parameters:
        degree (int): Degree of the Fourier polynomial.
        """
        super(FourierActivation, self).__init__()
        self.degree = degree
        self.requires_grad = requires_grad
        self.cache_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"cache_fourier_{degree}.pkl"
        )
        self.load_cached = load_cached
        # Initialize coefficients as learnable parameters
        self.fundamental = nn.Parameter(torch.empty(1), requires_grad=requires_grad)
        self.phases = nn.Parameter(torch.empty(degree), requires_grad=requires_grad)

        # init.normal_(self.fundamental, 0, math.pi * math.pi * math.sqrt(2 / 45))
        init.constant_(
            self.fundamental, 0.66549 * math.sqrt(6.0) / math.pi
        )  # sqrt((pi^2 /6) - zeta(4))
        init.constant_(self.phases, math.pi / 4)

        self.coefficients = nn.Parameter(
            (1 / torch.arange(1, degree + 1) ** (3 / 2)) * math.sqrt(6.0) / math.pi,
            requires_grad=requires_grad,
        )
        # to have same gain as GELU, useful for drop-in replacement
        grid = torch.arange(1, self.degree + 1) * math.pi / math.sqrt(3)
        # self.register_buffer("grid", grid)
        self.grid = nn.Parameter(grid, requires_grad=requires_grad)

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
                self.phases = nn.Parameter(
                    cached_data["phases"], requires_grad=self.requires_grad
                )
                self.fundamental = nn.Parameter(
                    cached_data["fundamental"], requires_grad=self.requires_grad
                )
                self.grid = nn.Parameter(
                    cached_data["grid"], requires_grad=self.requires_grad
                )
                return
        # Compute the coefficients and frequencies at init with a Hermite interpolation
        lim = self.degree * 2
        act = self.act_init
        coefficients = nn.Parameter(
            self.coefficients.clone() + torch.randn_like(self.coefficients)
        )
        grid = nn.Parameter(self.grid.clone() + torch.randn_like(self.grid))
        phases = nn.Parameter(self.phases.clone() + torch.randn_like(self.phases))
        fundamental = nn.Parameter(
            self.fundamental.clone() + torch.randn_like(self.fundamental)
        )

        def model():
            x = (torch.rand(2**8) - 0.5) * 2 * lim
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

            x = x.unsqueeze(-1)
            z = grid * x - phases
            x = z.cos()
            x = (x * coefficients).sum(-1) + fundamental

            x_deriv = z + math.pi / 2
            x_deriv = x_deriv.cos()
            x_deriv = (x_deriv * coefficients * grid).sum(-1)
            return x, x_deriv, y, y_deriv

        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = torch.optim.Adam([fundamental, grid, phases, coefficients], lr=0.1)
        loss_min = 1e8
        coefficients_min = None
        phases_min = None
        grid_min = None
        fundamental_min = None

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
                phases_min = phases.clone().detach()
                fundamental_min = fundamental.clone().detach()
                grid_min = grid.clone().detach()

            if loss < 1e-4:
                coefficients_min = coefficients.clone().detach()
                phases_min = phases.clone().detach()
                fundamental_min = fundamental.clone().detach()
                grid_min = grid.clone().detach()
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
                    "phases": phases_min,
                    "fundamental": fundamental_min,
                    "grid": grid_min,
                },
                f,
            )
            print("Cached coefficients to", self.cache_file)
        self.coefficients = nn.Parameter(
            coefficients_min, requires_grad=self.requires_grad
        )
        self.phases = nn.Parameter(phases_min, requires_grad=self.requires_grad)
        self.fundamental = nn.Parameter(
            fundamental_min, requires_grad=self.requires_grad
        )
        self.grid = nn.Parameter(grid_min, requires_grad=self.requires_grad)

    def fourier_polynomials(self, x):
        """
        Compute the Fourier polynomial using the explicit formula.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Evaluated Fourier polynomial.
        """
        x = x.unsqueeze(-1)
        x = self.grid * x - self.phases
        x = x.cos()
        return x

    def forward(self, x):
        """
        Forward pass for the Fourier activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying Fourier activation function.
        """
        # Calculate the Fourier polynomial
        x = self.fourier_polynomials(x)
        # Multiply by learnable coefficients and sum
        x = (x * self.coefficients).sum(-1) + self.fundamental
        return x

    def extra_repr(self) -> str:
        return f"degree={self.degree}"
