"""
torchortho - Learnable Tropical and Tropical Rational Activation Functions

Author: Ismail Khalfaoui-Hassani
Date: February 2025
License: GPL-3.0
Description: Implements Tropical polynomial and rational activations.

For reproducing experiments:
- Vision: https://github.com/K-H-Ismail/ConvNeXt-ortho
- Language: https://github.com/K-H-Ismail/pytorch-language-models
"""

import torch
import torch.nn as nn
import os
import pickle


class TropicalActivation(nn.Module):
    def __init__(self, degree):
        """
        Initializes the Tropical polynomial activation function.

        """
        super(TropicalActivation, self).__init__()
        self.degree = degree
        self.register_buffer("powers", torch.arange(0, degree + 1))
        self.coefficients = nn.Parameter(torch.ones(degree + 1))

    def forward(self, x):
        """
        Forward pass for the Tropical activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying Tropical activation function.
        """
        return (x.unsqueeze(-1) * self.powers + self.coefficients).max(-1)[0]


class TropicalRationalActivation(nn.Module):
    def __init__(self, degree, act_init=None, requires_grad=True, load_cached=True):
        """
        Initializes the Tropical rational activation function.

        """
        super(TropicalRationalActivation, self).__init__()
        self.degree = degree
        self.requires_grad = requires_grad
        self.cache_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"cache_tropical_rational_{degree}.pkl",
        )
        self.load_cached = load_cached
        self.powers_num = nn.Parameter(
            torch.randn(degree + 1), requires_grad=requires_grad
        )
        self.powers_denom = nn.Parameter(
            torch.randn(degree + 1), requires_grad=requires_grad
        )
        self.coefficients_num = nn.Parameter(
            torch.randn(degree + 1), requires_grad=requires_grad
        )
        self.coefficients_denom = nn.Parameter(
            torch.randn(degree + 1), requires_grad=requires_grad
        )
        if act_init:
            self.act_init = act_init
            self.init_coeffs()

    def init_coeffs(self):
        # Check for cached results
        if os.path.exists(self.cache_file) and self.load_cached:
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                print("Loaded cached coefficients")
                self.coefficients_num = nn.Parameter(
                    cached_data["coefficients_num"], requires_grad=self.requires_grad
                )
                self.coefficients_denom = nn.Parameter(
                    cached_data["coefficients_denom"], requires_grad=self.requires_grad
                )
                self.powers_num = nn.Parameter(
                    cached_data["powers_num"], requires_grad=self.requires_grad
                )
                self.powers_denom = nn.Parameter(
                    cached_data["powers_denom"], requires_grad=self.requires_grad
                )
                return
        # Compute the coefficients and frequencies at init with a Hermite interpolation
        lim = self.degree
        act = self.act_init
        coefficients_num = nn.Parameter(
            self.coefficients_num.clone() + torch.randn_like(self.coefficients_num)
        )
        coefficients_denom = nn.Parameter(
            self.coefficients_denom.clone() + torch.randn_like(self.coefficients_denom)
        )
        powers_num = nn.Parameter(
            self.powers_num.clone() + torch.randn_like(self.powers_num)
        )
        powers_denom = nn.Parameter(
            self.powers_denom.clone() + torch.randn_like(self.powers_denom)
        )

        def model():
            x = torch.linspace(
                torch.tensor(-lim), torch.tensor(lim), 2**8
            )  # (torch.rand(2**8) - 0.5) * 2 * lim
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

            x_num = x.unsqueeze(-1) * powers_num
            x_denom = x.unsqueeze(-1) * powers_denom
            x = (x_num + coefficients_num).max(-1)[0] - (
                x_denom + coefficients_denom
            ).max(-1)[0]

            x_deriv = None  # self.powers[index].to(x)

            return x, x_deriv, y, y_deriv

        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = torch.optim.Adam(
            [coefficients_num, coefficients_denom, powers_num, powers_denom], lr=0.01
        )
        loss_min = 1e8
        (
            coefficients_num_min,
            coefficients_denom_min,
            powers_num_min,
            powers_denom_min,
        ) = (
            None,
            None,
            None,
            None,
        )

        # Training loop
        num_epochs = 40000
        print("Init activation coefficients")
        for _ in range(num_epochs):
            # Forward pass
            y_pred, y_pred_deriv, y, y_deriv = model()

            # Compute loss
            loss = loss_fn(y_pred, y)  # + loss_fn(y_pred_deriv, y_deriv)

            if loss < loss_min:
                loss_min = loss
                coefficients_num_min = coefficients_num.clone().detach()
                coefficients_denom_min = coefficients_denom.clone().detach()
                powers_num_min = powers_num.clone().detach()
                powers_denom_min = powers_denom.clone().detach()

            if loss < 1e-4:
                coefficients_num_min = coefficients_num.clone().detach()
                coefficients_denom_min = coefficients_denom.clone().detach()
                powers_num_min = powers_num.clone().detach()
                powers_denom_min = powers_denom.clone().detach()
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
                    "coefficients_num": coefficients_num_min,
                    "coefficients_denom": coefficients_denom_min,
                    "powers_num": powers_num_min,
                    "powers_denom": powers_denom_min,
                },
                f,
            )
            print("Cached coefficients to", self.cache_file)
        (
            self.coefficients_num,
            self.coefficients_denom,
            self.powers_num,
            self.powers_denom,
        ) = (
            nn.Parameter(coefficients_num_min, requires_grad=self.requires_grad),
            nn.Parameter(coefficients_denom_min, requires_grad=self.requires_grad),
            nn.Parameter(powers_num_min, requires_grad=self.requires_grad),
            nn.Parameter(powers_denom_min, requires_grad=self.requires_grad),
        )

    def forward(self, x):
        """
        Forward pass for the Tropical activation function.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying Tropical activation function.
        """
        x_num = x.unsqueeze(-1) * self.powers_num
        x_denom = x.unsqueeze(-1) * self.powers_denom
        x = (x_num + self.coefficients_num).max(-1)[0] - (
            x_denom + self.coefficients_denom
        ).max(-1)[0]
        return x
