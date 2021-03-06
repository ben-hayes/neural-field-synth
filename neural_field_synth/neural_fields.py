import numpy as np
import torch
import torch.nn as nn

"""
SIREN
"""


class BaseSineLayer(nn.Module):
    """Base class for SIREN layers"""

    def __init__(
        self,
        in_features: torch.tensor,
        out_features: torch.tensor,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )


class SineLayer(BaseSineLayer):
    """Implements the basic Sine layer at the core of SIREN"""

    def forward(self, input: torch.tensor) -> torch.tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class SineFiLMLayer(BaseSineLayer):
    """Implements the basic Sine layer at the core of SIREN"""

    def forward(
        self, input: torch.tensor, gamma: torch.tensor, beta: torch.tensor
    ) -> torch.tensor:
        # TODO: compare omega_0 * (x + beta) to (omega_0 * x) + beta
        return torch.sin(self.omega_0 * self.linear(input) * gamma + beta)


class BaseSiren(nn.Module):
    """Base class for SIREN models"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        first_omega_0: int = 30,
        hidden_omega_0: int = 30.0,
        freeze: bool = False,
        layer=SineLayer,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            layer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)
        )
        if freeze:
            self.net[-1].linear.weight.requires_grad_(False)
            self.net[-1].linear.bias.requires_grad_(False)

        for _ in range(hidden_layers):
            self.net.append(
                layer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
            if freeze:
                self.net[-1].linear.weight.requires_grad_(False)
                self.net[-1].linear.bias.requires_grad_(False)

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            if freeze:
                final_linear.weight.requires_grad_(False)
                final_linear.bias.requires_grad_(False)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                layer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )


class Siren(BaseSiren):
    """Basic SIREN implementation"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        first_omega_0: int = 30,
        hidden_omega_0: int = 30.0,
        freeze: bool = False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
            freeze,
            layer=SineLayer,
        )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords: torch.tensor) -> torch.tensor:
        output = self.net(coords)
        return output


class SirenFiLM(BaseSiren):
    """FiLM conditioned SIREN implementation"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        first_omega_0: int = 30,
        hidden_omega_0: int = 30.0,
        freeze: bool = False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
            freeze,
            layer=SineFiLMLayer,
        )

        self.net = nn.ModuleList(self.net)

    def forward(
        self, coords: torch.tensor, scale: torch.tensor, shift: torch.tensor
    ) -> torch.tensor:
        x = coords
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x, scale[..., i, :], shift[..., i, :])
        output = self.net[-1](x)

        return output
