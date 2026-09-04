"""Passive BiV multilayer perceptron with optional layer normalization."""

from torch import nn

from pkg.train.layer.mlp_layer import MLPLayerBase
from pkg.train.module.activation import get_activation


class MLPLayer(MLPLayerBase):
    """Build a configured MLP and optionally normalize its final features."""

    def _init_graph(self) -> None:
        """Create fully connected blocks and the optional final layer norm."""
        for i in range(len(self._unit_sizes) - 1):
            cur_layer_name = f"{self._layer_name}_l{i + 1}"

            # Add the linear projection for this stage.
            self._init_fc(cur_layer_name, self._unit_sizes[i], self._unit_sizes[i + 1])

            # Apply the configured activation after each projection.
            if self._activation:
                self._mlp_layers.add_module(f"{cur_layer_name}_ac", get_activation(self._activation))

        # Normalize the final feature width when layer normalization is enabled.
        if self._layer_norm:
            self._mlp_layers.add_module(f"{self._layer_name}_ln", nn.LayerNorm(self._unit_sizes[-1], eps=1e-6))
