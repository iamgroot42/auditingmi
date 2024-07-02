"""
    Basic N-layer MLPs
"""
import torch.nn as nn
import copy
from typing import List


class MLP(nn.Module):
    """
    Generic MLP class wrapper.
    """
    def __init__(self, num_features:int, layer_depths: List[int], num_classes: int, add_sigmoid: bool = False):
        super().__init__()
        self.layer_depths = layer_depths
        if add_sigmoid and num_classes != 1:
            raise ValueError(f"Only working with single-class classification. Are you sure you want to add a sigmoid for {num_classes} classes?")

        # Add first layer
        if len(layer_depths) > 0:
            self.layers = [nn.Linear(num_features, layer_depths[0]), nn.ReLU()]
        else:
            self.layers = [nn.Linear(num_features, num_classes)]

        # Add intermediate layers
        for i in range(1, len(layer_depths)):
            self.layers.append(nn.Linear(layer_depths[i-1], layer_depths[i]))
            self.layers.append(nn.ReLU())

        # Final layer
        if len(layer_depths) > 0:
            self.layers.append(nn.Linear(layer_depths[-1], num_classes))
        if add_sigmoid:
            self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)

        # 600, relu, 512, relu, 256, relu, 128, relu, 64, relu, 100
        # 0     1     2    3     4    5     6    7    8    9,    10

    def make_later_layer_model(self, layer_readout: int):
        # Split existing models into two at layer_readout, create a new model for the second part
        # with existing parameters and return that
        # Make sure to copy the parameters
        if layer_readout >= len(self.layers) or layer_readout < 0:
            raise ValueError(f"Invalid layer_readout: {layer_readout}, should be between 0 and {len(self.layers) - 1}")
        later_layers = copy.deepcopy(self.layers[layer_readout:])
        new_model = nn.Sequential(*later_layers)
        return new_model

    def forward(self, x,
                get_all: bool = False,
                layer_readout: int = None):

        all_embeds = []
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if layer_readout == i:
                return out
            if get_all and i != len(self.layers) - 1 and i % 2 == 1:
                # Last layer is logits, will collect that anyway
                all_embeds.append(out)

        if get_all:
            return all_embeds

        return out


class MLPQuadLoss(MLP):
    def __init__(self, num_features:int, layer_depths: List[int], num_classes: int):
        super().__init__(num_features, layer_depths, num_classes, add_sigmoid=True)


if __name__ == "__main__":
    # Test MLP
    mlp = MLP(100, [32, 16, 4], 3)
    print(mlp.layers)
    print(mlp.make_later_layer_model(4))
