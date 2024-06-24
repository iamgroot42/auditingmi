"""
    Basic N-layer CNNs
"""

import torch.nn as nn


class CNN(nn.Module):
    """
    Generic CNN class wrapper.
    """

    def __init__(
        self,
        scales: int,
        filters: int,
        pooling: str,
        nin: int,
        num_classes: int,
        filters_max: int = 1024,
    ):
        super().__init__()
        def nf(scale):
            return min(filters_max, filters << scale)

        self.layers = [
            nn.Conv2d(in_channels=nin, out_channels=nf(0), kernel_size=3, padding="same"),
            nn.LeakyReLU(),
        ]

        def pooling_fn():
            if pooling == "max":
                return nn.MaxPool2d(2, 2)
            elif pooling == "avg":
                return nn.AvgPool2d(2, 2)
            else:
                raise ValueError(f"Pooling {pooling} not supported.")

        for i in range(scales):
            self.layers.extend([
                nn.Conv2d(in_channels=nf(i), out_channels=nf(i), kernel_size=3, padding="same"),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=nf(i), out_channels=nf(i + 1), kernel_size=3, padding="same"),
                nn.LeakyReLU(),
                pooling_fn()
            ])

        self.layers.append(
            nn.Conv2d(in_channels=nf(scales), out_channels=num_classes, kernel_size=3, padding="same")
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # Handle first layer
        out = self.layers[0](x)

        for layer in self.layers[1:]:
            out = layer(out)

        # Apply mean reduction over last 2 dimensions
        out = out.mean(dim=[2, 3])

        return out
