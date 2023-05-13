
import torch
from torch import nn
from src.resnet.blocks.block import Block

class Bottleneck(Block):
    def __init__(self, kernel_size, in_channels, bottleneck_channels, out_channels, downsample=False) -> None:
        self.bottleneck_channels = bottleneck_channels
        super().__init__(kernel_size, in_channels, out_channels, downsample)

    def _set_residual_block(self):
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=self.first_stride, bias=False),
            nn.BatchNorm2d(num_features=self.bottleneck_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.bottleneck_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False),
            nn.BatchNorm2d(num_features=self.bottleneck_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )


if __name__ == "__main__":
    batch_size = 100

    model = Bottleneck(3, 256, 64, 256)
    x = torch.ones(batch_size, 256, 56, 56)
    print(model)
    x = model(x)
    print(x.shape)
    assert x.shape == (batch_size, 256, 56, 56)

    model = Bottleneck(3, 512, 256, 1024, downsample=True)
    x = torch.ones(batch_size, 512, 14, 14)
    print(model)
    x = model(x)
    print(x.shape)
    assert x.shape == (batch_size, 1024, 7, 7)

    print("Bottleneck tests passed")