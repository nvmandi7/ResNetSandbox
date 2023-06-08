
import torch
from torch import nn
from src.resnet.blocks.block import Block

'''
Block that contains Blocks within it's residual stack.
Higher order semantic features may still benefit from the regularizing nature of residual learning
'''
class Metablock(Block):
    def __init__(self, kernel_size, in_channels, out_channels, downsample=False) -> None:
        super().__init__(kernel_size, in_channels, out_channels, downsample)

    def _set_residual_block(self):
        self.residual_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.first_stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            # todo
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
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

    print("Metablock tests passed")