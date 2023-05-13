
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, downsample=False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size//2

        self.first_stride = 2 if downsample else 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = nn.Sequential()

        if downsample or self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.first_stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

        self._set_residual_block()

        self.final_activation = nn.ReLU()

    def _set_residual_block(self):
        self.residual_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.first_stride, padding=self.padding, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, stride=1, padding=self.padding, bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        assert in_channels == self.in_channels
        
        identity = x
        x = self.residual_block(x)
        
        # Adjust identity shape to match x, if necessary
        identity = self.shortcut(identity)
        assert x.shape == identity.shape

        x = self.final_activation(x + identity)
        assert x.shape == (batch_size, self.out_channels, height//self.first_stride, width//self.first_stride)

        return x



if __name__ == "__main__":
    model = Block(3, 64, 64)
    x = torch.ones(1, 64, 56, 56)
    print(model)
    x = model(x)
    print(x.shape)
    assert x.shape == (1, 64, 56, 56)

    model = Block(3, 64, 128, downsample=True)
    x = torch.ones(1, 64, 56, 56)
    print(model)
    x = model(x)
    print(x.shape)
    assert x.shape == (1, 128, 28, 28)

    print("Block tests passed")