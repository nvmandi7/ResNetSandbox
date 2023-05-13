
import torch
from torch import nn
from src.resnet.blocks.block import Block

class ResNet18WithoutBase(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block args: kernel_size, in_channels, out_channels
        self.block2_1 = Block(3, 64, 64)
        self.block2_2 = Block(3, 64, 64)

        self.block3_1 = Block(3, 64, 128, downsample=True)
        self.block3_2 = Block(3, 128, 128)

        self.block4_1 = Block(3, 128, 256, downsample=True)
        self.block4_2 = Block(3, 256, 256)

        self.block5_1 = Block(3, 256, 512, downsample=True)
        self.block5_2 = Block(3, 512, 512)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Manipulate x and return modified version
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 224)

        x = self.stem(x)
        assert x.shape == (batch_size, 64, 56, 56)

        x = self.block2_1(x)
        x = self.block2_2(x)
        assert x.shape == (batch_size, 64, 56, 56)

        x = self.block3_1(x)
        x = self.block3_2(x)
        assert x.shape == (batch_size, 128, 28, 28)

        x = self.block4_1(x)
        x = self.block4_2(x)
        assert x.shape == (batch_size, 256, 14, 14)

        x = self.block5_1(x)
        x = self.block5_2(x)
        assert x.shape == (batch_size, 512, 7, 7)


        # Classification head
        x = self.avgpool(x)
        assert x.shape == (batch_size, 512, 1, 1)
        x = x.reshape(batch_size, -1)
        assert x.shape == (batch_size, 512)
        x = self.fc(x)
        assert x.shape == (batch_size, 1000)
        x = self.softmax(x)
        assert x.shape == (batch_size, 1000)
        return x
    

if __name__ == "__main__":
    model = ResNet18WithoutBase(3)
    x = torch.ones(1, 3, 224, 224)
    print(model)
    x = model(x)
    print(f"X shape: {x.shape}")

