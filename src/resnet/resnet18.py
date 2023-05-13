
import torch
from torch import nn
from pytorch_model_summary import summary
from src.resnet.resnet_base import ResNetBase
from src.resnet.blocks.block import Block


class ResNet18(ResNetBase):

    def _set_block_stack(self):
        self.block_stack = nn.Sequential(
            Block(3, 64, 64),
            Block(3, 64, 64),

            Block(3, 64, 128, downsample=True),
            Block(3, 128, 128),

            Block(3, 128, 256, downsample=True),
            Block(3, 256, 256),

            Block(3, 256, 512, downsample=True),
            Block(3, 512, 512),
        )

    

if __name__ == "__main__":
    model = ResNet18(3)
    x = torch.ones(1, 3, 224, 224)
    print(summary(model, x))
    x = model(x)
    print(f"X shape: {x.shape}")

