
import torch
from torch import nn

from pytorch_model_summary import summary
from src.resnet.resnet_base import ResNetBase
from src.resnet.blocks.bottleneck import Bottleneck

class ResNet50(ResNetBase):

    def _set_block_stack(self):
        self.block_stack = nn.Sequential(
            Bottleneck(3, 64, 64, 256),
            Bottleneck(3, 256, 64, 256),
            Bottleneck(3, 256, 64, 256),

            Bottleneck(3, 256, 128, 512, downsample=True),
            Bottleneck(3, 512, 128, 512),
            Bottleneck(3, 512, 128, 512),
            Bottleneck(3, 512, 128, 512),

            Bottleneck(3, 512, 256, 1024, downsample=True),
            Bottleneck(3, 1024, 256, 1024),
            Bottleneck(3, 1024, 256, 1024),
            Bottleneck(3, 1024, 256, 1024),
            Bottleneck(3, 1024, 256, 1024),
            Bottleneck(3, 1024, 256, 1024),

            Bottleneck(3, 1024, 512, 2048, downsample=True),
            Bottleneck(3, 2048, 512, 2048),
            Bottleneck(3, 2048, 512, 2048),
        )

    

if __name__ == "__main__":
    model = ResNet50(3)
    x = torch.ones(1, 3, 224, 224)
    print(summary(model, x))
    x = model(x)
    print(f"X shape: {x.shape}")

