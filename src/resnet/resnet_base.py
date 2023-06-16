
import torch
from torch import nn
import torch.nn.init as init

class ResNetBase(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()

        self._set_stem(in_channels)
        self._set_block_stack()
        assert isinstance(self.block_stack, nn.Sequential)

        self.final_channels = self.block_stack[-1].out_channels
        self._set_classification_head()
        # self._init_weights()


    def _set_stem(self, in_channels):
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def _set_block_stack(self):
        raise NotImplementedError

    def _set_classification_head(self):
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(self.final_channels, 1000),
            nn.Softmax(dim=1)
        )

    def _init_weights(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                init.xavier_uniform_(m.weight, gain=0.1)
            # if hasattr(m, "bias"):
            #     init.constant_(m.bias, 0.0)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 224)

        x = self.stem(x)
        assert x.shape == (batch_size, 64, 56, 56)

        x = self.block_stack(x)
        assert x.shape == (batch_size, self.final_channels, 7, 7)

        # Classification head
        x = self.classification_head(x)
        assert x.shape == (batch_size, 1000)

        return x
    

if __name__ == "__main__":
    try:
        model = ResNetBase(3)
    except NotImplementedError:
        print("ResNetBase is an abstract class")

