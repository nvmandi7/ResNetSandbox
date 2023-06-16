
import torch
from torch import transforms

class BaseTransform:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(240, padding=4),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x)