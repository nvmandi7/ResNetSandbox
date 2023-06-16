
import pytest
import torch
from src.transforms.base_transform import BaseTransform

def test_base_transform():
    x = torch.rand(32, 3, 224, 224)
    transform = BaseTransform()
    x = transform(x)
    assert x.shape == (32, 3, 240, 240)
