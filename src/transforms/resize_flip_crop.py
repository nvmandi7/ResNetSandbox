
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class ResNetResizeFlipCrop(object):
    def __init__(self) -> None:
        assert ...

    def __call__(self, img: Image.Image) -> Image.Image:
        short_size = np.randint(256, 480)
        if img.h > img.w:
            img = transforms.Resize(short_size)(img)
            
        # Resize the image to have shorter side randomly sampled in [256, 480]

        # Flip with p = 0.5

        # Randomly crop to 224x224

        return 