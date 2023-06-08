
import torch
import torchvision.transforms as transforms
from PIL import Image

# TODO determine how to structure this code, how it will be used by training session
# - option: make a class to hold the options, and a choose method that essentially routes string to bound var with a dict

base_transform = transforms.Compose([
    # Resize the image to have shorter side randomly sampled in [256, 480]. Then flip with p = 0.5 and randomly crop to 224x224
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    # per-pixel mean subtraction
    # standard color augmentation
    transforms.ToTensor(),
])




# Temporarily in this file
class ResNetResizeFlipCrop(object):
    def __init__(self) -> None:
        assert ...

    def __call__(self, img: Image.Image) -> Image.Image:
        pass
        # Resize the image to have shorter side randomly sampled in [256, 480]

        # Flip with p = 0.5

        # Randomly crop to 224x224

        return 