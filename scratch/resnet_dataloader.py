
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from resnet.resnet_dataset import ResNetDataset
import tqdm


class ResNetDataloader(DataLoader):
    def __init__(self, dataset: ResNetDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> None:
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.transforms = transforms.Compose([
            # Resize the image to have shorter side randomly sampled in [256, 480]. Then flip with p = 0.5 and randomly crop to 224x224
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            # per-pixel mean subtraction
            # standard color augmentation
            transforms.ToTensor(),
        ])

    def __iter__(self):
        for batch in tqdm(super().__iter__()):
            x, y = batch
            x = self.transforms(x)
            yield x, y





# Temporarily in this file
class ResNetResizeFlipCrop(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: Image.Image) -> Image.Image:
        pass
        # Resize the image to have shorter side randomly sampled in [256, 480]

        # Flip with p = 0.5

        # Randomly crop to 224x224