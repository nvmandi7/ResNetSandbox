
import torch

from src.transforms.base_transform import base_transform
import torchvision





class TrainingUtils:

    def get_dataset(self, config):
        return torchvision.datasets.ImageFolder(
            root=config.dataset_path,
            transform=base_transform, # TODO make congifurable
        )
    