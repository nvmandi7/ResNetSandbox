
import torch

from src.transforms.resize_flip_crop import base_transform
import torchvision





class TrainingUtils:

    def get_dataset(self, config):
        return torchvision.datasets.ImageFolder(
            root=config.dataset_path,
            transform=base_transform, # TODO make congifurable
        )
    