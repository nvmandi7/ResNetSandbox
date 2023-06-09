
import importlib
import torch
from torch import transforms

class BaseTransform:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x)


class TransformFactory:
    '''
    Choose from a set of prebuilt transforms by name
    '''



    @classmethod
    def transform_from_name(cls, class_name: str):
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        instance = class_()


    transforms = {
        'null_transform': TransformSelector.create_null_transform,
        'base_transform': TransformSelector.create_base_transform,
        'original_paper': TransformSelector.create_original_paper,
    }

    @classmethod
    def parse_transform(cls, name):
        t_func = cls.transforms[name]
        return t_func()
    
    @classmethod
    def create_null_transform():
        return transforms.toTensor()
    
    @classmethod
    def create_base_transform():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
    
    @classmethod
    def create_original_paper():
        return transforms.Compose([ #TODO
            # Resize the image to have shorter side randomly sampled in [256, 480]. Then flip with p = 0.5 and randomly crop to 224x224
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            # per-pixel mean subtraction
            # standard color augmentation
            transforms.ToTensor(),
        ])