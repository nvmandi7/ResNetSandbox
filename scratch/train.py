
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.resnet.resnet18 import ResNet18
from src.training.trainer import Trainer

# Run a training loop as a simple script here,
# then replicate that functionality in training session with a config
# this script is essentially the run method of training session

# imagenet dataset
    # dataset = torch.utils.data.Dataset(torchvision.datasets.ImageNet(root="data/tiny-imagenet-200", split="train"))
    # dataset = CIFAR10(root='data/', download=True, transform=ToTensor())

def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Resize(size=256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(
        root='data/tiny-imagenet-200/train',
        transform=transform,
    )

    # dataset = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train", transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


    # model
    model = ResNet18(3)
    # model = torchvision.models.resnet18(num_classes=200)

    # optimization
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 3
    torch.autograd.set_detect_anomaly(True)

    outputs = {}
    def hook(module, input, output):
        outputs[module.__class__.__name__] = output
        print("HOOK OUTPUT: ", output.detach())

    model.stem.register_forward_hook(hook)
    model.block_stack.register_forward_hook(hook)
    model.classification_head.register_forward_hook(hook)

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )

    trainer.train()


if __name__ == "__main__":
    main()