
import torch
from torch import nn
from resnet_dataloader import ResNetDataloader


class Trainer:
    def __init__(self, model: torch.nn.Module, dataloader: ResNetDataloader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer # TODO
        self.device = device


    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            running_loss = 0.0
            for i, batch in enumerate(self.dataloader, 0):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199: # Print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
            
        
        