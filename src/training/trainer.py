
import torch
from torch import nn


class Trainer:
    def __init__(self, model: torch.nn.Module, dataloader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

    def train(self):
        print(f"Starting training model {self.model.__class__.__name__}")
        print(f"Loss function: {self.loss_fn.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Total Batches per Epoch: {len(self.dataloader)}")

        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f"EPOCH {epoch+1}/{self.epochs}")
            self.train_epoch(epoch)
        
        torch.save(self.model.state_dict(), "models/model.pth")
        print("Finished Training")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_index, (input, targets) in enumerate(self.dataloader):
            input = input.to(self.device)
            targets = targets.to(self.device)

            output = self.model(input)
            loss = self.loss_fn(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # View progress every 50 mini-batches
            running_loss += loss.item()
            if batch_index % 50 == 0:
                print(f'[{epoch + 1}, {batch_index + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

                # Health check on gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(f"Gradient for {name} - min: {param.grad.min()} max: {param.grad.max()}")
        
        