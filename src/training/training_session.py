
import torch
from torch import nn
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig
from src.transforms.base_transform import base_transform

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import SGD



class TrainingSession:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        self.configure_device()
        self.set_up_data()
        self.create_model()
        self.configure_training()
        self.create_trainer()
        
    def configure_device(self):
        if self.config.device == "auto":
            self.config.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.config.device = torch.device(self.config.device)

    def set_up_data(self):
        self.dataset = ImageFolder(
            root=self.config.dataset_path,
            transform=base_transform, # TODO make configurable
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
    
    def create_model(self):
        self.model = eval(self.config.model)(in_channels=3)
    
    def configure_training(self):
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        ...


    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            dataloader=self.dataloader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.epochs
        )



if __name__ == "__main__":
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()