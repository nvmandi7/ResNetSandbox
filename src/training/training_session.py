
import torch
from torch import nn
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig


class TrainingSession:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        self.extract_fields_from_config()
        self.configure_device()
        self.create_trainer()
        
    # Perhaps organize into data, model, training, etc.
    def extract_fields_from_config(self):
        self.model = self.config.model
        self.train_dataloader = self.config.train_dataloader
        self.optimizer = self.config.optimizer
        ...

    def configure_device(self):
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            optimizer=self.optimizer,
            config=self.config,
        )



if __name__ == "__main__":
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()