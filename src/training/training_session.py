
import torch
from torch import nn
from src.training.trainer import Trainer
from src.training.training_config import TrainingConfig
from src.transforms.transform_factory import TransformFactory
from src.transforms.debug_selector import DebugSelector

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim



class TrainingSession:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        self.configure_device()
        self.set_up_data()
        self.create_model()
        self.configure_training()
        self.create_trainer()
        self.configure_debug()
        self.trainer.train()
        
    def configure_device(self):
        if self.config.device == "auto":
            self.config.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.config.device = torch.device(self.config.device)

    def set_up_data(self):
        transform = TransformFactory.transform_from_name(config.transform_name)
        # transform = TransformSelector.parse_transform(config.transform)
        self.dataset = ImageFolder(
            root=self.config.dataset_path,
            transform=transform,
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
        self.loss_fn = eval('nn.' + config.loss_fn)()

        self.optimizer = eval('optim.' + config.optimizer)(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            dataloader=self.dataloader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.epochs
        )
    
    def configure_debug(self):
        if config.debug:
            outputs = DebugSelector.parse_debug(config.debug, self.model)



if __name__ == "__main__":
    config = TrainingConfig.parse_args()
    session = TrainingSession(config)
    session.run()