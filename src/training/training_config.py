
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    # Dataset
    
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_workers: int = 4


if __name__ == "__main__":
    config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "num_workers": 4
    }
    training_config = TrainingConfig(**config)

    print(training_config.learning_rate)