from dataclasses import dataclass

import lightning as L
import torch

from .data import BlinkDataset
from .model import BlinkModule


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs_count: int = 20
    init_learning_rate: float = 0.001
    log_interval: int = 5
    test_interval: int = 1

def run(blink_ear_file: str, non_blink_ear_file: str, model_file: str):
    torch.Generator().manual_seed(TrainingConfig.seed)

    dataset = BlinkDataset(blink_ear_file, non_blink_ear_file)

    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    model = BlinkModule()

    trainer = L.Trainer(max_epochs=TrainingConfig.epochs_count)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    model.to_onnx(model_file, export_params=True)
