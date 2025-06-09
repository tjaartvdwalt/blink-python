import lightning as L
import torch
import torchmetrics


class BlinkModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(13, 13),
            torch.nn.ReLU(),
            torch.nn.Linear(13, 13),
            torch.nn.ReLU(),
            torch.nn.Linear(13, 13),
            torch.nn.ReLU(),
            torch.nn.Linear(13, 2),
        )

        self.example_input_array = torch.randn(1, 13)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        target_max = torch.nn.functional.softmax(y, dim=1)
        prob = torch.nn.functional.softmax(y_hat, dim=1)

        target = target_max.data.max(dim=1)[1]
        pred = prob.data.max(dim=1)[1]

        acc = self.accuracy(pred, target)
        return loss, acc

    #
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x, y = batch
    #     y_hat = self.model(x)
    #     return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
