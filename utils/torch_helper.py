from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset

    from torchmetrics import Accuracy
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import Callback


# Convert benchopt benchmark into a lightning callback, used to monitor the
# objective and to stop the solver when needed.
class BenchoptCallback(Callback):
    def __init__(self, callback):
        super().__init__()
        self.cb_ = callback

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = not self.cb_(pl_module)


class BenchPLModule(LightningModule):
    """Lightning module for benchopt inspired by
    https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb#scrollTo=bd97d928
    """
    def __init__(self, model):

        super().__init__()
        self.model = model
        self.accuracy = Accuracy()
        self.loss_type = 'nll'

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def loss_logits_y(self, batch):
        x, y = batch
        logits = self(x)
        if self.loss_type == 'nll':
            loss = F.nll_loss(logits, y)
        elif self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
        return loss, logits, y

    def test_step(self, batch, batch_idx):
        old_loss_type = self.loss_type
        self.loss_type = 'nll'
        loss, logits, y = self.loss_logits_y(batch)
        self.loss_type = old_loss_type
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("loss", loss, prog_bar=True)
        self.log("acc", self.accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss_logits_y(batch)[0]


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform, normalization):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.normalization = normalization

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.normalization:
            x = self.normalization(x)
        return x, y
