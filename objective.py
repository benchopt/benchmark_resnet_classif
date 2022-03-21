from benchopt import BaseObjective
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


class Objective(BaseObjective):
    """Classification objective"""
    name = "ResNet classification fitting"

    # XXX: this might be a good spot to specify the size of the ResNet
    parameters = {
    }


    def __init__(self, _debug=False):
        self.model = models.resnet18()
        self._debug = _debug
        self.trainer = Trainer()

    def set_data(self, dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=64)
        self.pl_module = BenchPLModule(self.model, self.data_loader)

    def compute(self, pl_module):
        loss = self.trainer.test(pl_module)
        return loss

    def to_dict(self):
        return dict(
            pl_module=self.pl_module,
            trainer=self.trainer,
        )


class BenchPLModule(LightningModule):
    """Lightning module for benchopt"""
    def __init__(self, model, loader):

        super().__init__()
        self.model = model
        self.loader = loader


    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def train_dataloader(self):
        return super().train_dataloader()

    def test_dataloader(self):
        return self.test_loader
