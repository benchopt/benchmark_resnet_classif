from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    from pytorch_lightning import LightningModule, Trainer

    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    import torchvision.models as models


class Objective(BaseObjective):
    """Classification objective"""
    name = "ResNet classification fitting"

    # XXX: this might be a good spot to specify the size of the ResNet
    parameters = {
        'batch_size': [64],
    }

    def __init__(self, batch_size=64):
        # XXX: seed everything correctly
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
        # XXX: modify this with the correct amount of CPUs/GPUs
        self.trainer = Trainer()
        self.batch_size = batch_size

    def set_data(self, dataset):
        self.dataset = dataset

    def compute(self, pl_module):
        loss = self.trainer.test(pl_module)
        # XXX: allow to return accuracy as well
        # this will allow to have a more encompassing benchmark that also
        # captures speed on accuracy
        return loss[0]['train_loss']

    def to_dict(self):
        model = models.resnet18()
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        pl_module = BenchPLModule(model, data_loader)
        return dict(
            pl_module=pl_module,
            trainer=self.trainer,
        )


class BenchPLModule(LightningModule):
    """Lightning module for benchopt inspired by
    https://colab.research.google.com/github/PyTorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb#scrollTo=bd97d928
    """
    def __init__(self, model, loader):

        super().__init__()
        self.model = model
        self.loader = loader

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def train_dataloader(self):
        return self.loader

    def test_dataloader(self):
        return self.loader
