from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    from torch.nn import functional as F

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
