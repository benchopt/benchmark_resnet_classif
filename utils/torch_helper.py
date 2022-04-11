import logging
from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import torch
    from torch.nn import functional as F

    from torchmetrics import Accuracy
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.strategies import DDPStrategy
    from torch.nn.parallel.distributed import DistributedDataParallel
    from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_11
    from pytorch_lightning.utilities.rank_zero import rank_zero_info
    from pytorch_lightning.trainer.states import TrainerFn

log = logging.getLogger(__name__)


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

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def loss_logits_y(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss, logits, y

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.loss_logits_y(batch)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("loss", loss, prog_bar=True)
        self.log("acc", self.accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self.loss_logits_y(batch)[0]


class DDPStrategyNoTeardown(DDPStrategy):
    # from
    # https://pytorch-lightning.readthedocs.io/en/1.6.0/_modules/pytorch_lightning/strategies/ddp.html#DDPStrategy.teardown
    def teardown(self):
        log.detail(f"{self.__class__.__name__}: tearing down strategy")
        super().teardown()

        if isinstance(self.model, DistributedDataParallel):
            if (
                _TORCH_GREATER_EQUAL_1_11
                and not self.model.static_graph
                and self.model._get_ddp_logging_data().get(
                    "can_set_static_graph",
                )
            ):
                rank_zero_info(
                    "Your model can run with static graph optimizations."
                    "For future training runs, we suggest you"
                    f" pass `Trainer(..., strategy={self.__class__.__name__}"
                    "(static_graph=True))` to enable them."
                )
            # unwrap model
            self.model = self.lightning_module

        if (
            self.lightning_module.trainer is not None
            and self.lightning_module.trainer.state.fn == TrainerFn.FITTING
            and self._layer_sync
        ):
            # `self.lightning_module.trainer` can be None if teardown gets
            # called on an exception before
            # the trainer gets set on the LightningModule
            self.model = self._layer_sync.revert(self.model)

        if self.root_device.type == "cuda":
            # GPU teardown
            log.detail(f"{self.__class__.__name__}: !not! moving model to CPU")
            # clean up memory
            torch.cuda.empty_cache()
