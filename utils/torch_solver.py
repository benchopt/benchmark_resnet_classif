from benchopt import BaseSolver

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


class TorchSolver(BaseSolver):
    """Torch base solver"""

    stopping_strategy = 'callback'

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.main_trainer = trainer  # we use this in order
        # to access some elements from the trainer when
        # initializing it below

    def run(self, callback):
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)]
        )
        trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module


# Convert benchopt benchmark into a lightning callback, used to monitor the
# objective and to stop the solver when needed.
class BenchoptCallback(Callback):
    def __init__(self, callback):
        super().__init__()
        self.cb_ = callback

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = not self.cb_(pl_module)
