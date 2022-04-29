from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import torch
    from pytorch_lightning import Trainer

    BenchoptCallback = import_ctx.import_from(
        'torch_helper', 'BenchoptCallback'
    )


class TorchSolver(BaseSolver):
    """Torch base solver"""

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    parameters = {
        'batch_size': [64],
    }

    def set_objective(self, pl_module, torch_dataset, tf_model, tf_dataset):
        self.pl_module = pl_module
        # TODO: num_worker should not be hard coded. Finding a sensible way to
        # set this value is necessary here.
        self.dataloader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=self.batch_size,
            num_workers=6,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.pl_module)

        # Setup the trainer
        # TODO: for now, we are limited to 1 device due to pytorch_lightning
        # bad interaction with benchopt. Removing this limitation would be
        # nice to allow multi-GPU training.
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)],
            accelerator="auto", devices=1
        )
        trainer.fit(model=self.pl_module, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.pl_module
