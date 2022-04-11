from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from pytorch_lightning import Trainer
    BenchoptCallback = import_ctx.import_from(
        'torch_helper', 'BenchoptCallback'
    )


class TorchSolver(BaseSolver):
    """Torch base solver"""

    stopping_strategy = 'callback'

    parameters = {
        'batch_size': [64],
    }

    def set_objective(self, pl_module, torch_dataset, tf_model, tf_dataset):
        self.pl_module = pl_module
        self.dataloader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=self.batch_size
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.pl_module)

        # Setup the trainer
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)],
            accelerator="gpu" if torch.cuda.is_available() else None
        )
        trainer.fit(model=self.pl_module, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.pl_module
