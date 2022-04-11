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

    def set_objective(self, model, dataset):
        self.model = model
        self.dataset = dataset  # we use this in order
        # to access some elements from the trainer when
        # initializing it below
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.model)

        # Setup the trainer
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)]
        )
        trainer.fit(self.model, train_dataloaders=self.dataloader)

    def get_result(self):
        return self.model
