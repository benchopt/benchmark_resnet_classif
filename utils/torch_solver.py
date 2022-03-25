from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:

    from pytorch_lightning import Trainer
    BenchoptCallback = import_ctx.import_from(
        'torch_helper', 'BenchoptCallback'
    )


class TorchSolver(BaseSolver):
    """Torch base solver"""

    stopping_strategy = 'callback'

    def set_objective(self, pl_module, trainer, tf_model, tf_dataset):
        self.pl_module = pl_module
        self.main_trainer = trainer  # we use this in order
        # to access some elements from the trainer when
        # initializing it below

    def run(self, callback):
        # Initial evaluation
        callback(self.pl_module, None)

        # Setup the trainer
        trainer = Trainer(
            max_epochs=-1, callbacks=[BenchoptCallback(callback)]
        )
        trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module, None
