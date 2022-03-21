from benchopt import BaseSolver
from pytorch_lightning import Trainer
from torch.optim import Adam


class Solver(BaseSolver):
    """Adam solver"""
    name = 'Adam'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
    }

    stopping_strategy = 'iteration'

    def skip(self, pl_module, trainer):
        return False, None

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.main_trainer = trainer  # we use this in order
        # to access some elements from the trainer when3
        # initializing it below
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )

    def run(self, n_iter):
        trainer = Trainer(max_steps=n_iter)
        trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
