from benchopt import BaseSolver
from torch.optim import RMSPROP


class Solver(BaseSolver):
    """RMSPROP solver, optionally accelerated."""
    name = 'RMSPROP'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [0.99, 0.9],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    def skip(self, pl_module, trainer):
        pass

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.trainer = trainer
        self.pl_module.configure_optimizers = lambda: RMSPROP(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
        )


    def run(self, n_iter):
        self.trainer.max_steps = n_iter
        self.trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
