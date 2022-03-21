from benchopt import BaseSolver
from torch.optim import Adam


class Solver(BaseSolver):
    """Adam solver, optionally accelerated."""
    name = 'Adam'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
    }

    def skip(self, pl_module, trainer):
        pass

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.trainer = trainer
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )


    def run(self, n_iter):
        self.trainer.max_steps = n_iter
        self.trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
