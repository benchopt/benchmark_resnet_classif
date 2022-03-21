from benchopt import BaseSolver
from torch.optim import SGD


class Solver(BaseSolver):
    """Stochastic Gradient descent solver, optionally accelerated."""
    name = 'SGD'

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, pl_module, trainer):
        self.pl_module = pl_module
        self.trainer = trainer
        self.pl_module.configure_optimizers = lambda: SGD(
            self.pl_module.parameters(),
            lr=1e-3,
            nesterov=self.use_acceleration,
        )


    def run(self, n_iter):
        self.trainer.max_steps = n_iter
        self.trainer.fit(self.pl_module)

    def get_result(self):
        return self.pl_module
