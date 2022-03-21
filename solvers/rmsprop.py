from .base_nn_solver import BaseNNSolver
from torch.optim import RMSprop


class Solver(BaseNNSolver):
    """RMSPROP solver"""
    name = 'RMSPROP'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [0.99, 0.9],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    def skip(self, pl_module, trainer):
        return False, None

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: RMSprop(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
        )
