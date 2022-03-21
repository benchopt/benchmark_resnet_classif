from .base_nn_solver import BaseNNSolver
from torch.optim import SGD


class Solver(BaseNNSolver):
    """Stochastic Gradient descent solver"""
    name = 'SGD'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov': [False, True],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    def skip(self, pl_module, trainer):
        if not self.momentum and self.nesterov:
            return True, 'Nesterov cannot be used without momentum'

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: SGD(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
