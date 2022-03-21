from .base_nn_solver import BaseNNSolver
from torch.optim import Adam


class Solver(BaseNNSolver):
    """Adam solver"""
    name = 'Adam'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
    }

    def skip(self, pl_module, trainer):
        return False, None

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )
