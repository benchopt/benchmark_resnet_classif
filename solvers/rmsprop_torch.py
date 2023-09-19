from benchopt import safe_import_context

from benchmark_utils.torch_solver import TorchSolver

with safe_import_context() as import_ctx:
    from torch.optim import RMSprop


class Solver(TorchSolver):
    """RMSPROP solver"""
    name = 'RMSPROP-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TorchSolver.parameters,
        'lr': [1e-3],
        'rho': [0.99, 0.9],
        'momentum': [0, 0.9],
        'coupled_weight_decay': [0.0, 1e-4, 0.02],
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        self.optimizer_klass = RMSprop
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.rho,
            weight_decay=self.coupled_weight_decay,
        )
