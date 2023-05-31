from benchopt import safe_import_context

from benchmark_utils.torch_solver import TorchSolver

with safe_import_context() as import_ctx:
    from torch.optim import Adam, AdamW


class Solver(TorchSolver):
    """Adam solver"""
    name = 'Adam-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TorchSolver.parameters,
        'lr': [1e-3],
        'coupled_weight_decay': [0.0, 0.02],
        'decoupled_weight_decay': [0.0, 0.02],
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        self.optimizer_klass = Adam
        wd = self.coupled_weight_decay
        if self.decoupled_weight_decay > 0:
            self.optimizer_klass = AdamW
            wd = self.decoupled_weight_decay
        self.optimizer_kwargs = dict(
            lr=self.lr,
            weight_decay=wd,
        )
