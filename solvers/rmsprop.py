from benchopt import safe_import_context
from torch.optim import RMSprop
with safe_import_context() as import_ctx:
    TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')

class Solver(TorchSolver):
    """RMSPROP solver"""
    name = 'RMSPROP'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [0.99, 0.9],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    stopping_strategy = 'iteration'

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
