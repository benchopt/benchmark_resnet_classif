from benchopt import safe_import_context
from torch.optim import SGD
with safe_import_context() as import_ctx:
    TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')

class Solver(TorchSolver):
    """Stochastic Gradient descent solver"""
    name = 'SGD'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov': [False, True],
        'lr': [1e-3],
        'momentum': [0, 0.9],
    }

    stopping_strategy = 'iteration'

    def skip(self, pl_module, trainer):
        if not self.momentum and self.nesterov:
            return True, 'Nesterov cannot be used without momentum'
        return False, None

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: SGD(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
