from benchopt import safe_import_context
from torch.optim import Adam
with safe_import_context() as import_ctx:
    TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')

class Solver(TorchSolver):
    """Adam solver"""
    name = 'Adam'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
    }

    stopping_strategy = 'iteration'

    def skip(self, pl_module, trainer):
        return False, None

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )
