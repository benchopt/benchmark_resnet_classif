from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import RMSprop

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

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: RMSprop(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
        )
