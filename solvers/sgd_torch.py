from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from torch.optim import SGD

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Stochastic Gradient descent solver"""
    name = 'SGD'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov, momentum': [(False, 0), (True, 0.9)],
        'lr': [1e-3],
    }

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)

        self.pl_module.configure_optimizers = lambda: SGD(
            self.pl_module.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
