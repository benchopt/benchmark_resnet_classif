from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from torch.optim import SGD

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Stochastic Gradient descent solver"""
    name = 'SGD-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov, momentum': [(False, 0), (True, 0.9)],
        'lr': [1e-1],
        'weight_decay': [0.0, 5e-4],
        **TorchSolver.parameters
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        wd = self.weight_decay
        self.optimizer_klass = SGD
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=wd,
        )
