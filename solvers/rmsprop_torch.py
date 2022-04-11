from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import RMSprop

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """RMSPROP solver"""
    name = 'RMSPROP-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'alpha': [0.99, 0.9],
        'lr': [1e-3],
        'momentum': [0, 0.9],
        **TorchSolver.parameters
    }

    def set_objective(self, model, dataset):
        super().set_objective(model, dataset)
        self.model.configure_optimizers = lambda: RMSprop(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
        )
