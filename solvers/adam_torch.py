from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import Adam

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Adam solver"""
    name = 'Adam-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
        **TorchSolver.parameters
    }

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )
