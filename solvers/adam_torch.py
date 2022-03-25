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
    }

    def set_objective(self, pl_module, trainer):
        super().set_objective(pl_module, trainer)
        self.pl_module.configure_optimizers = lambda: Adam(
            self.pl_module.parameters(),
            lr=self.lr,
        )
