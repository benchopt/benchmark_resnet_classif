from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import Adam, AdamW

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
        self.optimizer_klass = Adam
        wd = self.coupled_weight_decay
        if self.decoupled_weight_decay > 0:
            self.optimizer_klass = AdamW
            wd = self.decoupled_weight_decay
        self.optimizer_kwargs = dict(
            lr=self.lr,
            weight_decay=wd,
        )
