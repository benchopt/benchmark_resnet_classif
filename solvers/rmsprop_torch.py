from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import RMSprop

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """RMSPROP solver"""
    name = 'RMSPROP-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
        'rho': [0.99, 0.9],
        'momentum': [0, 0.9],
        **TorchSolver.parameters
    }

    def skip(self, model_init_fn, dataset):
        if self.decoupled_weight_decay:
            return True, 'RMSProp does not support decoupled weight decay'
        return super().skip(model_init_fn, dataset)

    def set_objective(self, **kwargs):
        super().set_objective(**kwargs)
        self.optimizer_klass = RMSprop
        self.optimizer_kwargs = dict(
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.rho,
            weight_decay=self.coupled_weight_decay,
        )
