from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import Adam, AdamW

LightningSolver = import_ctx.import_from('lightning_solver', 'LightningSolver')


class Solver(LightningSolver):
    """Adam solver"""
    name = 'Adam-lightning'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **LightningSolver.parameters,
        'lr': [1e-3],
        'coupled_weight_decay': [0.0, 0.02],
        'decoupled_weight_decay': [0.0, 0.02],
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
