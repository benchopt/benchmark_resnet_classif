from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from torch.optim import Adam, AdamW

TorchSolver = import_ctx.import_from('torch_solver', 'TorchSolver')


class Solver(TorchSolver):
    """Adam solver"""
    name = 'Adam-torch'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TorchSolver.parameters
    }

    def set_objective(self, model, dataset):
        super().set_objective(model, dataset)
        optimizer_klass = Adam
        wd = self.coupled_weight_decay
        if self.decoupled_weight_decay > 0:
            optimizer_klass = AdamW
            wd = self.decoupled_weight_decay
        optimizer = optimizer_klass(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=wd,
        )
        self.set_lr_schedule_and_optimizer(optimizer)
