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
        **TorchSolver.parameters
    }

    def set_objective(self, model, dataset):
        super().set_objective(model, dataset)
        optimizer = SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )
        self.set_lr_schedule_and_optimizer(optimizer)
