from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import SGD

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """SGD solver"""
    name = 'SGD-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov, momentum': [(False, 0), (True, 0.9)],
        **TFSolver.parameters,
    }

    def set_objective(self, model, dataset):
        self.optimizer = SGD(
            learning_rate=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        super().set_objective(model, dataset)
