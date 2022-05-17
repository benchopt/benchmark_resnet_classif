from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import SGD

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """SGD solver"""
    name = 'SGD-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov, momentum': [(False, 0), (False, 0.9), (True, 0.9)],
        'lr': [1e-1],
        'decoupled_weight_decay': [0.0, 5e-4],
        'coupled_weight_decay': [0.0, 5e-4],
        'steps': [1/2, 3/4],
        'gamma': [0.1],
        # 'weight_decay': [0.0, 5e-4],
        **TFSolver.parameters,
    }

    def set_objective(self, **kwargs):
        self.optimizer_klass = SGD
        self.optimizer_kwargs = dict(
            learning_rate=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        super().set_objective(**kwargs)
