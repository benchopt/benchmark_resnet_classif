from benchopt import safe_import_context

from benchmark_utils.tf_solver import TFSolver

with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import SGD


class Solver(TFSolver):
    """SGD solver"""
    name = 'SGD-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TFSolver.parameters,
        'nesterov, momentum': [(False, 0), (False, 0.9), (True, 0.9)],
        'lr': [1e-1],
        'decoupled_weight_decay': [0.0, 5e-4],
        'coupled_weight_decay': [0.0, 5e-4],
        # 'weight_decay': [0.0, 5e-4],
    }

    def set_objective(self, **kwargs):
        self.optimizer_klass = SGD
        self.optimizer_kwargs = dict(
            learning_rate=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        super().set_objective(**kwargs)
