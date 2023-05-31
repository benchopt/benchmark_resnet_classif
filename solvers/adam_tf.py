from benchopt import safe_import_context

from benchmark_utils.tf_solver import TFSolver

with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers.legacy import Adam


class Solver(TFSolver):
    """Adam solver"""
    name = 'Adam-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TFSolver.parameters,
        'lr': [1e-3],
        'decoupled_weight_decay': [0.0, 0.02],
        'coupled_weight_decay': [0.0, 0.02],
    }

    def set_objective(self, **kwargs):
        self.optimizer_klass = Adam
        self.optimizer_kwargs = dict(learning_rate=self.lr, epsilon=1e-8)
        super().set_objective(**kwargs)
