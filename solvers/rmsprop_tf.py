from benchopt import safe_import_context

from benchmark_utils.tf_solver import TFSolver

with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import RMSprop


class Solver(TFSolver):
    """RMSProp solver"""
    name = 'RMSProp-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        **TFSolver.parameters,
        'lr': [1e-3],
        'rho': [0.99, 0.9],
        'momentum': [0, 0.9],
        'decoupled_weight_decay': [0.0, 1e-4, 0.02],
        'coupled_weight_decay': [0.0, 1e-4, 0.02],
    }

    def set_objective(self, **kwargs):
        self.optimizer_klass = RMSprop
        self.optimizer_kwargs = dict(
            learning_rate=self.lr,
            momentum=self.momentum,
            rho=self.rho,
        )
        super().set_objective(**kwargs)
