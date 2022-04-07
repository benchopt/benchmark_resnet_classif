from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import RMSprop

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """RMSProp solver"""
    name = 'RMSProp-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
        'momentum': [0, 0.9],
        'rho': [0.99, 0.9],
    }

    def set_objective(self, pl_module, trainer, tf_model, tf_dataset):
        self.optimizer = RMSprop(
            learning_rate=self.lr,
            momentum=self.momentum,
            rho=self.rho,
        )
        super().set_objective(pl_module, trainer, tf_model, tf_dataset)
