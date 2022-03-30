from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import RMSProp

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """RMSProp solver"""
    name = 'RMSProp-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'nesterov, momentum': [(False, 0), (True, 0.9)],
        'lr': [1e-3],
    }

    def set_objective(self, pl_module, trainer, tf_model, tf_dataset):
        self.optimizer = RMSProp(
            learning_rate=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        super().set_objective(pl_module, trainer, tf_model, tf_dataset)
