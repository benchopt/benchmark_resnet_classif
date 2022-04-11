from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from tensorflow.keras.optimizers import Adam

TFSolver = import_ctx.import_from('tf_solver', 'TFSolver')


class Solver(TFSolver):
    """Adam solver"""
    name = 'Adam-tf'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'lr': [1e-3],
    }

    def set_objective(self, model, dataset):
        self.optimizer = Adam(learning_rate=self.lr)
        super().set_objective(model, dataset)
