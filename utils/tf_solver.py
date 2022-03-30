from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:

    import tensorflow as tf
    BenchoptCallback = import_ctx.import_from(
        'tf_helper', 'BenchoptCallback'
    )


class TFSolver(BaseSolver):
    """TF base solver"""

    stopping_strategy = 'callback'

    def set_objective(self, pl_module, trainer, tf_model, tf_dataset):
        self.tf_model = tf_model
        self.tf_model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
        )
        self.tf_dataset = tf_dataset

    def run(self, callback):
        # Initial evaluation
        callback(self.tf_model)

        # Launch training
        self.tf_model.fit(self.tf_dataset, callbacks=[BenchoptCallback(callback)])


    def get_result(self):
        return None, self.tf_model
