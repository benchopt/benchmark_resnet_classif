from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:

    BenchoptCallback = import_ctx.import_from(
        'tf_helper', 'BenchoptCallback'
    )

MAX_EPOCHS = int(1e9)


class TFSolver(BaseSolver):
    """TF base solver"""

    stopping_strategy = 'callback'

    def set_objective(self, pl_module, trainer, tf_model, tf_dataset):
        self.tf_model = tf_model
        self.tf_model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            # XXX: there might a problem here if the race is tight
            # because this will compute accuracy for each batch
            # we might need to define a custom training step with an
            # encompassing model that will not compute metrics for
            # each batch.
            metrics='accuracy',
        )
        self.tf_dataset = tf_dataset

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.tf_model)

        # Launch training
        self.tf_model.fit(
            self.tf_dataset,
            callbacks=[BenchoptCallback(callback)],
            epochs=MAX_EPOCHS,
        )

    def get_result(self):
        return self.tf_model
