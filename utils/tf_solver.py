from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import tensorflow as tf

    BenchoptCallback = import_ctx.import_from(
        'tf_helper', 'BenchoptCallback'
    )

MAX_EPOCHS = int(1e9)


class TFSolver(BaseSolver):
    """TF base solver"""

    stopping_criterion = SufficientProgressCriterion(
        patience=20, strategy='callback'
    )

    parameters = {
        'batch_size': [64],
        'data_aug': [False, True],
    }

    def skip(self, model_init_fn, dataset):
        if not isinstance(dataset, tf.data.Dataset):
            return True, 'Not a TF dataset'
        return False, None

    def set_objective(self, model_init_fn, dataset):
        self.dataset = dataset
        self.model_init_fn = model_init_fn

        if self.data_aug:
            data_aug_layer = tf.keras.models.Sequential([
                tf.keras.layers.ZeroPadding2D(padding=4),
                tf.keras.layers.RandomCrop(height=32, width=32),
                tf.keras.layers.RandomFlip('horizontal'),
            ])

            # XXX: unfortunately we need to do this before
            # batching since the random crop layer does not
            # crop at different locations in the same batch
            # https://github.com/keras-team/keras/issues/16399
            self.dataset = self.dataset.map(
                lambda x, y: (data_aug_layer(x[None], training=True)[0], y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        self.dataset = self.dataset.batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        self.model = self.model_init_fn()
        self.optimizer = self.optimizer_klass(**self.optimizer_kwargs)
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            # XXX: there might a problem here if the race is tight
            # because this will compute accuracy for each batch
            # we might need to define a custom training step with an
            # encompassing model that will not compute metrics for
            # each batch.
            metrics='accuracy',
        )
        # Initial evaluation
        callback(self.model)

        # Launch training
        self.model.fit(
            self.dataset,
            callbacks=[BenchoptCallback(callback)],
            epochs=MAX_EPOCHS,
        )

    def get_result(self):
        return self.model
