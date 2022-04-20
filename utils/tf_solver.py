from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import tensorflow as tf
    BenchoptCallback = import_ctx.import_from(
        'tf_helper', 'BenchoptCallback'
    )

MAX_EPOCHS = int(1e9)


class TFSolver(BaseSolver):
    """TF base solver"""

    stopping_strategy = 'callback'

    parameters = {
        'lr': [1e-3],
        'batch_size': [64],
        'data_aug': [False, True],
        'lr_schedule': [None, 'step', 'cosine'],
    }

    def __init__(self, **parameters):
        self.data_aug_layer = tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=4),
            tf.keras.layers.RandomCrop(height=32, width=32),
            tf.keras.layers.RandomFlip('horizontal'),
        ])
        if self.lr_schedule == 'step':
            self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr,
                decay_rate=0.1,
                decay_steps=30,
                staircase=True,
            )
        elif self.lr_schedule == 'cosine':
            self.lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                self.lr,
                200,  # the equivalent of T_max
            )
        else:
            self.lr_scheduler = lambda epoch: self.lr
        self.lr_cback = tf.keras.callbacks.LearningRateScheduler(
            self.lr_scheduler,
        )

    def skip(self, model, dataset):
        if not isinstance(model, tf.keras.Model):
            return True, 'Not a TF dataset'
        return False, None

    def set_objective(self, model, dataset):
        self.tf_model = model
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
        self.tf_dataset = dataset
        if self.data_aug:
            # XXX: unfortunately we need to do this before
            # batching since the random crop layer does not
            # crop at different locations in the same batch
            # https://github.com/keras-team/keras/issues/16399
            self.tf_dataset = self.tf_dataset.map(
                lambda x, y: (
                    self.data_aug_layer(x[None], training=True)[0],
                    y,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        self.tf_dataset = self.tf_dataset.batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        # Initial evaluation
        callback(self.tf_model)

        # Launch training
        self.tf_model.fit(
            self.tf_dataset,
            callbacks=[BenchoptCallback(callback), self.lr_cback],
            epochs=MAX_EPOCHS,
        )

    def get_result(self):
        return self.tf_model
