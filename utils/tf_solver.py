from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import tensorflow as tf
    from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay
    BenchoptCallback = import_ctx.import_from(
        'tf_helper', 'BenchoptCallback'
    )
    LRWDSchedulerCallback = import_ctx.import_from(
        'tf_helper', 'LRWDSchedulerCallback'
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
        'lr_schedule': [None, 'step', 'cosine'],
        'decoupled_weight_decay': [0.0, 1e-4, 0.02],
        'coupled_weight_decay': [0.0, 1e-4, 0.02],
    }

    install_cmd = 'conda'
    requirements = ['tensorflow-addons']

    def skip(self, model_init_fn, dataset):
        if not isinstance(dataset, tf.data.Dataset):
            return True, 'Not a TF dataset'
        if self.coupled_weight_decay and self.decoupled_weight_decay:
            return True, 'Cannot use both decoupled and coupled weight decay'
        return False, None

    def set_objective(self, model_init_fn, dataset):
        # NOTE: in the following, we need to multiply by the weight decay
        # by the learning rate to have a comparable settign with PyTorch
        if self.lr_schedule == 'step':
            self.lr_scheduler, self.wd_scheduler = [
                tf.keras.optimizers.schedules.ExponentialDecay(
                    value,
                    decay_rate=0.1,
                    decay_steps=30,
                    staircase=True,
                ) for value in [self.lr, self.decoupled_weight_decay*self.lr]
            ]
        elif self.lr_schedule == 'cosine':
            self.lr_scheduler, self.wd_scheduler = [
                tf.keras.optimizers.schedules.CosineDecay(
                    value,
                    200,  # the equivalent of T_max
                ) for value in [self.lr, self.decoupled_weight_decay*self.lr]
            ]
        else:
            self.lr_scheduler = lambda epoch: self.lr
            self.wd_scheduler = lambda epoch: self.decoupled_weight_decay
        self.lr_wd_cback = LRWDSchedulerCallback(
            lr_schedule=self.lr_scheduler,
            wd_schedule=self.wd_scheduler,
        )
        self.optimizer_klass = extend_with_decoupled_weight_decay(
            self.optimizer_klass,
        )
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
        self.optimizer = self.optimizer_klass(
            weight_decay=self.decoupled_weight_decay*self.lr,
            **self.optimizer_kwargs,
        )
        if self.coupled_weight_decay:
            # this is equivalent to adding L2 regularization to all
            # the weights and biases of the model (even if adding
            # weight decay to the biases is not recommended), of a factor
            # halved
            l2_reg_factor = self.coupled_weight_decay / 2
            # taken from
            # https://sthalles.github.io/keras-regularizer/
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            target_regularizers = [
                'kernel_regularizer',
                'bias_regularizer',
                'beta_regularizer',
                'gamma_regularizer',
            ]
            for layer in self.model.layers:
                for attr in target_regularizers:
                    if hasattr(layer, attr):
                        setattr(layer, attr, regularizer)
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
            callbacks=[BenchoptCallback(callback), self.lr_wd_cback],
            epochs=MAX_EPOCHS,
        )

    def get_result(self):
        return self.model
