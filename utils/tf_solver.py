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
        patience=50, strategy='callback'
    )

    parameters = {
        'batch_size': [128],
        'data_aug': [False, True],
        'lr_schedule': [None, 'step', 'cosine'],
    }

    install_cmd = 'conda'
    requirements = ['pip:tensorflow-addons']

    def skip(
        self,
        model_init_fn,
        dataset,
        normalization,
        framework,
        symmetry,
        image_width,
    ):
        if framework != 'tensorflow':
            return True, 'Not a TF dataset/objective'
        coupled_wd = getattr(self, 'coupled_weight_decay', 0.0)
        decoupled_wd = getattr(self, 'decoupled_weight_decay', 0.0)
        if coupled_wd and decoupled_wd:
            return True, 'Cannot use both decoupled and coupled weight decay'
        return False, None

    def get_lr_wd_cback(self, max_epochs=200):
        # NOTE: in the following, we need to multiply by the weight decay
        # by the learning rate to have a comparable setting with PyTorch
        self.coupled_wd = getattr(self, 'coupled_weight_decay', 0.0)
        self.decoupled_wd = getattr(self, 'decoupled_weight_decay', 0.0)
        if self.decoupled_wd == 0.0:
            self.decoupled_wd = getattr(self, 'weight_decay', 0.0)
        if self.lr_schedule == 'step':
            self.lr_scheduler, self.wd_scheduler = [
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                    [max_epochs//2, max_epochs*3//4],
                    [value, value*1e-1, value*1e-2],
                ) for value in [self.lr, self.decoupled_wd*self.lr]
            ]
        elif self.lr_schedule == 'cosine':
            self.lr_scheduler, self.wd_scheduler = [
                tf.keras.optimizers.schedules.CosineDecay(
                    value,
                    max_epochs,  # the equivalent of T_max
                ) for value in [self.lr, self.decoupled_wd*self.lr]
            ]
        else:
            self.lr_scheduler = lambda epoch: self.lr
            self.wd_scheduler = lambda epoch: self.decoupled_wd * self.lr

        # we set the decoupled weight decay always, and when it's 0
        # the WD cback and the decoupled weight decay extension are
        # essentially no-ops
        lr_wd_cback = LRWDSchedulerCallback(
            lr_schedule=self.lr_scheduler,
            wd_schedule=self.wd_scheduler,
        )
        return lr_wd_cback

    def set_objective(
        self,
        model_init_fn,
        dataset,
        normalization,
        framework,
        symmetry,
        image_width,
    ):
        self.optimizer_klass = extend_with_decoupled_weight_decay(
            self.optimizer_klass,
        )
        self.dataset = dataset
        self.model_init_fn = model_init_fn
        self.framework = framework
        self.symmetry = symmetry
        self.image_width = image_width

        if self.data_aug:
            data_aug_list = [
                tf.keras.layers.ZeroPadding2D(padding=4),
                tf.keras.layers.RandomCrop(
                    height=self.image_width,
                    width=self.image_width,
                ),
            ]
            if self.symmetry is not None and 'horizontal' in self.symmetry:
                data_aug_list.append(tf.keras.layers.RandomFlip('horizontal'))
            data_aug_layer = tf.keras.models.Sequential(data_aug_list)

            # XXX: unfortunately we need to do this before
            # batching since the random crop layer does not
            # crop at different locations in the same batch
            # https://github.com/keras-team/keras/issues/16399
            self.dataset = self.dataset.map(
                lambda x, y: (data_aug_layer(x[None], training=True)[0], y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        self.dataset = self.dataset.shuffle(
            buffer_size=1000,  # For now a hardcoded value
            reshuffle_each_iteration=True,
        ).batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        if normalization is not None:
            self.dataset = self.dataset.map(
                lambda x, y: (normalization(x), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        self.dataset = self.dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, callback):
        self.model = self.model_init_fn()
        max_epochs = callback.stopping_criterion.max_runs
        lr_wd_cback = self.get_lr_wd_cback(max_epochs)
        self.optimizer = self.optimizer_klass(
            # this scaling is needed as in TF the weight decay is
            # not multiplied by the learning rate
            weight_decay=self.decoupled_wd*self.lr,
            **self.optimizer_kwargs,
        )
        if self.coupled_wd:
            # this is equivalent to adding L2 regularization to all
            # the weights and biases of the model (even if adding
            # weight decay to the biases is not recommended), of a factor
            # halved
            l2_reg_factor = self.coupled_wd / 2
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

        cback_list = tf.keras.callbacks.CallbackList(
            [BenchoptCallback(callback), lr_wd_cback],
            model=self.model,
        )
        # It's important to create the callback list ourselves in order
        # to avoid the overhead of having to store a history of the
        # training and using a progressbar

        # Initial evaluation
        callback(self.model)
        # Launch training
        self.model.fit(
            self.dataset,
            callbacks=cback_list,
            epochs=MAX_EPOCHS,
            verbose=0,
        )

    def get_result(self):
        return self.model
