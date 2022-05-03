from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import backend

    class BenchoptCallback(tf.keras.callbacks.Callback):
        def __init__(self, callback):
            super().__init__()
            self.cb_ = callback

        def on_epoch_end(self, epoch, logs=None):
            self.model.stop_training = not self.cb_(self.model)

    class LRWDSchedulerCallback(tf.keras.callbacks.LearningRateScheduler):
        """Callback that schedules jointly the learning rate and the weight decay

        This is necessary as in TensorFlow, the decoupled weight decay is not
        multiplied by the learning rate.
        This is mentionned in the docs:
        https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/extend_with_decoupled_weight_decay
        In this case, they mention a per-step schedule, but in order to stick
        to the PyTorch Lightning way of updating LR and WD, we stick to a
        per-epoch schedule which is conveniently done with a Callback.
        """
        def __init__(self, lr_schedule, wd_schedule, verbose=0):
            super(LRWDSchedulerCallback, self).__init__(
                schedule=lr_schedule,
                verbose=verbose,
            )
            self.wd_schedule = wd_schedule

        def on_epoch_begin(self, epoch, logs=None):
            super().on_epoch_begin(epoch, logs)
            if not hasattr(self.model.optimizer, 'weight_decay'):
                return
            try:  # new API
                wd = float(backend.get_value(
                    self.model.optimizer.weight_decay,
                ))
                wd = self.wd_schedule(epoch, wd)
            except TypeError:  # Support for old API for backward compatibility
                wd = self.wd_schedule(epoch)
            if not isinstance(wd, (tf.Tensor, float, np.float32, np.float64)):
                raise ValueError(
                    'The output of the "schedule" function '
                    f'should be float. Got: {wd}')
            if isinstance(wd, tf.Tensor) and not wd.dtype.is_floating:
                raise ValueError(
                    'The dtype of `wd` Tensor should be float. '
                    f'Got: {wd.dtype}',
                )
            backend.set_value(
                self.model.optimizer.weight_decay,
                backend.get_value(wd),
            )

        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            logs['wd'] = backend.get_value(self.model.optimizer.weight_decay)

    class BenchoptModelWrapper(tf.keras.models.Model):
        """Wraps a model with a training step that does not compute
        the extra metrics
        """
        def __init__(self, model, **kwargs):
            super().__init__(**kwargs)
            self.model = model

        def call(self, x, training=None):
            return self.model(x, training=training)

        def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    regularization_losses=self.losses,
                )

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Return a dict mapping metric names to current value
            return {m.name: np.nan for m in self.metrics}
