from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import numpy as np
    import tensorflow as tf
    from keras.utils import io_utils
    from tensorflow.keras import backend

    class BenchoptCallback(tf.keras.callbacks.Callback):
        def __init__(self, callback):
            super().__init__()
            self.cb_ = callback

        def on_epoch_end(self, epoch, logs=None):
            self.model.stop_training = not self.cb_(self.model)

    class LRWDSchedulerCallback(tf.keras.callbacks.LearningRateScheduler):
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
                wd = self.schedule(epoch, wd)
            except TypeError:  # Support for old API for backward compatibility
                wd = self.schedule(epoch)
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
            if self.verbose > 0:
                io_utils.print_msg(
                    f'\nEpoch {epoch + 1}: LearningRateScheduler'
                    ' setting learning '
                    f'rate to {wd}.')

        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            logs['wd'] = backend.get_value(self.model.optimizer.weight_decay)
