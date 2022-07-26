from pathlib import Path
import tempfile

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
            if hasattr(self.model.optimizer, 'weight_decay'):
                logs['wd'] = backend.get_value(
                    self.model.optimizer.weight_decay,
                )

    class RandomResizedCrop(tf.keras.layers.Layer):
        # taken from
        # https://keras.io/examples/vision/nnclr/#random-resized-crops
        def __init__(self, scale, ratio, crop_shape):
            super(RandomResizedCrop, self).__init__()
            self.scale = scale
            self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
            self.crop_shape = crop_shape

        @tf.function(input_signature=[1, None, None, 3], jit_compile=True)
        def get_crop_params(self, images):
            # reusing code from
            # https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L911
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
            area = tf.cast(height * width, tf.float32)
            for _ in range(10):
                target_area = area * tf.random.uniform((1,), *self.scale)
                aspect_ratio = tf.exp(tf.random.uniform((1,), *self.log_ratio))

                w = tf.cast(tf.sqrt(target_area * aspect_ratio), tf.int32)
                h = tf.cast(tf.sqrt(target_area / aspect_ratio), tf.int32)

                if 0 < w <= width and 0 < h <= height:
                    return h, w

            # Fallback to central crop
            ratio = tf.exp(self.log_ratio)
            in_ratio = float(width) / float(height)
            if in_ratio < tf.minimum(ratio):
                w = width
                h = tf.cast(w / tf.minimum(ratio), tf.int32)
            elif in_ratio > tf.maximum(ratio):
                h = height
                w = tf.cast(h * tf.maximum(ratio), tf.int32)
            else:  # whole image
                w = width
                h = height
            return h, w

        def call(self, images):
            batch_size = tf.shape(images)[0]
            tf.assert_equal(batch_size, 1)  # for now we will make this op work
            # only with one image since we select only one crop region

            h, w = self.get_crop_params(images)
            cropped_images = tf.image.random_crop(images, [1, h[0], w[0], 3])
            resized_images = tf.image.resize(cropped_images, self.crop_shape)
            return resized_images


def filter_ds_on_indices(ds, indices):
    """Filter a tensorflow dataset on a list of indices
    using a tf.lookup.StaticHashTable"""
    # from https://stackoverflow.com/a/66411957/4332585
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(indices),
            values=tf.ones_like(indices),  # Ones will be casted to True.
        ),
        default_value=0,  # If index not in table, return 0.
    )

    def hash_table_filter(index, value):
        table_value = table.lookup(index)  # 1 if index in arr, else 0.
        index_in_arr = tf.cast(table_value, tf.bool)  # 1 -> True, 0 -> False
        return index_in_arr

    ds = ds.enumerate().filter(hash_table_filter).map(
        lambda id, value: value,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def set_regularizer_model(model, regularizer):
    target_regularizers = [
        'kernel_regularizer',
        'bias_regularizer',
        'beta_regularizer',
        'gamma_regularizer',
    ]
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            set_regularizer_model(layer, regularizer)
        else:
            for attr in target_regularizers:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)


def apply_coupled_weight_decay(model, wd):
    l2_reg_factor = wd / 2
    # taken from
    # https://sthalles.github.io/keras-regularizer/
    regularizer = tf.keras.regularizers.l2(l2_reg_factor)
    set_regularizer_model(model, regularizer)
    # When we change the layers attributes, the change only happens
    #  in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = Path(tempfile.gettempdir()) / 'tmp_weights.h5'
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model
