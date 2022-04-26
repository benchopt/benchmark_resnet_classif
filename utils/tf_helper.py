from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import tensorflow as tf


# Convert benchopt benchmark into a lightning callback, used to monitor the
# objective and to stop the solver when needed.
class BenchoptCallback(tf.keras.callbacks.Callback):
    def __init__(self, callback):
        super().__init__()
        self.cb_ = callback

    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training = not self.cb_(self.model)


class TFDatasetCapsule:
    def __init__(self, dataset, image_preprocessing):
        self.dataset = dataset
        self.image_preprocessing = image_preprocessing
