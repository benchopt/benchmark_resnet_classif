from benchopt import safe_import_context

with safe_import_context() as import_ctx:

    import numpy as np
    import tensorflow as tf
    from torch.utils.data import DataLoader


# Convert benchopt benchmark into a lightning callback, used to monitor the
# objective and to stop the solver when needed.
class BenchoptCallback(tf.keras.callbacks.Callback):
    def __init__(self, callback):
        super().__init__()
        self.cb_ = callback

    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training = not self.cb_(self.model)


def torch_image_dataset_to_tf_dataset(torch_dataset):
    try:
        X = torch_dataset.data
    except AttributeError:
        _loader = DataLoader(
            torch_dataset,
            batch_size=len(torch_dataset),
        )
        _sample = next(iter(_loader))
        X = _sample[0]
        y = _sample[1]
    else:
        try:
            y = torch_dataset.targets
        except AttributeError:
            y = torch_dataset.labels
    try:
        X = X.numpy()
    except AttributeError:
        pass
    else:
        y = y.numpy()
    if X.shape[1] in [1, 3]:
        # reshape X from NCHW to NHWC
        X = np.transpose(X, (0, 2, 3, 1))
    width = X.shape[1]
    n_classes = len(np.unique(y))
    if not isinstance(y[0], np.ndarray) or not len(y) > 1:
        y = tf.one_hot(y, n_classes)
    tf_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return tf_dataset, width, n_classes
