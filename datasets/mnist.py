from benchopt import safe_import_context

from benchmark_utils.multi_frameworks_dataset import MultiFrameworkDataset

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    from torchvision import transforms
    import torchvision.datasets as datasets


def grayscale_to_rbg_torch(image):
    return image.repeat(3, 1, 1)


def grayscale_to_rbg_tf(image):
    return tf.tile(image, [1, 1, 3])


class Dataset(MultiFrameworkDataset):

    name = "MNIST"

    # from
    # https://stackoverflow.com/a/67233938/4332585
    normalization_mean = (0.1307,)
    normalization_std = (0.3081,)

    ds_description = dict(
        n_samples_train=50_000,
        n_samples_val=10_000,
        n_samples_test=10_000,
        image_width=28,
        n_classes=10,
        symmetry=None,
    )

    torch_ds_klass = datasets.MNIST

    tf_ds_name = 'mnist'

    def get_torch_preprocessing_step(self):
        # Here this dataset is in gray scale so we adapt the preprocessing.
        normalization_transform = transforms.Compose([
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
            transforms.Lambda(grayscale_to_rbg_torch),
        ])
        return normalization_transform

    def get_tf_preprocessing_step(self):

        # Data preprocessing steps for grayscale
        keras_normalization = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        return lambda x: grayscale_to_rbg_tf(keras_normalization(x/255))
