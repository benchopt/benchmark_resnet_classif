from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    from torchvision import transforms
    import torchvision.datasets as datasets

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


def grayscale_to_rbg_torch(image):
    return image.repeat(3, 1, 1)


def grayscale_to_rbg_tf(image):
    return tf.tile(image, [1, 1, 3])


class Dataset(MultiFrameworkDataset):

    name = "MNIST"
    parameters = {
        'debug': [True],
        'framework': ['pytorch', 'tensorflow'],
    }

    # from
    # https://stackoverflow.com/a/67233938/4332585
    normalization_mean = (0.1307,)
    normalization_std = (0.3081,)

    ds_description = dict(
        n_samples_train=60_000,
        n_samples_test=10_000,
        image_width=28,
        n_classes=10,
    )

    torch_ds_klass = datasets.MNIST

    tf_ds_name = 'mnist'

    def __init__(self, debug=True, **parameters):
        # TODO: implement subsampling for mnist debug
        super().__init__(**parameters)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
            transforms.Lambda(grayscale_to_rbg_torch),
        ])
        keras_normalization = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        self.image_preprocessing = (
            lambda x: grayscale_to_rbg_tf(keras_normalization(x/255))
        )
