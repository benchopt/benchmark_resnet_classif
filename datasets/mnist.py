from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from torch.utils.data import Subset
    from torchvision import transforms
    import torchvision.datasets as datasets
    import numpy as np

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

    def get_torch_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
            transforms.Lambda(grayscale_to_rbg_torch),
        ])
        mnist_trainset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )
        if self.debug:
            mnist_trainset = Subset(mnist_trainset, range(1000))

        return 'object', dict(dataset=mnist_trainset)

    def get_tf_data(self):
        ds = tfds.load('mnist', split='train',  as_supervised=True)
        normalization_layer = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        ds = ds.map(
            lambda x, y: (grayscale_to_rbg_tf(normalization_layer(x)), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        if self.debug:
            ds = ds.take(1000)

        return 'dataset', dict(dataset=ds)
