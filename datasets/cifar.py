from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from torchvision import transforms
    import torchvision.datasets as datasets
    import numpy as np

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


class Dataset(MultiFrameworkDataset):

    name = "CIFAR"

    # from
    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L34
    normalization_mean = (0.4914, 0.4822, 0.4465)
    normalization_std = (0.2023, 0.1994, 0.2010)

    parameters = {
        'framework': ['pytorch', 'tensorflow'],
    }

    def get_torch_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
        ])
        cifar_trainset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform,
        )

        data = dict(dataset=cifar_trainset)

        return 'object', data

    def get_tf_data(self):
        ds = tfds.load('cifar', split='train',  as_supervised=True)
        normalization_layer = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        ds = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return 'dataset', dict(dataset=ds)
