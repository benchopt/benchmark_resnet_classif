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

    name = "SVHN"

    # from
    # https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/_modules/deepobs/pytorch/datasets/svhn.html
    normalization_mean = (0.4376821, 0.4437697, 0.47280442)
    normalization_std = (0.19803012, 0.20101562, 0.19703614)

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
        svhn_trainset = datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transform,
        )

        return 'object', dict(dataset=svhn_trainset)

    def get_tf_data(self):
        ds = tfds.load('svhn_cropped', split='train',  as_supervised=True)
        normalization_layer = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        ds = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return 'dataset', dict(dataset=ds)
