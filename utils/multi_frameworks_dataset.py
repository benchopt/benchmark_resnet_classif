from abc import ABC

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from torchvision import transforms


class MultiFrameworkDataset(BaseDataset, ABC):
    def __init__(self, framework='pytorch'):
        # Store the parameters of the dataset
        self.framework = framework
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
        ])
        self.image_preprocessing = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )

    def get_torch_data(self):
        data_dict = dict(**self.ds_description)
        for key, train in zip(['dataset', 'test_dataset'], [True, False]):
            data_dict[key] = self.torch_ds_klass(
                root='./data',
                train=train,
                download=True,
                transform=self.transform,
            )
        return 'object', data_dict

    def get_tf_data(self):
        data_dict = dict(**self.ds_description)
        for key, split in zip(['dataset', 'test_dataset'], ['train', 'test']):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            ds = ds.map(
                lambda x, y: (self.image_preprocessing(x), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            data_dict[key] = ds
        return 'object', data_dict

    def get_data(self):
        if self.framework == 'pytorch':
            return self.get_torch_data()
        elif self.framework == 'tensorflow':
            return self.get_tf_data()
