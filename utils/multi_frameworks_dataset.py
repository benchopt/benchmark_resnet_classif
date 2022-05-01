from abc import ABC

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from torchvision import transforms


class MultiFrameworkDataset(BaseDataset, ABC):
    torch_split_kwarg = 'train'

    parameters = {
        'framework': ['pytorch', 'tensorflow'],
    }

    install_cmd = 'conda'
    requirements = ['pip:tensorflow-datasets']

    def get_torch_preprocessing_step(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
        ])

    def get_torch_data(self):

        # Data preprocessing steps
        transform = self.get_torch_preprocessing_step()

        # Map correct split name for torch_ds_klass
        if self.torch_split_kwarg == 'train':
            splits = [True, False]
        elif self.torch_split_kwarg == 'split':
            splits = ['train', 'test']
        else:
            raise ValueError(f"unknown split_kwargs {self.torch_split_kwarg}")

        # Load data
        data_dict = dict(framework=self.framework, **self.ds_description)
        for key, split in zip(['dataset', 'test_dataset'], splits):
            split_kwarg = {self.torch_split_kwarg: split}
            data_dict[key] = self.torch_ds_klass(
                root='./data',
                download=True,
                transform=transform,
                **split_kwarg,
            )
        return 'object', data_dict

    def get_tf_preprocessing_step(self):

        # Data preprocessing steps
        keras_normalization = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        return lambda x: keras_normalization(x/255)

    def get_tf_data(self):

        image_preprocessing = self.get_tf_preprocessing_step()

        # Load data
        data_dict = dict(framework=self.framework, **self.ds_description)
        for key, split in zip(['dataset', 'test_dataset'], ['train', 'test']):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            data_dict[key] = ds.map(
                lambda x, y: (image_preprocessing(x), y,),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        return 'object', data_dict

    def get_data(self):
        """Switch to select the data from the right framework."""
        if self.framework == 'pytorch':
            return self.get_torch_data()
        elif self.framework == 'tensorflow':
            return self.get_tf_data()
