from abc import ABC

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import torch
    from torchvision import transforms

    AugmentedDataset = import_ctx.import_from(
        'torch_helper', 'AugmentedDataset'
    )
    TFDatasetCapsule = import_ctx.import_from(
        'tf_helper', 'TFDatasetCapsule'
    )


class MultiFrameworkDataset(BaseDataset, ABC):
    torch_split_kwarg = 'train'

    def __init__(self, framework='pytorch', one_hot=True):
        # Store the parameters of the dataset
        self.framework = framework
        self.one_hot = one_hot
        self.transform = transforms.PILToTensor()
        self.normalization = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                self.normalization_mean,
                self.normalization_std,
            ),
        ])
        keras_normalization = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        self.image_preprocessing = lambda x: keras_normalization(x/255)

    def get_torch_data(self):
        data_dict = dict(framework=self.framework, **self.ds_description)
        if self.torch_split_kwarg == 'train':
            splits = [True, False]
        elif self.torch_split_kwarg == 'split':
            splits = ['train', 'test']
        for key, split in zip(['dataset', 'test_dataset'], splits):
            split_kwarg = {self.torch_split_kwarg: split}
            ds = self.torch_ds_klass(
                root='./data',
                download=True,
                transform=self.transform,
                **split_kwarg,
            )
            # XXX: maybe consider AugMixDataset from
            # https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/data/dataset.py
            data_dict[key] = AugmentedDataset(
                ds,
                None,
                self.normalization,
            )
        return 'object', data_dict

    def get_tf_data(self):
        data_dict = dict(framework=self.framework, **self.ds_description)
        for key, split in zip(['dataset', 'test_dataset'], ['train', 'test']):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            ds = ds.map(
                lambda x, y: (
                    self.image_preprocessing(x)
                    if key == 'test_dataset' else x,
                    tf.one_hot(y, self.ds_description['n_classes'])
                    if self.one_hot else y,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            data_dict[key] = ds
            if key == 'dataset':
                data_dict[key] = TFDatasetCapsule(
                    data_dict[key],
                    self.image_preprocessing,
                )
        return 'object', data_dict

    def get_data(self):
        if self.framework == 'pytorch':
            return self.get_torch_data()
        elif self.framework == 'tensorflow':
            return self.get_tf_data()
