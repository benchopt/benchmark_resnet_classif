from abc import ABC
from pathlib import Path

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import torch
    from torchvision import transforms
    from torch.utils.data import random_split, Subset

    AugmentedDataset = import_ctx.import_from(
        "lightning_helper", "AugmentedDataset"
    )
    filter_ds_on_indices = import_ctx.import_from(
        "tf_helper", "filter_ds_on_indices"
    )


class MultiFrameworkDataset(BaseDataset, ABC):
    torch_split_kwarg = "train"

    parameters = {
        # WARNING: this order is very important
        # as tensorflow takes all the memory and doesn't have a mechanism to
        # release it
        'framework': ['pytorch', 'lightning', 'tensorflow'],
        'random_state': [42]
    }

    install_cmd = "conda"
    requirements = ["pip:tensorflow-datasets", "scikit-learn"]

    def get_registration_indices(self, split='train'):
        registration_dir = Path() / "torch_tf_datasets_registrations"
        filepath = registration_dir / f"{self.tf_ds_name}_{split}.npy"
        if filepath.exists():
            return np.load(filepath)
        print('Registration file not found')
        return None

    def set_train_val_indices(self):
        """Train/Val split with cross framework compat."""
        registration_indices = self.get_registration_indices()
        if registration_indices is None:
            self.train_val_split_spec = False
        self.tf_train_indices, self.tf_val_indices = train_test_split(
            np.arange(len(registration_indices)),
            test_size=self.ds_description["n_samples_val"],
            train_size=self.ds_description["n_samples_train"],
            random_state=self.random_state,
        )
        self.tf_train_indices = np.sort(self.tf_train_indices)
        self.tf_val_indices = np.sort(self.tf_val_indices)
        self.torch_train_indices = registration_indices[self.tf_train_indices]
        self.torch_val_indices = registration_indices[self.tf_val_indices]
        self.train_val_split_spec = True

    def get_torch_preprocessing_step(self):
        normalization_transform = transforms.Normalize(
            self.normalization_mean,
            self.normalization_std,
        )
        return normalization_transform

    def get_torch_data(self):

        # Data preprocessing steps
        normalization_transform = self.get_torch_preprocessing_step()

        # Map correct split name for torch_ds_klass
        if self.torch_split_kwarg == "train":
            splits = [True, False]
        elif self.torch_split_kwarg == "split":
            splits = ["train", "test"]
        else:
            raise ValueError(f"unknown split_kwargs {self.torch_split_kwarg}")

        # Load data
        data_dict = dict(
            framework=self.framework,
            normalization=normalization_transform,
            **self.ds_description,
        )
        for key, split in zip(["dataset", "test_dataset"], splits):
            split_kwarg = {self.torch_split_kwarg: split}
            transform_list = [transforms.ToTensor()]
            if key != "dataset":
                transform_list.append(normalization_transform)
            transform = transforms.Compose(transform_list)
            data_dict[key] = self.torch_ds_klass(
                root="./data",
                download=True,
                transform=transform,
                **split_kwarg,
            )
        train_idx, val_idx = self.get_train_val_indices()
        train_dataset = Subset(data_dict["dataset"], train_idx)
        val_dataset = Subset(data_dict["dataset"], val_idx)
        data_dict["dataset"] = train_dataset
        data_dict["val_dataset"] = AugmentedDataset(
            val_dataset, None, normalization_transform
        )

        return data_dict

    def get_tf_preprocessing_step(self):

        # Data preprocessing steps
        keras_normalization = tf.keras.layers.Normalization(
            mean=self.normalization_mean,
            variance=np.square(self.normalization_std),
        )
        return lambda x: keras_normalization(x / 255)

    def get_tf_data(self):
        image_preprocessing = self.get_tf_preprocessing_step()

        # Load data
        data_dict = dict(
            framework=self.framework,
            normalization=image_preprocessing,
            **self.ds_description,
        )
        splits = ["test"]
        datasets = ["test_dataset"]
        if self.train_val_split_spec:
            splits.append('train')
            datasets.append('dataset')
        else:
            splits += [
                f'train[:{self.ds_description["n_samples_train"]}]',
                f'train[{self.ds_description["n_samples_train"]}:]',
            ]
            datasets += ["dataset", "val_dataset"]
        for key, split in zip(datasets, splits):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            if key != "dataset":
                ds = ds.map(
                    lambda x, y: (
                        image_preprocessing(x),
                        y,
                    ),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            data_dict[key] = ds
        
        train_idx, val_idx = self.get_train_val_indices()
        data_dict["val_dataset"] = filter_ds_on_indices(
            data_dict["dataset"], val_idx,
        ).map(
            lambda x, y: (
                image_preprocessing(x),
                y,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        data_dict["dataset"] = filter_ds_on_indices(
            data_dict["dataset"], train_idx
        )
        return data_dict

    def get_data(self):
        """Switch to select the data from the right framework."""
        if self.framework in ['pytorch', 'lightning']:
            return self.get_torch_data()
        elif self.framework == "tensorflow":
            return self.get_tf_data()
