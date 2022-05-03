from abc import ABC

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import torch
    from torchvision import transforms
    from torch.utils.data import random_split

    AugmentedDataset = import_ctx.import_from("torch_helper", "AugmentedDataset")


class MultiFrameworkDataset(BaseDataset, ABC):
    torch_split_kwarg = "train"

    parameters = {
        "framework": ["pytorch", "tensorflow"],
    }

    install_cmd = "conda"
    requirements = ["pip:tensorflow-datasets"]

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
            if key == "test_dataset":
                transform_list.append(normalization_transform)
            transform = transforms.Compose(transform_list)
            data_dict[key] = self.torch_ds_klass(
                root="./data",
                download=True,
                transform=transform,
                **split_kwarg,
            )
        train_dataset, val_dataset = random_split(
            data_dict["dataset"],
            [
                self.ds_description["n_samples_train"],
                self.ds_description["n_samples_val"],
            ],
            generator=torch.Generator().manual_seed(self.random_state),
        )
        data_dict["dataset"] = train_dataset
        data_dict["val_dataset"] = AugmentedDataset(
            val_dataset, None, normalization_transform
        )

        return "object", data_dict

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
        splits = [
            f'train[:{self.ds_description["n_samples_train"]}]',
            f'train[{self.ds_description["n_samples_train"]}:]',
            "test",
        ]
        for key, split in zip(["dataset", "val_dataset", "test_dataset"], splits):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            if key == "test_dataset":
                ds = ds.map(
                    lambda x, y: (
                        image_preprocessing(x),
                        y,
                    ),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            data_dict[key] = ds
        return "object", data_dict

    def get_data(self):
        self.random_state = 42  # Hackish
        """Switch to select the data from the right framework."""
        if self.framework == "pytorch":
            return self.get_torch_data()
        elif self.framework == "tensorflow":
            return self.get_tf_data()
