from abc import ABC
from pathlib import Path

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from torchvision import transforms
    from torch.utils.data import Subset

    AugmentedDataset = import_ctx.import_from(
        "lightning_helper", "AugmentedDataset"
    )
    filter_ds_on_indices = import_ctx.import_from(
        "tf_helper", "filter_ds_on_indices"
    )


class MultiFrameworkDataset(BaseDataset, ABC):
    torch_split_kwarg = "train"
    torch_dl = True
    extra_torch_test_transforms = None

    tf_test_image_processing = None
    tf_splits = ['test', 'train']

    parameters = {
        # WARNING: this order is very important
        # as tensorflow takes all the memory and doesn't have a mechanism to
        # release it
        'framework': ['pytorch', 'lightning', 'tensorflow'],
        'random_state': [42, 43, 44, 45, 46],
        'with_validation': [True, False],
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

    def get_train_val_indices(self):
        """Train/Val split with cross framework compat."""
        registration_indices = self.get_registration_indices()

        tf_train_indices, tf_val_indices = train_test_split(
            np.arange(len(registration_indices)),
            test_size=self.ds_description["n_samples_val"],
            train_size=self.ds_description["n_samples_train"],
            random_state=self.random_state,
        )
        tf_train_indices.sort(), tf_val_indices.sort()
        if self.framework == "tensorflow" or registration_indices is None:
            return tf_train_indices, tf_val_indices

        torch_train_indices = registration_indices[tf_train_indices]
        torch_val_indices = registration_indices[tf_val_indices]
        return torch_train_indices, torch_val_indices

    def get_torch_preprocessing_step(self):
        normalization_transform = transforms.Normalize(
            self.normalization_mean,
            self.normalization_std,
        )
        return normalization_transform

    def get_torch_splits(self):
        return ["train", "test"]

    def get_torch_data(self):

        # Data preprocessing steps
        normalization_transform = self.get_torch_preprocessing_step()

        # Map correct split name for torch_ds_klass
        if self.torch_split_kwarg == "train":
            splits = [True, False]
        elif self.torch_split_kwarg == "split":
            splits = self.get_torch_splits()
        else:
            raise ValueError(f"unknown split_kwargs {self.torch_split_kwarg}")

        # Load data
        data_dict = dict(
            framework=self.framework,
            normalization=normalization_transform,
            extra_test_transform=transforms.Compose(
                self.extra_torch_test_transforms,
            ) if self.extra_torch_test_transforms else None,
            **self.ds_description,
        )
        for key, split in zip(["dataset", "test_dataset"], splits):
            kwargs = {self.torch_split_kwarg: split}
            if self.torch_dl:
                kwargs["download"] = True
            transform_list = []
            if key != "dataset":
                if self.extra_torch_test_transforms is not None:
                    transform_list = self.extra_torch_test_transforms
                transform_list.append(transforms.ToTensor())
                transform_list.append(normalization_transform)
            transform = transforms.Compose(transform_list)
            data_dict[key] = self.torch_ds_klass(
                root="./data",
                transform=transform,
                **kwargs,
            )
        if self.with_validation:
            train_idx, val_idx = self.get_train_val_indices()
            train_dataset = Subset(data_dict["dataset"], train_idx)
            val_dataset = Subset(data_dict["dataset"], val_idx)
            data_dict["dataset"] = train_dataset
            data_dict["val_dataset"] = AugmentedDataset(
                val_dataset, None, normalization_transform
            )
        else:
            data_dict["val_dataset"] = None

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
            extra_test_transform=self.tf_test_image_processing,
            **self.ds_description,
        )
        datasets = ["test_dataset", "dataset"]
        for key, split in zip(datasets, self.tf_splits):
            ds = tfds.load(
                self.tf_ds_name,
                split=split,
                as_supervised=True,
            )
            if key != "dataset":
                if self.tf_test_image_processing is None:
                    test_image_processing = image_preprocessing
                else:
                    test_image_processing = self.tf_test_image_processing
                ds = ds.map(
                    lambda x, y: (
                        test_image_processing(x),
                        y,
                    ),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
            data_dict[key] = ds

        if self.with_validation:
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
        else:
            data_dict["val_dataset"] = None
        return data_dict

    def get_data(self):
        """Switch to select the data from the right framework."""
        if self.framework in ['pytorch', 'lightning']:
            return self.get_torch_data()
        elif self.framework == "tensorflow":
            return self.get_tf_data()
