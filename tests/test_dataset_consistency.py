import os

import numpy as np
import pytest
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset

from benchopt.utils.safe_import import set_benchmark

# this means this test has to be run from the root
set_benchmark('.')

CI = os.environ.get('CI', False)


def tf_dataset_to_np_array(tf_dataset, n_samples):
    tf_np_array = tf_dataset.batch(n_samples).as_numpy_iterator().next()
    return tf_np_array


def torch_dataset_to_np_array(torch_dataset, n_samples):
    _loader = DataLoader(
        torch_dataset,
        batch_size=n_samples,
    )
    _sample = next(iter(_loader))
    X = _sample[0]
    y = _sample[1]
    try:
        X = X.numpy()
    except AttributeError:
        pass
    else:
        y = y.numpy()
    return X, y


def assert_tf_images_equal_torch_images(tf_images, torch_images):
    # make torch_images channel last
    torch_images = np.transpose(torch_images, (0, 2, 3, 1))
    np.testing.assert_array_almost_equal(tf_images, torch_images)


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform, normalization=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.normalization = normalization

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.normalization:
            x = self.normalization(x)
        return x, y


@pytest.mark.parametrize('dataset_module_name', [
    'cifar',
    'cifar_100',
    'mnist',
    'simulated',
    'svhn',
])
@pytest.mark.parametrize('dataset_type', [
    'dataset',
    'val_dataset',
    'test_dataset',
])
def test_datasets_consistency(dataset_module_name, dataset_type):
    if dataset_module_name == 'svhn' and dataset_type == 'dataset' and CI:
        pytest.skip('SVHN dataset is too heavy for CI')
    from datasets import (  # noqa: F401
        cifar,
        cifar_100,
        mnist,
        simulated,
        svhn,
    )
    dataset = eval(dataset_module_name)
    d_tf = dataset.Dataset.get_instance(
        framework='tensorflow',
        with_validation=True,
    )
    d_torch = dataset.Dataset.get_instance(
        framework='pytorch',
        with_validation=True,
    )
    tf_data = d_tf.get_data()
    torch_data = d_torch.get_data()

    for k in torch_data:
        if k not in ['dataset', 'val_dataset', 'test_dataset',
                     'framework', 'normalization']:
            assert torch_data[k] == tf_data[k], (
                f"ds_description do not match between framework for key {k}"
            )

    tf_dataset = tf_data[dataset_type]
    torch_dataset = torch_data[dataset_type]
    if dataset_type == 'dataset':
        tf_normalization = tf_data['normalization']
        torch_normalization = torch_data['normalization']
        tf_dataset = tf_dataset.map(
            lambda x, y: (tf_normalization(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        torch_dataset = AugmentedDataset(
            torch_dataset,
            None,
            normalization=torch_normalization,
        )

    # Conver to numpy arrays
    n_samples = len(torch_dataset)
    tf_np_array = tf_dataset_to_np_array(tf_dataset, n_samples)
    torch_np_array = torch_dataset_to_np_array(torch_dataset, n_samples)
    if dataset_type == 'test_dataset' and dataset_module_name != 'simulated':
        registration_indices = d_tf.get_registration_indices(split='test')
        X_torch = torch_np_array[0][registration_indices]
        y_torch = torch_np_array[1][registration_indices]
    else:
        X_torch = torch_np_array[0]
        y_torch = torch_np_array[1]

    # images
    assert_tf_images_equal_torch_images(tf_np_array[0], X_torch)
    # labels
    np.testing.assert_array_almost_equal(tf_np_array[1], y_torch)
