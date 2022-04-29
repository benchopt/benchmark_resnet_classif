import os
import warnings

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

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


def order_images_labels(images, labels):
    images_mean = np.mean(images, axis=(1, 2, 3), dtype=np.float64)
    images_ordering = np.argsort(images_mean)
    images_ordered = images[images_ordering]
    labels_ordered = labels[images_ordering]
    return images_ordered, labels_ordered


@pytest.mark.parametrize('dataset_module_name', [
    'cifar',
    'mnist',
    'simulated',
    'svhn',
])
@pytest.mark.parametrize('dataset_type', ['dataset', 'test_dataset'])
def test_datasets_consistency(dataset_module_name, dataset_type):
    if dataset_module_name == 'svhn' and dataset_type == 'dataset' and CI:
        pytest.skip('SVHN dataset is too heavy for CI')
    from datasets import (  # noqa: F401
        cifar,
        mnist,
        simulated,
        svhn,
    )
    dataset = eval(dataset_module_name)
    d_tf = dataset.Dataset.get_instance(framework='tensorflow')
    d_torch = dataset.Dataset.get_instance(framework='pytorch')
    _, tf_data = d_tf.get_data()
    _, torch_data = d_torch.get_data()

    for k in torch_data:
        if k not in ['dataset', 'test_dataset', 'framework']:
            assert torch_data[k] == tf_data[k], (
                f"ds_description do not match between framework for key {k}"
            )

    tf_dataset = tf_data[dataset_type]
    torch_dataset = torch_data[dataset_type]
    assert len(tf_dataset) == len(torch_dataset), (
        "len of the 2 datsets do not match"
    )

    # Convert to numpy arrays
    n_samples = len(torch_dataset)
    tf_np_array = tf_dataset_to_np_array(tf_dataset, n_samples)
    X_tf, y_tf = order_images_labels(*tf_np_array)
    torch_np_array = torch_dataset_to_np_array(torch_dataset, n_samples)
    X_torch, y_torch = order_images_labels(*torch_np_array)

    try:
        assert_tf_images_equal_torch_images(X_tf, X_torch)
    except AssertionError:
        clf = NearestNeighbors(n_neighbors=1)
        X_torch_channel_last = np.transpose(X_torch, (0, 2, 3, 1))
        clf.fit(X_tf.reshape(X_tf.shape[0], -1))
        indices = clf.kneighbors(
            X_torch_channel_last.reshape(X_torch_channel_last.shape[0], -1),
            return_distance=False,
        )
        ordered_X_torch = X_torch[indices]
        assert_tf_images_equal_torch_images(X_tf, ordered_X_torch)
        y_torch = y_torch[indices]
    finally:
        if dataset_module_name != 'svhn' or dataset_type == 'test_dataset':
            np.testing.assert_array_equal(y_tf, y_torch)
        else:
            warnings.warn(
                'Label equality test not carried for SVHN '
                'because of a weird duplicate with 2 different '
                'labels',
            )
