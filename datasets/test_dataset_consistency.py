from benchopt.utils.safe_import import set_benchmark
import numpy as np
import pytest
from torch.utils.data import DataLoader

# this means this test has to be run from the root
set_benchmark('./')


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
    np.testing.assert_array_equal(tf_images, torch_images)


def order_images_labels(images, labels):
    images_mean = np.mean(images, axis=(1, 2, 3))
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
    n_samples_key = 'n_samples_train' if dataset_type == 'dataset' \
        else 'n_samples_test'
    n_samples = tf_data[n_samples_key]
    assert n_samples == torch_data[n_samples_key], \
        'Number of samples is different'
    tf_dataset = tf_data[dataset_type]
    torch_dataset = torch_data[dataset_type]
    tf_np_array = tf_dataset_to_np_array(tf_dataset, n_samples)
    X_tf, y_tf = order_images_labels(*tf_np_array)
    torch_np_array = torch_dataset_to_np_array(torch_dataset, n_samples)
    X_torch, y_torch = order_images_labels(*torch_np_array)
    assert_tf_images_equal_torch_images(X_tf, X_torch)
    np.testing.assert_array_equal(y_tf, y_torch)
