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
    return X, y


def assert_tf_images_equal_torch_images(tf_images, torch_images):
    # make torch_images channel last
    torch_images = np.transpose(torch_images, (0, 2, 3, 1))
    np.testing.assert_array_equal(tf_images, torch_images)


@pytest.mark.parametrize('dataset_module_name', [
    'cifar',
    'mnist',
    'simulated',
    'svhn',
])
def test_datasets_consistency(dataset_module_name):
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
    for dataset_type, n_samples_key in zip(
        ['dataset', 'test_dataset'],
        ['n_samples_train', 'n_samples_test'],
    ):
        n_samples = tf_data[n_samples_key]
        assert n_samples == torch_data[n_samples_key], \
            'Number of samples is different'
        tf_dataset = tf_data[dataset_type]
        torch_dataset = torch_data[dataset_type]
        tf_np_array = tf_dataset_to_np_array(tf_dataset, n_samples)
        torch_np_array = torch_dataset_to_np_array(torch_dataset, n_samples)
        assert_tf_images_equal_torch_images(tf_np_array[0], torch_np_array[0])
        np.testing.assert_array_equal(tf_np_array[1], torch_np_array[1])
