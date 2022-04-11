from benchopt.utils.safe_import import set_benchmark
import numpy as np
import pytest
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    np.testing.assert_array_almost_equal(tf_images, torch_images)


def order_images_labels(images, labels):
    images_mean = np.mean(images, axis=(1, 2, 3), dtype=np.float64)
    images_ordering = np.argsort(images_mean)
    images_ordered = images[images_ordering]
    labels_ordered = labels[images_ordering]
    return images_ordered, labels_ordered


@pytest.mark.parametrize('dataset_module_name', [
    # 'cifar',
    'mnist',
    # 'simulated',
    # 'svhn',
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
    try:
        assert_tf_images_equal_torch_images(X_tf, X_torch)
    except AssertionError:
        # TODO: refactor all this BS
        # easy cases where there is a correct ordering
        X_torch_channel_last = np.transpose(X_torch, (0, 2, 3, 1))
        diff = X_tf - X_torch_channel_last
        close = np.abs(diff) <= 1.5e-6
        close = np.all(close, axis=(1, 2, 3))
        matched_indices = np.where(close)[0]
        np.testing.assert_array_equal(y_tf[matched_indices], y_torch[matched_indices])
        unmatched_tf_indices = np.where(~close)[0]
        unmatched_torch_indices = list(np.copy(unmatched_tf_indices))
        y_tf = y_tf[unmatched_tf_indices]
        y_torch = y_torch[unmatched_torch_indices]
        print('Total N', len(X_tf))
        print('N unmatched', len(unmatched_torch_indices))

        # same as above but with a stride of 1
        X_tf = X_tf[unmatched_tf_indices]
        X_torch_channel_last = X_torch_channel_last[unmatched_torch_indices]
        diff_strided = X_tf[:-1] - X_torch_channel_last[1:]
        close = np.abs(diff_strided) <= 1.5e-6
        close = np.all(close, axis=(1, 2, 3))
        matched_indices_tf = np.where(close)[0]
        matched_indices_torch = matched_indices_tf + 1
        np.testing.assert_array_equal(
            y_tf[matched_indices_tf],
            y_torch[matched_indices_torch],
        )
        unmatched_tf_indices = list(np.where(~close)[0])
        unmatched_torch_indices = list(np.copy(unmatched_tf_indices) + 1)
        unmatched_tf_indices.append(len(X_tf) - 1)
        unmatched_torch_indices = [0] + unmatched_torch_indices
        y_tf = y_tf[unmatched_tf_indices]
        y_torch = y_torch[unmatched_torch_indices]
        print('N unmatched', len(unmatched_torch_indices))

        # redo no difference
        X_tf = X_tf[unmatched_tf_indices]
        X_torch_channel_last = X_torch_channel_last[unmatched_torch_indices]
        diff = X_tf - X_torch_channel_last
        close = np.abs(diff) <= 1.5e-6
        close = np.all(close, axis=(1, 2, 3))
        matched_indices = np.where(close)[0]
        np.testing.assert_array_equal(
            y_tf[matched_indices],
            y_torch[matched_indices],
        )
        unmatched_tf_indices = np.where(~close)[0]
        unmatched_torch_indices = list(np.copy(unmatched_tf_indices))
        y_tf = y_tf[unmatched_tf_indices]
        print('N unmatched', len(unmatched_torch_indices))


        for i, tf_image in tqdm(enumerate(X_tf[unmatched_tf_indices])):
            diff = np.abs(X_torch_channel_last[unmatched_torch_indices[:10]] - tf_image)
            total_diff = np.sum(diff, axis=(1, 2, 3))
            candidate_indices = np.where(total_diff < 1)[0]
            is_matched = [
                np.allclose(
                    X_torch_channel_last[unmatched_torch_indices][candidate_index],
                    tf_image,
                    rtol=0,
                    atol=1.5e-6,
                )
                for candidate_index in candidate_indices
            ]
            one_close = np.any(is_matched)
            assert one_close, 'Image is not close'
            matched_torch_index = candidate_indices[is_matched.index(True)]
            assert y_tf[i] == y_torch[unmatched_torch_indices][matched_torch_index]
            unmatched_torch_indices.pop(matched_torch_index)
    else:
        np.testing.assert_array_equal(y_tf, y_torch)
