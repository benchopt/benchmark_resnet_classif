from benchopt.utils.safe_import import set_benchmark
import numpy as np
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
    _sample = next(_loader)
    X = _sample[0]
    y = _sample[1]
    return X, y


def test_simulated_consistency():
    from datasets.simulated import Dataset
    d_tf = Dataset.get_instance(framework='tensorflow')
    d_torch = Dataset.get_instance(framework='pytorch')
    _, tf_data = d_tf.get_data()
    tf_dataset = tf_data['dataset']
    _, torch_data = d_torch.get_data()
    torch_dataset = torch_data['dataset']
    tf_np_array = tf_dataset_to_np_array(tf_dataset, d_tf.n_samples)
    torch_np_array = torch_dataset_to_np_array(
        torch_dataset,
        d_torch.n_samples,
    )
    np.testing.assert_array_equal(tf_np_array[0], torch_np_array[0])
    np.testing.assert_array_equal(tf_np_array[1], torch_np_array[1])
