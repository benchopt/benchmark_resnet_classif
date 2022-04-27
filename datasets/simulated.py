from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import tensorflow as tf
    import torch
    from torch.utils.data import TensorDataset

    MultiFrameworkDataset = import_ctx.import_from(
        'multi_frameworks_dataset',
        'MultiFrameworkDataset',
    )


def make_channels_last(images):
    return np.transpose(images, (0, 2, 3, 1))


class Dataset(MultiFrameworkDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, img_size': [
            (128, 32),
        ],
        'framework': ['pytorch', 'tensorflow'],
    }

    test_parameters = {
        'framework': ['pytorch', 'tensorflow'],
    }

    ds_description = dict(
        n_classes=2,
    )

    def __init__(
        self,
        n_samples=10,
        img_size=32,
        train_frac=0.8,
        framework='pytorch',
        random_state=27,
        one_hot=True,
    ):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.img_size = img_size
        self.train_frac = train_frac
        self.framework = framework
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.one_hot = one_hot

    def get_np_data(self):
        n_train = int(self.n_samples * self.train_frac)
        self.ds_description['n_samples_train'] = n_train
        self.ds_description['n_samples_test'] = self.n_samples - n_train
        self.ds_description['image_width'] = self.img_size
        # inputs are channel first
        inps = self.rng.normal(
            size=(self.n_samples, 3, self.img_size, self.img_size,),
        ).astype(np.float32)
        tgts = self.rng.integers(0, 2, (self.n_samples,)).astype(np.int32)
        inps_train, inps_test = inps[:n_train], inps[n_train:]
        tgts_train, tgts_test = tgts[:n_train], tgts[n_train:]
        return inps_train, inps_test, tgts_train, tgts_test

    def get_torch_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self.get_np_data()
        dataset = TensorDataset(
            torch.Tensor(inps_train),
            torch.Tensor(tgts_train).type(torch.LongTensor),
        )
        test_dataset = TensorDataset(
            torch.Tensor(inps_test),
            torch.Tensor(tgts_test).type(torch.LongTensor),
        )

        data = dict(
            dataset=dataset,
            test_dataset=test_dataset,
            framework='pytorch',
            **self.ds_description,
        )

        return 'object', data

    def get_tf_data(self):
        inps_train, inps_test, tgts_train, tgts_test = self.get_np_data()
        if self.one_hot:
            y_train = tf.one_hot(tgts_train, 2)
            y_test = tf.one_hot(tgts_test, 2)
        else:
            y_train = tgts_train
            y_test = tgts_test
        dataset = tf.data.Dataset.from_tensor_slices(
            (make_channels_last(inps_train), y_train),
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (make_channels_last(inps_test), y_test),
        )

        data = dict(
            dataset=dataset,
            test_dataset=test_dataset,
            framework='tensorflow',
            **self.ds_description,
        )

        return 'object', data
